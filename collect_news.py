import os
import time
import json
import random
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
from datetime import datetime, timedelta
from gdeltdoc import GdeltDoc, Filters

# ---------- CONFIG ----------
keywords = ["artificial intelligence", "Google Gemini", 
        "Anthropic","OpenAI","Grok","xAI","Meta llama", "LLM",
        "ChatGPT", "Claude", "Bard","Perplexity AI", "AI chatbot",
        "Meta AI","DeepSeek","large language model","generative AI",
        "AI agent", "AI assistant", "machine learning"][13:]
domains = [
    "nbcnews.com", "yahoo.com", "cnbc.com", "cbsnews.com",
    "reuters.com", "npr.org", "nytimes.com", "forbes.com", "cnn.com",
    "msn.com", "tmz.com", "bloomberg.com", "buzzfeed.com", "today.com",
    "foxnews.com", "politico.com"
]
START = "2025-01-01"
END   = "2025-12-31"
RECORDS_PER_QUERY = 250
BASE_SLEEP_SECONDS = 1.4
OUTPUT_CSV = "gdelt_ai_narratives_EN_US.csv"  # new file name

# Retry config
MAX_RETRIES = 6
BACKOFF_FACTOR = 3
MAX_BACKOFF = 60
JITTER = (0.25, 0.75)

# Language & country filters (per gdeltdoc docs)
LANGUAGE_FILTER = "eng"      # ISO 639-1 code
COUNTRY_FILTER  = "US"    # FIPS 2-letter country code

# Columns to keep (fixed order) — language removed
SELECTED_COLS = ["url", "title", "socialimage", "domain", "query_keyword"]

# Checkpoint config
CHECKPOINT_PATH = "gdelt_checkpoint.jsonl"
RESET_CHECKPOINT = os.getenv("RESET_CHECKPOINT", "0") == "1"

# Logging config
LOG_FILE = "gdelt_scrape.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------- LOGGING SETUP ----------
logger = logging.getLogger("gdelt")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

ch = logging.StreamHandler()
ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))

fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(lineno)d | %(message)s",
    "%Y-%m-%d %H:%M:%S"
))

if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)

def scalar_or_list(v):
    if isinstance(v, (list, tuple, set)):
        v = list(v)
        return v if len(v) > 1 else v[0]
    return v

# ---------- HELPERS ----------
def month_windows(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d").replace(day=1)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    cur = start
    while cur <= end_dt.replace(day=1):
        next_month = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = min(next_month - timedelta(days=1), end_dt)
        yield cur.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")
        cur = next_month

def task_key(kw, dom, start, end):
    return f"{kw}|{dom}|{start}|{end}"

def load_checkpoint_keys(path):
    done = set()
    if RESET_CHECKPOINT or not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    done.add(task_key(rec["kw"], rec["dom"], rec["start"], rec["end"]))
            except Exception:
                continue
    return done

def append_checkpoint(path, kw, dom, start, end, status="ok", rows=0, err=None):
    rec = {
        "kw": kw, "dom": dom, "start": start, "end": end,
        "status": status, "rows": int(rows), "ts": datetime.utcnow().isoformat() + "Z",
    }
    if err:
        rec["error"] = str(err)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg or "quota" in msg:
        return True
    resp = getattr(err, "response", None)
    code = getattr(resp, "status_code", None)
    return code == 429

def is_transient_error(err: Exception) -> bool:
    msg = str(err).lower()
    needles = ["timeout", "timed out", "temporarily unavailable", "connection reset",
               "connection aborted", "connection refused", "remote disconnected",
               "service unavailable", "503", "bad gateway", "502", "gateway timeout", "504"]
    return any(n in msg for n in needles)

def fetch_with_retry(gd: GdeltDoc, f: Filters, ctx: dict):
    attempt = 0
    while True:
        try:
            logger.debug("Request→ kw=%s dom=%s win=%s→%s attempt=%d",
                         ctx['kw'], ctx['dom'], ctx['start'], ctx['end'], attempt+1)
            return gd.article_search(f)
        except Exception as e:
            attempt += 1
            retryable = is_rate_limit_error(e) or is_transient_error(e)
            if not retryable or attempt > MAX_RETRIES:
                logger.error("Giving up after %d attempts | kw=%s dom=%s window=%s→%s | err=%s",
                             attempt, ctx["kw"], ctx["dom"], ctx["start"], ctx["end"], e)
                raise
            sleep_for = min((BACKOFF_FACTOR ** (attempt - 1)), MAX_BACKOFF) + random.uniform(*JITTER)
            logger.warning("Retry %d/%d (%.1fs) | kw=%s dom=%s window=%s→%s | err=%s",
                           attempt, MAX_RETRIES, sleep_for, ctx["kw"], ctx["dom"], ctx["start"], ctx["end"], e)
            time.sleep(sleep_for)

def project_selected_columns(df: pd.DataFrame, kw: str) -> pd.DataFrame:
    """Keep exactly SELECTED_COLS (ordered); create missing cols as empty strings."""
    df = df.copy()
    df["query_keyword"] = kw
    for c in SELECTED_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[SELECTED_COLS]

# ---------- MAIN ----------
def main():
    gd = GdeltDoc()
    uniq_domains = sorted(set(domains))
    file_exists = os.path.isfile(OUTPUT_CSV)

    windows = list(month_windows(START, END))
    total_windows = len(windows)
    total_tasks = len(keywords) * len(uniq_domains) * total_windows

    done_keys = load_checkpoint_keys(CHECKPOINT_PATH)
    logger.info("Starting run | tasks=%d (kw=%d × dom=%d × win=%d) | completed=%d | resume=%s",
                total_tasks, len(keywords), len(uniq_domains), total_windows, len(done_keys),
                "fresh" if RESET_CHECKPOINT else "from checkpoint")
    logger.info("Output CSV: %s | Log file: %s | Level: %s | Checkpoint: %s",
                OUTPUT_CSV, LOG_FILE, LOG_LEVEL, CHECKPOINT_PATH)
    logger.info("Writing columns: %s", SELECTED_COLS)

    processed = 0
    skipped = 0

    for kw in keywords:
        logger.info("=== Keyword: %s ===", kw)
        for dom in uniq_domains:
            logger.info("---> Domain: %s", dom)
            for i, (start_date, end_date) in enumerate(windows, start=1):
                key = task_key(kw, dom, start_date, end_date)
                if key in done_keys:
                    skipped += 1
                    if skipped % 200 == 0:
                        logger.info("    ↩︎ Skipped %d already-completed tasks so far...", skipped)
                    continue

                if i == 1 or i % 3 == 0:
                    logger.info("    ⏳ Window %d/%d: %s → %s", i, total_windows, start_date, end_date)
                else:
                    logger.debug("    (window %d/%d: %s → %s)", i, total_windows, start_date, end_date)

                ctx = {"kw": kw, "dom": dom, "start": start_date, "end": end_date}

                try:
                    f = Filters(
                        keyword=kw,
                        start_date=start_date,
                        end_date=end_date,
                        num_records=RECORDS_PER_QUERY,
                        domain_exact=dom,
                        language=scalar_or_list(LANGUAGE_FILTER),  # "en"
                        country=scalar_or_list(COUNTRY_FILTER),    # "US"
                    )
                    df = fetch_with_retry(gd, f, ctx)

                    if df is not None and not df.empty:
                        out = project_selected_columns(df, kw)
                        if not out.empty:
                            out.to_csv(
                                OUTPUT_CSV,
                                mode="a",
                                header=not file_exists,
                                index=False
                            )
                            file_exists = True
                            rows = len(out)
                            logger.info("    ✅ Saved %d rows | kw=%s dom=%s month=%s",
                                        rows, kw, dom, start_date[:7])
                        else:
                            rows = 0
                            logger.debug("    (0 rows after projection)")
                    else:
                        rows = 0
                        if i % 6 == 0:
                            logger.info("    (no results) kw=%s dom=%s month=%s", kw, dom, start_date[:7])

                    append_checkpoint(CHECKPOINT_PATH, kw, dom, start_date, end_date, status="ok", rows=rows)
                    done_keys.add(key)
                    processed += 1

                except Exception as e:
                    logger.exception("    [warn] kw=%s dom=%s %s→%s | %s",
                                     kw, dom, start_date, end_date, e)
                    append_checkpoint(CHECKPOINT_PATH, kw, dom, start_date, end_date, status="error", err=e)

                time.sleep(BASE_SLEEP_SECONDS + random.uniform(0, 0.3))

    logger.info("🎉 Finished. Appended data to %s | processed=%d | skipped=%d | total=%d",
                OUTPUT_CSV, processed, skipped, total_tasks)

if __name__ == "__main__":
    main()
