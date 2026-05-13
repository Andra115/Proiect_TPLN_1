"""
Build an out-of-distribution gold-evaluation set by fetching Romanian text from
Wikipedia (a different distribution than the news-domain RoLargeSum training
data), filtering to diacritic-rich sentences, deduplicating against the
training set, and applying the project's degradation pipeline.

Output is a JSONL file (default `data/gold_candidates.jsonl`) with the same
schema as `train.jsonl` / `val.jsonl`:

    {"input": <degraded>, "target": <clean>, "scenario": <str>,
     "changed_positions": [int, ...], "source": "wikipedia:<title>"}

Usage:
    python -m scripts.build_gold \\
        --train-file data/train.jsonl \\
        --output data/gold_candidates.jsonl \\
        --num-sentences 100 \\
        --per-sentence-variants 2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from app.degradation import (  # noqa: E402
    DegradationConfig,
    DegradationScenario,
    generate_samples,
)

logger = logging.getLogger(__name__)


WIKI_SUMMARY_URL = "https://ro.wikipedia.org/api/rest_v1/page/random/summary"
USER_AGENT = (
    "WikipediaFetchRandom/0.1"
)

# Quality filters for candidate sentences
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
ROMANIAN_DIACRITICS = set("ășțăâîȘȚĂÂÎ")
SENTENCE_MIN_LEN = 40
SENTENCE_MAX_LEN = 400
MIN_LETTER_RATIO = 0.70   # filter out heavily numeric / symbolic sentences


def fetch_random_summary(session: requests.Session) -> dict | None:
    """Fetch one random Romanian Wikipedia article summary. Returns None on error."""
    try:
        r = session.get(
            WIKI_SUMMARY_URL,
            timeout=10,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else 30.0
            logger.warning(
                "Wikipedia rate-limit hit (429). Sleeping %.0fs before retry...",
                wait,
            )
            time.sleep(wait)
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.warning("Wikipedia fetch failed: %s", e)
        return None


def is_good_sentence(s: str) -> bool:
    """Filter to sentences worth including in the gold set."""
    s = s.strip()
    if not (SENTENCE_MIN_LEN <= len(s) <= SENTENCE_MAX_LEN):
        return False
    # Must contain at least one Romanian diacritic - otherwise the model
    # has nothing interesting to do on this example.
    if not any(c in ROMANIAN_DIACRITICS for c in s):
        return False
    # Filter out math/list/table content: must be mostly letters
    n_letters = sum(1 for c in s if c.isalpha())
    if n_letters / max(len(s), 1) < MIN_LETTER_RATIO:
        return False
    # Wikipedia sentences sometimes contain residual citation markers like
    # [1], [12]. Drop sentences with more than one such marker - they're
    # usually dense reference paragraphs that don't read naturally.
    if len(re.findall(r"\[\d+\]", s)) > 1:
        return False
    return True


def collect_sentences(
    target_count: int,
    session: requests.Session,
    max_articles: int = 500,
    sleep_seconds: float = 0.3,
) -> list[tuple[str, str]]:
    """Fetch random Wikipedia articles until we have `target_count` good
    sentences. Returns list of (sentence, article_title) pairs."""
    out: list[tuple[str, str]] = []
    seen_sentences: set[str] = set()
    article_count = 0

    while len(out) < target_count and article_count < max_articles:
        article = fetch_random_summary(session)
        article_count += 1
        if article is None:
            time.sleep(sleep_seconds * 3)  # back off on errors
            continue

        title = article.get("title", "<unknown>")
        # `extract` is plain text intro; `extract_html` would have markup.
        text = article.get("extract") or ""
        if not text:
            continue

        for sent in _SENT_SPLIT.split(text):
            sent = sent.strip()
            if not is_good_sentence(sent):
                continue
            if sent in seen_sentences:
                continue
            seen_sentences.add(sent)
            out.append((sent, title))
            if len(out) >= target_count:
                break

        if article_count % 20 == 0:
            logger.info(
                "  fetched %d articles, %d good sentences so far",
                article_count, len(out),
            )
        time.sleep(sleep_seconds)

    if len(out) < target_count:
        logger.warning(
            "Only collected %d sentences (wanted %d) after %d articles.",
            len(out), target_count, article_count,
        )
    return out[:target_count]


def load_train_targets(train_jsonl: Path) -> set[str]:
    """Load every `target` string from train.jsonl into a set for exact-match deduplication"""
    targets: set[str] = set()
    if not train_jsonl.exists():
        logger.warning(
            "Train file %s not found — skipping dedup (gold candidates may "
            "overlap with training data!)", train_jsonl,
        )
        return targets

    logger.info("Loading training targets for dedup from %s ...", train_jsonl)
    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("target")
                if isinstance(t, str):
                    targets.add(t)
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d training targets for dedup.", len(targets))
    return targets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=str, default="data/gold_candidates.jsonl",help="JSONL output path for gold candidates")
    p.add_argument("--train-file", type=str, default="data/train.jsonl", help="Training JSONL used for dedup (overlap = leak)")
    p.add_argument("--num-sentences", type=int, default=60,help="How many clean sentences to fetch")
    p.add_argument("--per-sentence-variants", type=int, default=2,help="Number of degraded variants per sentence (each picks a different scenario)")
    p.add_argument("--seed", type=int, default=42,help="Random seed for the degradation pipeline")
    p.add_argument("--max-articles", type=int, default=500,help="Cap on Wikipedia API calls")
    p.add_argument("--sleep", type=float, default=10, help="Seconds to sleep between Wikipedia requests")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Dedup set from training data
    train_targets = load_train_targets(Path(args.train_file))

    # 2) Fetch candidate sentences from Wikipedia
    logger.info(
        "Fetching ~%d candidate sentences from Romanian Wikipedia ...",
        args.num_sentences,
    )
    # Over-fetch slightly because dedup may drop some
    fetch_target = int(args.num_sentences * 1.3)
    session = requests.Session()
    candidate_pairs = collect_sentences(
        target_count=fetch_target,
        session=session,
        max_articles=args.max_articles,
        sleep_seconds=args.sleep,
    )

    # 3) Filter out sentences that exist verbatim in train.jsonl
    deduped: list[tuple[str, str]] = []
    n_dropped = 0
    for sent, title in candidate_pairs:
        if sent in train_targets:
            n_dropped += 1
            continue
        deduped.append((sent, title))
    logger.info(
        "Dedup against train: dropped %d / %d (overlap with training data)",
        n_dropped, len(candidate_pairs),
    )
    deduped = deduped[: args.num_sentences]
    logger.info("Final candidate sentence count: %d", len(deduped))

    # 4) Apply degradation pipeline
    logger.info(
        "Applying degradation pipeline (%d variants per sentence) ...",
        args.per_sentence_variants,
    )
    config = DegradationConfig(
        scenario=DegradationScenario.MIXED,
        strip_probability=0.5,
        wrong_diacritic_probability=0.3,
        wrong_swap_probability=0.2,
        ocr_probability=0.02,
        seed=args.seed,
    )

    sentences = [s for s, _ in deduped]
    title_lookup = dict(deduped)
    samples = generate_samples(
        sentences, config, per_text_variants=args.per_sentence_variants
    )

    # 5) Write JSONL, streaming each line as we go so a Ctrl-C mid-run
    #    still leaves a usable partial file.
    logger.info("Writing %d candidate samples to %s ...", len(samples), output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            obj = {
                "input": s.input,
                "target": s.target,
                "scenario": s.scenario,
                "changed_positions": s.changed_positions,
                "source": f"wikipedia:{title_lookup.get(s.target, '?')}",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info(
        "Done. %d candidates written to %s.",
        len(samples), output_path,
    )


if __name__ == "__main__":
    main()
