"""Download and prepare the English, non-synthetic OASST1 prompt/reply pairs."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.messages.jsonl.gz?download=true"
RAW = Path("data/raw/2023-04-12_oasst_ready.messages.jsonl.gz")
OUTPUT = Path("data/oasst1_train.txt")


def clean(message: dict) -> bool:
    if message.get("lang") != "en" or message.get("deleted") is True:
        return False
    if message.get("review_result") is False or message.get("synthetic") is not False:
        return False
    labels = message.get("labels") or {}
    if isinstance(labels, dict) and "name" in labels:
        labels = dict(zip(labels.get("name", []), labels.get("value", [])))
    for key in ("spam", "pii", "not_appropriate", "hate_speech", "sexual_content"):
        item = labels.get(key)
        value = item.get("value") if isinstance(item, dict) else item
        if value is not None and float(value) > 0:
            return False
    text = (message.get("text") or "").strip()
    return 20 <= len(text) <= 5000


def main() -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print(f"Downloading OASST1 from {URL}")
        urlretrieve(URL, RAW)
    records = []
    with gzip.open(RAW, "rt", encoding="utf-8") as source:
        for line in source:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    by_id = {m.get("message_id"): m for m in records}
    pairs = []
    for reply in records:
        if reply.get("role") != "assistant" or not clean(reply):
            continue
        prompt = by_id.get(reply.get("parent_id"))
        if not prompt or prompt.get("role") != "prompter" or not clean(prompt):
            continue
        pairs.append(f"User: {prompt['text'].strip()}\nKalki: {reply['text'].strip()}\n\n")
    if not pairs:
        raise SystemExit("No eligible prompt/reply pairs found. Inspect dataset metadata and filters.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(pairs), encoding="utf-8")
    print(f"Prepared {len(pairs):,} English, reviewed, non-synthetic pairs -> {OUTPUT}")

if __name__ == "__main__":
    main()
