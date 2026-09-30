#!/usr/bin/env python3
"""
Gemini JSONL -> CSV benchmark runner for CIPHERGRID-style records.

Input JSONL record fields:
  - prompt-base (string)
  - query (string)
  - image (string; base64 image bytes or a data URL)
  - id (string or int)

Per-record prompt:
  prompt-base + "\n" + query + "\nAnswer only with the correct response!\n"

Output CSV columns:
  id,response,time,tokens,reasoning_tokens

Behavior:
  - Resume: skip ids with non-empty response in an existing CSV.
  - Failure: write nothing to the results CSV and continue.
  - Tokens: required. If total token usage is missing, treat the record as failed.
  - Reasoning: uses high thinking level.
  - API key: read from GEMINI_API_KEY or GOOGLE_API_KEY. Never hardcode secrets.

Diagnostics:
  - --debug prints full tracebacks.
  - --fail-log writes JSONL failure details per record.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set, Tuple

from google import genai
from google.genai import types


REQUIRED_FIELDS = ["prompt-base", "query", "image", "id"]
CSV_COLUMNS = ["id", "response", "time", "tokens", "reasoning_tokens"]


def raise_csv_field_limit() -> None:
    """
    Allow reading existing CSVs with very large model outputs.

    Python's csv module defaults to a relatively small field limit, which can
    fail when resuming from benchmark outputs containing long responses.
    """
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size = int(max_size / 10)


raise_csv_field_limit()


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def get_api_key() -> str:
    """
    Return the Gemini API key from the environment.

    GEMINI_API_KEY is checked first because it is explicit to this runner.
    GOOGLE_API_KEY is also supported because it is commonly used by Google
    Gemini examples and tooling.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing Gemini API key. Set one of:\n"
            "  export GEMINI_API_KEY='your-key-here'\n"
            "  export GOOGLE_API_KEY='your-key-here'"
        )
    return api_key


def is_data_url(value: str) -> bool:
    return value.startswith("data:") and ";base64," in value


def extract_b64_from_data_url(data_url: str) -> str:
    return data_url.split(";base64,", 1)[1]


def normalize_image_to_bytes(image_b64_or_dataurl: str) -> bytes:
    """
    Accept either a raw base64 image string or a base64 data URL.

    Returns decoded image bytes suitable for types.Part.from_bytes(...).
    """
    value = (image_b64_or_dataurl or "").strip()
    if not value:
        raise ValueError("Empty image field")

    image_b64 = extract_b64_from_data_url(value) if is_data_url(value) else value
    try:
        return base64.b64decode(image_b64, validate=True)
    except Exception as ex:
        raise ValueError(f"Invalid base64 image and not a valid data URL: {ex}") from ex


def read_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")

            yield line_no, obj


def ensure_csv_header(csv_path: Path) -> None:
    needs_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    if not needs_header:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)


def load_done_ids(csv_path: Path) -> Set[str]:
    """
    Return ids whose existing CSV rows have a non-empty response.

    This preserves the original resume semantics: partial/empty rows are not
    treated as complete.
    """
    done: Set[str] = set()

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return done

        if "id" not in reader.fieldnames or "response" not in reader.fieldnames:
            raise ValueError(f"Existing CSV missing required columns. Found: {reader.fieldnames}")

        for row in reader:
            record_id = (row.get("id") or "").strip()
            response = (row.get("response") or "").strip()
            if record_id and response:
                done.add(record_id)

    return done


def append_csv_row(csv_path: Path, row: Tuple[Any, str, float, int, Optional[int]]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)
        handle.flush()


def append_fail_log(fail_log: Optional[Path], payload: Dict[str, Any]) -> None:
    if fail_log is None:
        return

    fail_log.parent.mkdir(parents=True, exist_ok=True)
    with fail_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def get_attr_or_key(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def extract_usage(usage_metadata: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract total tokens and reasoning/thought tokens from Gemini usage metadata.

    Returns:
      (total_token_count, thoughts_token_count)
    """
    total = get_attr_or_key(usage_metadata, "total_token_count")
    thoughts = get_attr_or_key(usage_metadata, "thoughts_token_count")

    total_tokens = int(total) if isinstance(total, (int, float)) else None
    thoughts_tokens = int(thoughts) if isinstance(thoughts, (int, float)) else None

    return total_tokens, thoughts_tokens


def build_prompt(record: Dict[str, Any]) -> str:
    prompt_base = str(record["prompt-base"])
    query = str(record["query"])
    return prompt_base + "\n" + query + "\nAnswer only with the correct response!\n"


def build_contents(prompt_text: str, image_bytes: bytes, image_mime: str) -> list[Any]:
    return [
        types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
        prompt_text,
    ]


def build_generation_config() -> types.GenerateContentConfig:
    """
    Return the generation config used by the original diagnostic runner.

    Keep these values stable for benchmark comparability.
    """
    return types.GenerateContentConfig(
        max_output_tokens=128000,
        thinking_config=types.ThinkingConfig(thinking_level="high"),
        temperature=0.0,
        top_p=1.0,
    )


def call_gemini(
    client: genai.Client,
    *,
    model: str,
    prompt_text: str,
    image_bytes: bytes,
    image_mime: str,
    stream_to_stdout: bool,
) -> Tuple[str, float, Optional[int], Optional[int], Any]:
    """
    Call Gemini and return:
      (full_text, elapsed_seconds, total_tokens, thoughts_tokens, usage_metadata)
    """
    start = time.time()
    contents = build_contents(prompt_text, image_bytes, image_mime)
    config = build_generation_config()

    text_parts: list[str] = []
    last_usage = None

    if stream_to_stdout:
        stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                text_parts.append(chunk_text)
                print(chunk_text, end="", flush=True)

            usage_metadata = getattr(chunk, "usage_metadata", None)
            if usage_metadata is not None:
                last_usage = usage_metadata

        print(flush=True)
    else:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text_parts.append(getattr(response, "text", "") or "")
        last_usage = getattr(response, "usage_metadata", None)

    elapsed = time.time() - start
    full_text = "".join(text_parts)
    total_tokens, thoughts_tokens = extract_usage(last_usage)

    return full_text, elapsed, total_tokens, thoughts_tokens, last_usage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-mime", default="image/png")
    parser.add_argument(
        "--model",
        required=True,
        help="Gemini model name to use for both streaming and non-streaming calls.",
    )
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fail-log", default=None, help="Write failures to this JSONL path")
    return parser.parse_args()


def log_failure(
    *,
    fail_log: Optional[Path],
    line_no: int,
    record_id: Any,
    message: str,
    exception: Optional[BaseException] = None,
    debug: bool = False,
    usage_repr: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "provider": "gemini",
        "line_no": line_no,
        "id": record_id,
        "error": message,
    }

    if usage_repr is not None:
        payload["usage_repr"] = usage_repr

    if exception is not None:
        payload["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception),
        }
        payload["traceback"] = traceback.format_exc() if debug else None

    append_fail_log(fail_log, payload)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    fail_log = Path(args.fail_log) if args.fail_log else None

    if not input_path.exists():
        raise SystemExit(f"Input JSONL not found: {input_path}")

    api_key = get_api_key()

    done_ids = load_done_ids(output_path)
    if done_ids:
        eprint(f"[resume] Found {len(done_ids)} completed ids in existing CSV; skipping them.")

    ensure_csv_header(output_path)
    client = genai.Client(api_key=api_key)

    total = 0
    skipped = 0
    wrote = 0
    failed = 0

    for line_no, record in read_jsonl(input_path):
        total += 1

        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            failed += 1
            message = f"missing fields {missing}; skipping"
            eprint(f"[line {line_no}] {message}")
            log_failure(
                fail_log=fail_log,
                line_no=line_no,
                record_id=record.get("id"),
                message=message,
            )
            continue

        record_id = str(record.get("id"))
        if record_id in done_ids:
            skipped += 1
            continue

        try:
            prompt_text = build_prompt(record)
            image_bytes = normalize_image_to_bytes(str(record["image"]))

            if args.stream:
                print(f"\n--- id={record_id} (line {line_no}) ---", flush=True)
                print(prompt_text, flush=True)

            full_text, elapsed, total_tokens, thoughts_tokens, usage_metadata = call_gemini(
                client,
                model=args.model,
                prompt_text=prompt_text,
                image_bytes=image_bytes,
                image_mime=args.image_mime,
                stream_to_stdout=args.stream,
            )

            if not full_text.strip():
                failed += 1
                message = "empty response; skipping (no write)"
                eprint(f"[id={record_id} line={line_no}] {message}")
                log_failure(
                    fail_log=fail_log,
                    line_no=line_no,
                    record_id=record_id,
                    message=message,
                )
                continue

            if total_tokens is None:
                failed += 1
                message = f"token usage missing from API response; usage_metadata={usage_metadata!r}"
                eprint(f"[id={record_id} line={line_no}] {message} (no write)")
                log_failure(
                    fail_log=fail_log,
                    line_no=line_no,
                    record_id=record_id,
                    message=message,
                    usage_repr=repr(usage_metadata),
                )
                continue

            append_csv_row(
                output_path,
                (record_id, full_text, round(elapsed, 6), int(total_tokens), thoughts_tokens),
            )
            done_ids.add(record_id)
            wrote += 1

        except Exception as ex:
            failed += 1
            message = f"FAIL: {type(ex).__name__}: {ex}"
            eprint(f"[id={record_id} line={line_no}] {message}")

            if args.debug:
                eprint("  --- traceback ---")
                eprint(traceback.format_exc())

            log_failure(
                fail_log=fail_log,
                line_no=line_no,
                record_id=record_id,
                message=message,
                exception=ex,
                debug=args.debug,
            )

    print(f"\nDone. total={total} wrote={wrote} skipped={skipped} failed={failed} -> {output_path}")
    if fail_log:
        print(f"Failures log: {fail_log}")


if __name__ == "__main__":
    main()
