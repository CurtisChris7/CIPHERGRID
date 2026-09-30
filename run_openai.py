#!/usr/bin/env python3
"""
Run a JSONL benchmark through the OpenAI Responses API and append results to CSV.

Expected input JSONL fields per record:
  - prompt-base: string
  - query: string
  - image: base64 image bytes or a data URL
  - id: string or int

Per-record prompt:
  prompt-base + "\n" + query + "\nAnswer only with the correct response!\n"

Output CSV columns:
  id,response,time,reasoning_tokens

Repository / security notes:
  - Reads the API key from OPENAI_API_KEY only.
  - Does not hardcode secrets.
  - Resumes from an existing CSV by skipping ids with non-empty responses.
  - Writes successful rows immediately so interrupted runs can resume.
  - Retries transient API/network errors with exponential backoff.
  - Writes nothing for failed rows, preserving the existing CSV schema.

Example:
  export OPENAI_API_KEY="..."
  python scripts/run_openai.py \
    --input data/ciphergrid.jsonl \
    --output results/gpt_5_4.csv \
    --model gpt-5.4 \
    --reasoning high
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Set, Tuple

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


REQUIRED_FIELDS = ["prompt-base", "query", "image", "id"]
CSV_COLUMNS = ["id", "response", "time", "reasoning_tokens"]

RETRYABLE_EXCEPTIONS = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.ConnectError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


def eprint(*args: Any) -> None:
    """Print to stderr with immediate flushing."""
    print(*args, file=sys.stderr, flush=True)


def is_data_url(value: str) -> bool:
    """Return True when a string is already a base64 data URL."""
    return value.startswith("data:") and ";base64," in value


def normalize_image_to_data_url(image_b64_or_dataurl: str, default_mime: str = "image/png") -> str:
    """
    Convert a base64 image string to a data URL, or pass through an existing data URL.

    The Responses API accepts input images as data URLs. CIPHERGRID records usually
    store PNG bytes as raw base64, so this function normalizes both accepted forms.
    """
    value = (image_b64_or_dataurl or "").strip()
    if not value:
        raise ValueError("Empty image field")

    if is_data_url(value):
        return value

    try:
        base64.b64decode(value, validate=True)
    except Exception as ex:
        raise ValueError(f"Image field is not valid base64 and is not a data URL: {ex}") from ex

    return f"data:{default_mime};base64,{value}"


def get_nested_value(obj: Any, path: Sequence[str]) -> Any:
    """Read a nested attribute/dict path from SDK objects or plain dictionaries."""
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            if key not in cur:
                return None
            cur = cur[key]
        elif hasattr(cur, key):
            cur = getattr(cur, key)
        else:
            return None
    return cur


def safe_get_reasoning_tokens(usage: Any) -> Optional[int]:
    """Best-effort extraction of reasoning-token counts across SDK response shapes."""
    if usage is None:
        return None

    candidate_paths = [
        ("reasoning_tokens",),
        ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    ]

    for path in candidate_paths:
        value = get_nested_value(usage, path)
        if isinstance(value, int):
            return value

    return None


def read_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield ``(line_no, object)`` pairs from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(f"JSONL line {line_no} is invalid JSON: {ex}") from ex

            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")

            yield line_no, obj


def ensure_csv_header(csv_path: Path) -> None:
    """Create the CSV file and header when needed."""
    needs_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    if not needs_header:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)


def load_done_ids(csv_path: Path) -> Set[str]:
    """
    Return ids already completed in an existing output CSV.

    A row is considered complete only when both ``id`` and ``response`` are non-empty.
    This preserves the script's resume behavior while allowing empty/failed rows to be rerun.
    """
    done: Set[str] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return done

        missing = {"id", "response"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Existing CSV is missing required columns {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            record_id = (row.get("id") or "").strip()
            response = (row.get("response") or "").strip()
            if record_id and response:
                done.add(record_id)

    return done


def append_csv_row(csv_path: Path, row: Tuple[Any, str, float, Optional[int]]) -> None:
    """Append one result row and flush it to disk."""
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)
        handle.flush()


def build_input_payload(prompt_text: str, image_data_url: str) -> list[dict[str, Any]]:
    """Build the multimodal Responses API input payload for one record."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": image_data_url},
            ],
        }
    ]


def build_request(
    *,
    model: str,
    prompt_text: str,
    image_data_url: str,
    reasoning_effort: Optional[str],
    max_output_tokens: int,
    timeout_seconds: Optional[float],
    stream: bool,
) -> Dict[str, Any]:
    """Build keyword arguments for ``client.responses.create``."""
    request: Dict[str, Any] = {
        "model": model,
        "input": build_input_payload(prompt_text, image_data_url),
        "max_output_tokens": max_output_tokens,
        "stream": stream,
    }

    if timeout_seconds is not None:
        request["timeout"] = timeout_seconds

    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}

    return request


def extract_output_text(response: Any) -> str:
    """
    Best-effort text extraction from a Responses API response object.

    The SDK normally exposes ``response.output_text``. The fallback below is useful for
    unusual response shapes or partial objects captured from streaming completion events.
    """
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    output = getattr(response, "output", None)
    if not output:
        return ""

    text_parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not content:
            continue

        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)

    return "".join(text_parts)


def call_model_streaming(
    client: OpenAI,
    *,
    model: str,
    prompt_text: str,
    image_data_url: str,
    reasoning_effort: Optional[str],
    max_output_tokens: int,
    timeout_seconds: Optional[float],
    stream_to_stdout: bool,
    print_all_events: bool,
) -> Tuple[str, float, Optional[int]]:
    """Call the Responses API in streaming mode."""
    text_parts: list[str] = []
    final_response: Any = None
    start = time.time()

    request = build_request(
        model=model,
        prompt_text=prompt_text,
        image_data_url=image_data_url,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        stream=True,
    )

    stream = client.responses.create(**request)

    for event in stream:
        event_type = getattr(event, "type", None)

        if print_all_events and event_type:
            print(f"[event] {event_type}", flush=True)

        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            text_parts.append(delta)
            if stream_to_stdout:
                print(delta, end="", flush=True)

        elif event_type in ("response.completed", "response.incomplete", "response.failed"):
            final_response = getattr(event, "response", None)

        elif event_type == "error":
            final_response = getattr(event, "response", None)

    if stream_to_stdout:
        print(flush=True)

    elapsed = time.time() - start
    full_text = "".join(text_parts)

    # If no deltas were observed, try extracting text from the final response object.
    if not full_text and final_response is not None:
        full_text = extract_output_text(final_response)

    usage = getattr(final_response, "usage", None) if final_response is not None else None
    reasoning_tokens = safe_get_reasoning_tokens(usage)

    return full_text, elapsed, reasoning_tokens


def call_model_nonstreaming(
    client: OpenAI,
    *,
    model: str,
    prompt_text: str,
    image_data_url: str,
    reasoning_effort: Optional[str],
    max_output_tokens: int,
    timeout_seconds: Optional[float],
) -> Tuple[str, float, Optional[int]]:
    """Call the Responses API in non-streaming mode."""
    start = time.time()

    request = build_request(
        model=model,
        prompt_text=prompt_text,
        image_data_url=image_data_url,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        stream=False,
    )

    response = client.responses.create(**request)
    elapsed = time.time() - start

    full_text = extract_output_text(response)
    usage = getattr(response, "usage", None)
    reasoning_tokens = safe_get_reasoning_tokens(usage)

    return full_text, elapsed, reasoning_tokens


def call_with_retries(
    client: OpenAI,
    *,
    transport: str,
    model: str,
    prompt_text: str,
    image_data_url: str,
    reasoning_effort: Optional[str],
    max_output_tokens: int,
    timeout_seconds: Optional[float],
    stream_to_stdout: bool,
    print_all_events: bool,
    max_attempts: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
) -> Tuple[str, float, Optional[int]]:
    """Call the selected transport with bounded exponential backoff."""
    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        try:
            if transport == "stream":
                return call_model_streaming(
                    client,
                    model=model,
                    prompt_text=prompt_text,
                    image_data_url=image_data_url,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    timeout_seconds=timeout_seconds,
                    stream_to_stdout=stream_to_stdout,
                    print_all_events=print_all_events,
                )

            return call_model_nonstreaming(
                client,
                model=model,
                prompt_text=prompt_text,
                image_data_url=image_data_url,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
            )

        except RETRYABLE_EXCEPTIONS as ex:
            last_exc = ex
            if attempt >= max_attempts:
                break

            sleep_s = min(
                max_backoff_seconds,
                base_backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, 1.0),
            )
            eprint(f"[retry {attempt}/{max_attempts}] {type(ex).__name__}: {ex}; sleeping {sleep_s:.2f}s")
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Existing flags and defaults are intentionally preserved."""
    parser = argparse.ArgumentParser(
        description="Run JSONL benchmark records through the OpenAI Responses API."
    )

    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Path to output CSV (append + resume)")
    parser.add_argument("--model", required=True, help="Model name (for example: gpt-5.4)")

    parser.add_argument(
        "--reasoning",
        default=None,
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort. If omitted, reasoning is not sent.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=128000)

    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Per-request timeout in seconds.",
    )

    parser.add_argument(
        "--transport",
        choices=["nonstream", "stream"],
        default="nonstream",
        help="Use nonstream for long benchmark runs; stream for interactive debugging.",
    )

    parser.add_argument(
        "--stream",
        dest="stream_to_stdout",
        action="store_true",
        help="Print streamed deltas to stdout",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream_to_stdout",
        action="store_false",
        help="Do not print streamed deltas",
    )
    parser.set_defaults(stream_to_stdout=False)

    parser.add_argument(
        "--print-all-events",
        action="store_true",
        help="Print all SSE event types while streaming",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Retry attempts per record for transient failures",
    )
    parser.add_argument("--base-backoff", type=float, default=1.0, help="Base backoff in seconds")
    parser.add_argument("--max-backoff", type=float, default=30.0, help="Maximum backoff in seconds")

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap on number of input rows processed in this run.",
    )

    parser.add_argument(
        "--skip-empty-response",
        action="store_true",
        help="Treat empty model outputs as failures and write nothing for that row.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    """Validate path arguments and return normalized ``Path`` objects."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input JSONL not found: {input_path}")

    if not input_path.is_file():
        raise SystemExit(f"Input path is not a file: {input_path}")

    if args.max_output_tokens <= 0:
        raise SystemExit("--max-output-tokens must be positive")

    if args.timeout is not None and args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be positive")

    if args.base_backoff < 0:
        raise SystemExit("--base-backoff must be non-negative")

    if args.max_backoff < 0:
        raise SystemExit("--max-backoff must be non-negative")

    if args.sample_size is not None and args.sample_size < 0:
        raise SystemExit("--sample-size must be non-negative")

    return input_path, output_path


def get_api_key_from_environment() -> str:
    """Load the OpenAI API key from the environment with a repo-safe error message."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY.\n"
            "Set it before running, for example:\n"
            "  export OPENAI_API_KEY='your-api-key-here'"
        )
    return api_key


def main() -> None:
    args = parse_args()
    input_path, output_path = validate_args(args)
    api_key = get_api_key_from_environment()

    done_ids = load_done_ids(output_path)
    if done_ids:
        eprint(f"[resume] Found {len(done_ids)} completed ids in existing CSV; will skip them.")

    ensure_csv_header(output_path)

    client = OpenAI(api_key=api_key, max_retries=5, timeout=args.timeout)

    total = 0
    skipped = 0
    wrote = 0
    failed = 0
    processed = 0

    for line_no, record in read_jsonl(input_path):
        total += 1

        if args.sample_size is not None and processed >= args.sample_size:
            break

        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            failed += 1
            eprint(f"[line {line_no}] missing fields {missing}; skipping")
            continue

        record_id = str(record.get("id"))
        if record_id in done_ids:
            skipped += 1
            continue

        processed += 1

        try:
            prompt_base = str(record["prompt-base"])
            query = str(record["query"])
            prompt_text = f"{prompt_base}\n{query}\nAnswer only with the correct response!\n"
            image_data_url = normalize_image_to_data_url(str(record["image"]), default_mime="image/png")

            if args.transport == "stream" and args.stream_to_stdout:
                print(f"\n--- id={record_id} (line {line_no}) ---", flush=True)
                print(prompt_text, flush=True)

            full_text, elapsed, reasoning_tokens = call_with_retries(
                client,
                transport=args.transport,
                model=args.model,
                prompt_text=prompt_text,
                image_data_url=image_data_url,
                reasoning_effort=args.reasoning,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout,
                stream_to_stdout=args.stream_to_stdout,
                print_all_events=args.print_all_events,
                max_attempts=args.max_attempts,
                base_backoff_seconds=args.base_backoff,
                max_backoff_seconds=args.max_backoff,
            )

            if args.skip_empty_response and not full_text.strip():
                failed += 1
                eprint(f"[id={record_id} line={line_no}] empty response; skipping (no write)")
                continue

            append_csv_row(output_path, (record_id, full_text, round(elapsed, 6), reasoning_tokens))
            done_ids.add(record_id)
            wrote += 1

        except APIStatusError as ex:
            failed += 1
            eprint(
                f"[id={record_id} line={line_no}] APIStatusError: {ex} "
                f"status={getattr(ex, 'status_code', None)} "
                f"request_id={getattr(ex, 'request_id', None)} (no write)"
            )

        except Exception as ex:
            failed += 1
            eprint(f"[id={record_id} line={line_no}] error: {type(ex).__name__}: {ex} (no write)")

    print(
        f"\nDone. total={total} processed={processed} wrote={wrote} "
        f"skipped={skipped} failed={failed} -> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
