#!/usr/bin/env python3
"""
Run CIPHERGRID JSONL records through Qwen-VL / Qwen thinking models using
Alibaba DashScope's OpenAI-compatible Chat Completions API.

Input JSONL record fields:
  - prompt-base (string)
  - query (string)
  - image (string; base64 image bytes or a data URL)
  - id (string or int)

Per-record prompt:
  prompt-base + "\n" + query + "\nAnswer only with the correct response!\n"

Output CSV columns:
  id,response,time,reasoning_tokens

Notes:
  - This runner intentionally uses Chat Completions, not OpenAI Responses.
  - API credentials are read from the environment variable named by
    --api-key-env, which defaults to DASHSCOPE_API_KEY.
  - Streaming calls collect reasoning_content separately and write only the
    final answer/content to the main response column.
  - The CSV column name reasoning_tokens is kept for compatibility; for this
    API, the value stored there is completion_tokens when usage is available.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set, Tuple

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

ACTION_RE = re.compile(
    r"\b(?:ma|ga|za|va|ya|ba|sa|ra|ta)(?:-(?:ma|ga|za|va|ya|ba|sa|ra|ta))*\b"
)


def raise_csv_field_limit() -> None:
    """Allow reading CSV rows with very large response/reasoning fields."""
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size = int(max_size / 10)


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def is_data_url(value: str) -> bool:
    return value.startswith("data:") and ";base64," in value


def normalize_image_to_data_url(image_b64_or_dataurl: str, default_mime: str = "image/png") -> str:
    """Accept either a data URL or raw base64 image bytes, and return a data URL."""
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


def read_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (line_no, object) pairs from a JSONL file."""
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
    """Create the output CSV and header if needed."""
    needs_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    if not needs_header:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)


def load_done_ids(csv_path: Path) -> Set[str]:
    """Return IDs with non-empty responses in an existing output CSV."""
    done: Set[str] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return done
        if "id" not in reader.fieldnames or "response" not in reader.fieldnames:
            raise ValueError(f"Existing CSV is missing required columns. Found: {reader.fieldnames}")

        for row in reader:
            rid = (row.get("id") or "").strip()
            resp = (row.get("response") or "").strip()
            if rid and resp:
                done.add(rid)

    return done


def append_csv_row(csv_path: Path, row: Tuple[Any, str, float, Optional[int]]) -> None:
    """Append one completed benchmark row to the output CSV."""
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)
        handle.flush()


def safe_get_completion_tokens(usage: Any) -> Optional[int]:
    """
    Return completion_tokens from an OpenAI-compatible usage object/dict.

    The main CSV keeps the historical column name reasoning_tokens for
    compatibility with existing analysis scripts.
    """
    if usage is None:
        return None

    if hasattr(usage, "completion_tokens"):
        value = getattr(usage, "completion_tokens")
        if isinstance(value, int):
            return value

    if isinstance(usage, dict):
        value = usage.get("completion_tokens")
        if isinstance(value, int):
            return value

    return None


def build_messages(
    prompt_text: str,
    image_data_url: str,
    system_prompt: Optional[str],
) -> list[dict[str, Any]]:
    """Build Qwen/OpenAI-compatible multimodal chat messages."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    )
    return messages


def maybe_extract_action_sequence(text: str, *, enabled: bool) -> str:
    """Optionally keep only the last CIPHERGRID-looking action sequence."""
    stripped = (text or "").strip()
    if not enabled:
        return stripped

    matches = ACTION_RE.findall(stripped)
    if matches:
        return matches[-1].strip()
    return stripped


def build_extra_body(args: argparse.Namespace) -> dict[str, Any]:
    """Build DashScope-specific extra_body options."""
    extra_body: dict[str, Any] = {
        # Thinking is always enabled. We intentionally omit thinking_budget so
        # DashScope uses the model's default/maximum thinking budget.
        "enable_thinking": True,
    }

    if args.vl_high_resolution_images:
        extra_body["vl_high_resolution_images"] = True

    return extra_body


def build_chat_request(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    timeout_seconds: Optional[float],
    extra_body: Optional[dict[str, Any]],
    stream: bool,
) -> dict[str, Any]:
    """Build a Chat Completions request while omitting unset optional values."""
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if stream:
        request["stream_options"] = {"include_usage": True}
    if temperature is not None:
        request["temperature"] = temperature
    if top_p is not None:
        request["top_p"] = top_p
    if timeout_seconds is not None:
        request["timeout"] = timeout_seconds
    if extra_body is not None:
        request["extra_body"] = extra_body

    return request


def call_model_nonstreaming(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    timeout_seconds: Optional[float],
    extra_body: Optional[dict[str, Any]],
) -> Tuple[str, str, float, Optional[int]]:
    start = time.time()
    request = build_chat_request(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout_seconds=timeout_seconds,
        extra_body=extra_body,
        stream=False,
    )

    completion = client.chat.completions.create(**request)
    elapsed = time.time() - start

    message = completion.choices[0].message if completion.choices else None
    answer_content = ""
    reasoning_content = ""

    if message is not None:
        answer_content = getattr(message, "content", None) or ""
        reasoning_content = getattr(message, "reasoning_content", None) or ""

    completion_tokens = safe_get_completion_tokens(getattr(completion, "usage", None))
    return answer_content, reasoning_content, elapsed, completion_tokens


def call_model_streaming(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    timeout_seconds: Optional[float],
    extra_body: Optional[dict[str, Any]],
    stream_to_stdout: bool,
    print_reasoning: bool,
    print_all_chunks: bool,
) -> Tuple[str, str, float, Optional[int]]:
    start = time.time()
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: Any = None

    request = build_chat_request(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout_seconds=timeout_seconds,
        extra_body=extra_body,
        stream=True,
    )

    stream = client.chat.completions.create(**request)

    for chunk in stream:
        if print_all_chunks:
            print(chunk.model_dump_json(exclude_none=True), flush=True)

        if not getattr(chunk, "choices", None):
            usage = getattr(chunk, "usage", None) or usage
            continue

        delta = chunk.choices[0].delta
        reasoning_delta = getattr(delta, "reasoning_content", None)
        content_delta = getattr(delta, "content", None)

        if reasoning_delta:
            reasoning_parts.append(reasoning_delta)
            if stream_to_stdout and print_reasoning:
                print(reasoning_delta, end="", flush=True)

        if content_delta:
            answer_parts.append(content_delta)
            if stream_to_stdout:
                print(content_delta, end="", flush=True)

    if stream_to_stdout:
        print(flush=True)

    elapsed = time.time() - start
    completion_tokens = safe_get_completion_tokens(usage)
    return "".join(answer_parts), "".join(reasoning_parts), elapsed, completion_tokens


def call_with_retries(
    client: OpenAI,
    *,
    transport: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    timeout_seconds: Optional[float],
    extra_body: Optional[dict[str, Any]],
    stream_to_stdout: bool,
    print_reasoning: bool,
    print_all_chunks: bool,
    max_attempts: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
) -> Tuple[str, str, float, Optional[int]]:
    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        try:
            if transport == "stream":
                return call_model_streaming(
                    client,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout_seconds=timeout_seconds,
                    extra_body=extra_body,
                    stream_to_stdout=stream_to_stdout,
                    print_reasoning=print_reasoning,
                    print_all_chunks=print_all_chunks,
                )

            return call_model_nonstreaming(
                client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout_seconds=timeout_seconds,
                extra_body=extra_body,
            )
        except RETRYABLE_EXCEPTIONS as ex:
            last_exc = ex
            if attempt >= max_attempts:
                break

            sleep_s = min(
                max_backoff_seconds,
                base_backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, 1.0),
            )
            eprint(
                f"[retry {attempt}/{max_attempts}] {type(ex).__name__}: {ex}; "
                f"sleeping {sleep_s:.2f}s"
            )
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CIPHERGRID JSONL through Qwen-VL using DashScope "
            "OpenAI-compatible Chat Completions."
        )
    )

    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Path to output CSV; appends and resumes")
    parser.add_argument("--model", default="qwen3-vl-8b-thinking", help="Qwen model name")
    parser.add_argument(
        "--base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-us.aliyuncs.com/compatible-mode/v1"),
        help="DashScope OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="Environment variable containing the DashScope API key",
    )
    parser.add_argument("--max-tokens", type=int, default=64000, help="Chat Completions max_tokens")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--transport",
        choices=["nonstream", "stream"],
        default="stream",
        help=(
            "Use stream for thinking-model parsing; nonstream is supported "
            "when the model/API allows it"
        ),
    )
    parser.add_argument("--system-prompt", default=None, help="Optional system message")
    parser.add_argument("--default-mime", default="image/png", help="MIME type used when image field is raw base64")

    # Keep decoding defaults unset unless explicitly requested, except temperature,
    # which is intentionally 0.0 in the original benchmark runner.
    parser.add_argument("--top-p", type=float, default=None, help="Optional top_p. Omitted by default.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature. Default 0.0 for reproducible benchmark runs.",
    )

    parser.add_argument("--stream", dest="stream_to_stdout", action="store_true", help="Print streamed final-answer deltas to stdout")
    parser.add_argument("--no-stream", dest="stream_to_stdout", action="store_false", help="Do not print streamed final-answer deltas")
    parser.set_defaults(stream_to_stdout=False)
    parser.add_argument("--print-reasoning", action="store_true", help="When streaming to stdout, also print reasoning_content deltas")
    parser.add_argument("--print-all-chunks", action="store_true", help="Print raw streaming chunks for debugging")

    parser.add_argument(
        "--extract-action-sequence",
        action="store_true",
        help="Write only the last CIPHERGRID-looking action sequence to the CSV response column",
    )
    parser.add_argument(
        "--save-reasoning-output",
        default=None,
        help="Optional CSV path for id,reasoning. Useful for inspecting thinking traces separately.",
    )
    parser.add_argument(
        "--enable-thinking-extra-body",
        action="store_true",
        help="Deprecated/no-op: thinking is always enabled at the maximum available budget.",
    )
    parser.add_argument(
        "--vl-high-resolution-images",
        action="store_true",
        help="Pass extra_body={'vl_high_resolution_images': True}. Off by default.",
    )

    parser.add_argument("--max-attempts", type=int, default=5, help="Retry attempts per record for transient failures")
    parser.add_argument("--base-backoff", type=float, default=1.0, help="Base backoff in seconds")
    parser.add_argument("--max-backoff", type=float, default=30.0, help="Maximum backoff in seconds")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional cap on number of input rows processed in this run")
    parser.add_argument("--skip-empty-response", action="store_true", help="Treat empty model outputs as failures and write nothing for that row")

    return parser.parse_args()


def ensure_reasoning_csv(path: Optional[Path]) -> None:
    """Create the optional reasoning CSV and header if needed."""
    if path is None:
        return

    needs_header = (not path.exists()) or (path.stat().st_size == 0)
    if not needs_header:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "reasoning"])


def append_reasoning_row(path: Optional[Path], record_id: str, reasoning: str) -> None:
    """Append a reasoning trace when --save-reasoning-output is set."""
    if path is None:
        return

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([record_id, reasoning])
        handle.flush()


def get_required_api_key(env_var_name: str) -> str:
    """Read the API key from the requested environment variable or fail clearly."""
    api_key = os.getenv(env_var_name)
    if not api_key:
        raise SystemExit(
            f"Missing {env_var_name}. Set it before running, for example:\n"
            f"  export {env_var_name}='your-api-key-here'"
        )
    return api_key


def validate_input_record(record: Dict[str, Any], line_no: int) -> list[str]:
    """Return a list of required fields missing from an input record."""
    return [field for field in REQUIRED_FIELDS if field not in record]


def build_prompt(record: Dict[str, Any]) -> str:
    """Construct the benchmark prompt for one JSONL record."""
    prompt_base = str(record["prompt-base"])
    query = str(record["query"])
    return prompt_base + "\n" + query + "\nAnswer only with the correct response!\n"


def main() -> None:
    raise_csv_field_limit()
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    reasoning_output_path = Path(args.save_reasoning_output) if args.save_reasoning_output else None

    if not input_path.exists():
        raise SystemExit(f"Input JSONL not found: {input_path}")

    api_key = get_required_api_key(args.api_key_env)

    done_ids = load_done_ids(output_path)
    if done_ids:
        eprint(f"[resume] Found {len(done_ids)} completed ids in existing CSV; will skip them.")

    ensure_csv_header(output_path)
    ensure_reasoning_csv(reasoning_output_path)

    client = OpenAI(api_key=api_key, base_url=args.base_url, max_retries=5, timeout=args.timeout)
    extra_body = build_extra_body(args)

    total = 0
    skipped = 0
    wrote = 0
    failed = 0
    processed = 0

    for line_no, record in read_jsonl(input_path):
        total += 1

        if args.sample_size is not None and processed >= args.sample_size:
            break

        missing = validate_input_record(record, line_no)
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
            prompt_text = build_prompt(record)
            image_data_url = normalize_image_to_data_url(
                str(record["image"]),
                default_mime=args.default_mime,
            )
            messages = build_messages(prompt_text, image_data_url, args.system_prompt)

            if args.transport == "stream" and args.stream_to_stdout:
                print(f"\n--- id={record_id} line={line_no} ---", flush=True)

            answer_text, reasoning_text, elapsed, completion_tokens = call_with_retries(
                client,
                transport=args.transport,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_seconds=args.timeout,
                extra_body=extra_body,
                stream_to_stdout=args.stream_to_stdout,
                print_reasoning=args.print_reasoning,
                print_all_chunks=args.print_all_chunks,
                max_attempts=args.max_attempts,
                base_backoff_seconds=args.base_backoff,
                max_backoff_seconds=args.max_backoff,
            )

            final_response = maybe_extract_action_sequence(
                answer_text,
                enabled=args.extract_action_sequence,
            )

            if args.skip_empty_response and not final_response.strip():
                failed += 1
                eprint(f"[id={record_id} line={line_no}] empty response; skipping (no write)")
                continue

            append_csv_row(output_path, (record_id, final_response, round(elapsed, 6), completion_tokens))
            append_reasoning_row(reasoning_output_path, record_id, reasoning_text)
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
