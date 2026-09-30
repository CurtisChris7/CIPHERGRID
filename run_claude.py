#!/usr/bin/env python3
"""
Claude JSONL -> CSV benchmark runner for CIPHERGRID-style vision + text tasks.

Input JSONL record fields:
  - prompt-base (string)
  - query (string)
  - image (string; base64 image bytes or a data URL)
  - id (string or int)

Output CSV columns:
  id,response,time,tokens,reasoning_tokens

Behavior:
  - Resume: skips ids with a non-empty response in the existing output CSV.
  - Failure: writes nothing for the failed record and continues.
  - Tokens: required; records without usage metadata are treated as failures.
  - Response: stores the full model output in `response`, with newlines serialized
    so each result occupies one physical CSV line.
  - Retries transient Claude API failures such as timeouts, overloads, and 5xxs.

Authentication:
  - Reads the API key from --api-key or ANTHROPIC_API_KEY.
  - Do not hardcode API keys or commit .env files.
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
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set, Tuple

from anthropic import Anthropic
from anthropic import APIError, APIStatusError, APITimeoutError  # type: ignore


REQUIRED_FIELDS = ["prompt-base", "query", "image", "id"]
CSV_COLUMNS = ["id", "response", "time", "tokens", "reasoning_tokens"]


def raise_csv_field_limit() -> None:
    """
    Allow reading CSV rows with very large response fields.
    Python's csv module default can be too small for long model outputs.
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


def get_attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def is_data_url(value: str) -> bool:
    return value.startswith("data:") and ";base64," in value


def extract_b64_from_data_url(data_url: str) -> str:
    if not is_data_url(data_url):
        raise ValueError("Value is not a base64 data URL")
    return data_url.split(";base64,", 1)[1]


def normalize_image_to_b64(image_b64_or_dataurl: str) -> str:
    """
    Accept either raw base64 image bytes or a base64 data URL and return raw base64.
    """
    value = (image_b64_or_dataurl or "").strip()
    if not value:
        raise ValueError("Empty image field")

    image_b64 = extract_b64_from_data_url(value) if is_data_url(value) else value

    try:
        base64.b64decode(image_b64, validate=True)
    except Exception as ex:
        raise ValueError(f"Invalid base64 image and not a valid data URL: {ex}") from ex

    return image_b64


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


def serialize_for_csv(text: str) -> str:
    """
    Keep each CSV record on one physical line while preserving response content.
    """
    value = "" if text is None else str(text)
    value = value.replace("\\", "\\\\")
    value = value.replace("\r\n", "\\n")
    value = value.replace("\n", "\\n")
    value = value.replace("\r", "\\n")
    return value


def extract_text_from_message(message: Any) -> str:
    parts: list[str] = []
    for block in get_attr(message, "content") or []:
        if get_attr(block, "type") == "text":
            text = get_attr(block, "text")
            if text:
                parts.append(str(text))
    return "".join(parts)


def extract_claude_usage(message: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Return (input_tokens, output_tokens, reasoning_tokens).
    """
    usage = get_attr(message, "usage")
    input_tokens = get_attr(usage, "input_tokens")
    output_tokens = get_attr(usage, "output_tokens")
    reasoning_tokens = get_attr(usage, "thinking_tokens") or get_attr(usage, "reasoning_tokens")

    def to_int(value: Any) -> Optional[int]:
        return int(value) if isinstance(value, (int, float)) else None

    return to_int(input_tokens), to_int(output_tokens), to_int(reasoning_tokens)


def safe_response_preview(response: Any) -> Dict[str, Any]:
    """
    Safely collect useful details from an API response object without letting
    logging itself crash on an unread or already-consumed streaming response.
    """
    out: Dict[str, Any] = {}
    if response is None:
        return out

    out["response_class"] = type(response).__name__
    out["request_id"] = getattr(response, "request_id", None)

    for attr_name in ("is_closed", "is_stream_consumed"):
        try:
            if hasattr(response, attr_name):
                out[f"response_{attr_name}"] = getattr(response, attr_name)
        except Exception:
            pass

    try:
        if hasattr(response, "read"):
            response.read()
    except Exception as ex:
        out["response_read_error"] = f"{type(ex).__name__}: {ex}"
        return out

    try:
        response_text = getattr(response, "text", None)
        if response_text:
            out["response_text"] = response_text
    except Exception as ex:
        out["response_text_error"] = f"{type(ex).__name__}: {ex}"

    return out


def format_anthropic_exception(ex: Exception) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "type": type(ex).__name__,
        "message": str(ex),
    }

    if isinstance(ex, APITimeoutError):
        info["kind"] = "timeout"
        return info

    if isinstance(ex, APIStatusError):
        info["kind"] = "http"
        info["status_code"] = getattr(ex, "status_code", None)

        response = getattr(ex, "response", None)
        info["request_id"] = getattr(response, "request_id", None) or getattr(ex, "request_id", None)
        info.update(safe_response_preview(response))

        msg_lower = str(ex).lower()
        if "overloaded" in msg_lower:
            info["error_subtype"] = "overloaded_error"
        elif "internal server error" in msg_lower:
            info["error_subtype"] = "internal_server_error"

        return info

    if isinstance(ex, APIError):
        info["kind"] = "api_error"
        info["request_id"] = getattr(ex, "request_id", None)
        return info

    info["kind"] = "exception"
    return info


def is_retriable_anthropic_error(ex: Exception) -> Tuple[bool, str]:
    if isinstance(ex, APITimeoutError):
        return True, "timeout"

    if isinstance(ex, APIStatusError):
        msg_lower = str(ex).lower()
        status_code = getattr(ex, "status_code", None)

        if "overloaded" in msg_lower:
            return True, "overloaded_error"
        if "internal server error" in msg_lower:
            return True, "internal_server_error"
        if status_code is not None and status_code >= 500:
            return True, f"http_{status_code}"

    return False, type(ex).__name__


def build_prompt(prompt_base: str, query: str) -> str:
    """
    Preserve the prompt adaptation used in the original Claude runner.
    """
    return (
        "This is an academic pattern recognition study using abstract grid puzzles.\n"
        "Given the examples above showing input-output pairs, "
        "what would be the appropriate output for this input sequence?\n\n"
        + prompt_base.replace("You MUST provide a response", "Please provide your response")
        + "\n"
        + query
        + "\n\nResponse (in the same format as the examples):"
    )


def build_request(
    *,
    model: str,
    prompt_text: str,
    image_b64: str,
    image_mime: str,
    max_output_tokens: int,
    effort: str,
    enable_adaptive_thinking: bool,
) -> Dict[str, Any]:
    """
    Build the Claude Messages API request.

    The original benchmark runner always sent adaptive thinking and an output
    effort value. This preserves that request shape for reproducibility while
    retaining the --adaptive-thinking CLI flag for compatibility.
    """
    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_mime,
                "data": image_b64,
            },
        },
        {
            "type": "text",
            "text": prompt_text,
        },
    ]

    request: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": user_content}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
    }

    # The flag is intentionally retained for CLI compatibility. It does not
    # alter the default request because the uploaded runner always enabled
    # adaptive thinking.
    _ = enable_adaptive_thinking

    return request


def call_claude(
    client: Anthropic,
    *,
    model: str,
    prompt_text: str,
    image_b64: str,
    image_mime: str,
    max_output_tokens: int,
    effort: str,
    enable_adaptive_thinking: bool,
    stream_to_stdout: bool,
) -> Tuple[str, float, Optional[int], Optional[int], Any]:
    """
    Return (full_text, elapsed_seconds, total_tokens, reasoning_tokens, final_message).
    """
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be >= 1")

    request = build_request(
        model=model,
        prompt_text=prompt_text,
        image_b64=image_b64,
        image_mime=image_mime,
        max_output_tokens=max_output_tokens,
        effort=effort,
        enable_adaptive_thinking=enable_adaptive_thinking,
    )

    start = time.time()
    text_parts: list[str] = []
    final_message = None

    if stream_to_stdout:
        with client.messages.stream(**request) as stream:
            try:
                for text_delta in stream.text_stream:
                    text_parts.append(text_delta)
                    print(text_delta, end="", flush=True)
                print(flush=True)
            finally:
                try:
                    final_message = stream.get_final_message()
                except Exception:
                    final_message = None
    else:
        final_message = client.messages.create(**request)
        text = extract_text_from_message(final_message)
        if text:
            text_parts.append(text)

    elapsed = time.time() - start
    full_text = "".join(text_parts)

    if not full_text.strip() and final_message is not None:
        full_text = extract_text_from_message(final_message)

    input_tokens, output_tokens, reasoning_tokens = extract_claude_usage(final_message)
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )

    return full_text, elapsed, total_tokens, reasoning_tokens, final_message


def call_claude_with_retries(
    client: Anthropic,
    *,
    model: str,
    prompt_text: str,
    image_b64: str,
    image_mime: str,
    max_output_tokens: int,
    effort: str,
    enable_adaptive_thinking: bool,
    stream_to_stdout: bool,
    max_attempts: int,
    initial_delay: float,
) -> Tuple[str, float, Optional[int], Optional[int], Any]:
    delay = initial_delay
    last_ex: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return call_claude(
                client,
                model=model,
                prompt_text=prompt_text,
                image_b64=image_b64,
                image_mime=image_mime,
                max_output_tokens=max_output_tokens,
                effort=effort,
                enable_adaptive_thinking=enable_adaptive_thinking,
                stream_to_stdout=stream_to_stdout,
            )
        except Exception as ex:
            last_ex = ex
            retriable, reason = is_retriable_anthropic_error(ex)

            if not retriable or attempt >= max_attempts:
                raise

            sleep_seconds = delay * (1.0 + random.uniform(0.0, 0.25))
            eprint(
                f"[retry {attempt}/{max_attempts}] transient Claude failure "
                f"({reason}); sleeping {sleep_seconds:.2f}s"
            )
            time.sleep(sleep_seconds)
            delay = min(delay * 2.0, 30.0)

    assert last_ex is not None
    raise last_ex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Path to output CSV (append + resume)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-output-tokens", type=int, default=128000)
    parser.add_argument("--image-mime", default="image/png")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "max"])
    parser.add_argument(
        "--adaptive-thinking",
        action="store_true",
        help="Send thinking={type:'adaptive'} at top level",
    )
    parser.add_argument("--stream", dest="stream", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--debug", action="store_true", help="Print full tracebacks and rich API error info")
    parser.add_argument("--fail-log", default=None, help="Write failures to this JSONL path")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key. If omitted, reads ANTHROPIC_API_KEY from the environment.",
    )
    return parser.parse_args()


def get_api_key(cli_key: Optional[str]) -> str:
    api_key = cli_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing Anthropic API key. Provide --api-key or set ANTHROPIC_API_KEY, for example:\n"
            "  export ANTHROPIC_API_KEY='your-key-here'"
        )
    return api_key


def log_record_failure(
    *,
    fail_log: Optional[Path],
    line_no: int,
    record_id: Any,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "provider": "claude",
        "line_no": line_no,
        "id": record_id,
        "error": message,
    }
    if extra:
        payload.update(extra)
    append_fail_log(fail_log, payload)


def handle_refusal_or_empty(
    *,
    fail_log: Optional[Path],
    rec_id: str,
    line_no: int,
    final_message: Any,
    message: str,
) -> None:
    stop_type = get_attr(final_message, "type")
    stop_reason = get_attr(final_message, "stop_reason")
    content = get_attr(final_message, "content")
    usage = get_attr(final_message, "usage")

    eprint(f"[id={rec_id} line={line_no}] {message} (no write)")
    eprint(f"  final_message.type={stop_type!r}")
    eprint(f"  stop_reason={stop_reason!r}")
    eprint(f"  usage={usage!r}")
    eprint(f"  content_repr={content!r}")

    log_record_failure(
        fail_log=fail_log,
        line_no=line_no,
        record_id=rec_id,
        message=message,
        extra={
            "final_message_type": repr(stop_type),
            "stop_reason": repr(stop_reason),
            "usage_repr": repr(usage),
            "content_repr": repr(content),
        },
    )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    fail_log = Path(args.fail_log) if args.fail_log else None

    if not input_path.exists():
        raise SystemExit(f"Input JSONL not found: {input_path}")

    api_key = get_api_key(args.api_key)

    done_ids = load_done_ids(output_path)
    if done_ids:
        eprint(f"[resume] Found {len(done_ids)} completed ids in existing CSV; skipping them.")

    ensure_csv_header(output_path)
    client = Anthropic(api_key=api_key)

    total = 0
    skipped = 0
    wrote = 0
    failed = 0

    for line_no, record in read_jsonl(input_path):
        total += 1

        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            failed += 1
            msg = f"missing fields {missing}; skipping"
            eprint(f"[line {line_no}] {msg}")
            log_record_failure(
                fail_log=fail_log,
                line_no=line_no,
                record_id=record.get("id"),
                message=msg,
            )
            continue

        rec_id = str(record.get("id"))
        if rec_id in done_ids:
            skipped += 1
            continue

        try:
            prompt_text = build_prompt(str(record["prompt-base"]), str(record["query"]))
            image_b64 = normalize_image_to_b64(str(record["image"]))

            if args.stream:
                print(f"\n--- id={rec_id} (line {line_no}) ---", flush=True)

            full_text, elapsed, total_tokens, reasoning_tokens, final_message = call_claude_with_retries(
                client,
                model=args.model,
                prompt_text=prompt_text,
                image_b64=image_b64,
                image_mime=args.image_mime,
                max_output_tokens=args.max_output_tokens,
                effort=args.effort,
                enable_adaptive_thinking=args.adaptive_thinking,
                stream_to_stdout=args.stream,
                max_attempts=args.retries,
                initial_delay=args.retry_delay,
            )

            stop_reason = get_attr(final_message, "stop_reason")

            if stop_reason == "refusal":
                failed += 1
                handle_refusal_or_empty(
                    fail_log=fail_log,
                    rec_id=rec_id,
                    line_no=line_no,
                    final_message=final_message,
                    message="Claude returned stop_reason=refusal; no visible content was returned",
                )
                continue

            if not full_text.strip():
                failed += 1
                handle_refusal_or_empty(
                    fail_log=fail_log,
                    rec_id=rec_id,
                    line_no=line_no,
                    final_message=final_message,
                    message="empty response; skipping",
                )
                continue

            if total_tokens is None:
                failed += 1
                usage = get_attr(final_message, "usage")
                msg = f"token usage missing from API response; usage={usage!r}"
                eprint(f"[id={rec_id} line={line_no}] {msg} (no write)")
                log_record_failure(
                    fail_log=fail_log,
                    line_no=line_no,
                    record_id=rec_id,
                    message=msg,
                    extra={"usage_repr": repr(usage)},
                )
                continue

            append_csv_row(
                output_path,
                (
                    rec_id,
                    serialize_for_csv(full_text),
                    round(elapsed, 6),
                    int(total_tokens),
                    reasoning_tokens,
                ),
            )
            done_ids.add(rec_id)
            wrote += 1

        except Exception as ex:
            failed += 1
            info = format_anthropic_exception(ex)

            eprint(f"[id={rec_id} line={line_no}] FAIL: {info.get('type')}: {info.get('message')}")
            if info.get("status_code") is not None:
                eprint(f"  status_code: {info.get('status_code')}")
            if info.get("request_id"):
                eprint(f"  request_id: {info.get('request_id')}")
            if info.get("error_subtype"):
                eprint(f"  error_subtype: {info.get('error_subtype')}")
            if info.get("response_text"):
                eprint(f"  response_text: {info.get('response_text')}")
            if info.get("response_read_error"):
                eprint(f"  response_read_error: {info.get('response_read_error')}")
            if "overloaded" in str(info.get("message", "")).lower():
                eprint("  note: request reached the API, but the stream returned an overloaded error; this is transient")
            if args.debug:
                eprint("  --- traceback ---")
                eprint(traceback.format_exc())

            append_fail_log(
                fail_log,
                {
                    "provider": "claude",
                    "line_no": line_no,
                    "id": rec_id,
                    "exception": info,
                    "traceback": traceback.format_exc() if args.debug else None,
                },
            )
            continue

    print(f"\nDone. total={total} wrote={wrote} skipped={skipped} failed={failed} -> {output_path}")
    if fail_log:
        print(f"Failures log: {fail_log}")


if __name__ == "__main__":
    main()
