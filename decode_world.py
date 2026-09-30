#!/usr/bin/env python3
import sys
import argparse


def decode_world(encoded: str) -> str:
    """
    Decode a flat encoded world string into wa:-formatted rows
    using 'wa' as the row delimiter.
    """
    s = encoded.strip()
    if not s.startswith("wa"):
        raise ValueError("Encoded string must start with 'wa'")

    rows = []
    i = 0

    while i < len(s):
        if s[i:i+2] != "wa":
            raise ValueError(f"Expected 'wa' at position {i}")
        i += 2  # consume 'wa'

        row_tokens = []
        while i < len(s) and s[i:i+2] != "wa":
            row_tokens.append(s[i:i+2])
            i += 2

        rows.append("wa:" + "-".join(row_tokens))

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Decode an encoded world string.")
    parser.add_argument("encoded", help="Encoded world string (no delimiters)")
    args = parser.parse_args()

    print(decode_world(args.encoded))


if __name__ == "__main__":
    main()
