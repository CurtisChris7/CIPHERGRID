"""
Convert multiple lines like:

  wa:ea-na-la-ca-ha-ea-pa
  wa:ea-oa-ha-pa-oa-oa-pa-xa-pa

into ONE continuous string:

  waeanalacahaeapawaeaoahapaoaoapaxapa

Rules:
- Replace 'wa:' with 'wa'
- Remove all '-' delimiters
- Remove all newline boundaries
- Preserve character order exactly
"""

import sys
import argparse


def undelim_all(text: str) -> str:
    lines = text.splitlines()
    out = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("wa:"):
            line = "wa" + line[3:]
        out.append(line.replace("-", ""))

    return "".join(out)


def main():
    p = argparse.ArgumentParser(description="Remove delimiters and newlines from wa-strings.")
    p.add_argument("--in", dest="infile", default=None, help="Input file (default: stdin)")
    p.add_argument("--out", dest="outfile", default=None, help="Output file (default: stdout)")
    args = p.parse_args()

    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    result = undelim_all(text)

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
