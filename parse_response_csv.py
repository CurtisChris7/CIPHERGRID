import csv
import re
import sys


def set_max_csv_field_size() -> None:
    """
    Raise the CSV field size limit as high as the platform allows.
    Python's default limit is often too small for long model responses.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def extract_last_bold(text: str) -> str:
    """
    Extract the content inside the last pair of **...**.
    Returns the original text if no such match is found.
    """
    if text is None:
        return ""

    matches = list(re.finditer(r"\-\-\-(.*?)\-\-\-", text, flags=re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    return text.strip()


def rewrite_csv(input_csv: str, output_csv: str) -> None:
    set_max_csv_field_size()

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("Input CSV has no header row.")

        if "response" not in fieldnames:
            raise ValueError("Input CSV does not contain a 'response' column.")

        rows = []
        for row in reader:
            row["response"] = extract_last_bold(row["response"])
            rows.append(row)

    with open(output_csv, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_response_csv.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    rewrite_csv(input_csv, output_csv)
    print(f"Wrote cleaned CSV to: {output_csv}")
