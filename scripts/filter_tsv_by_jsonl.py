#!/usr/bin/python
# coding: utf-8

import argparse
import json
import os


def filter_tsv_by_jsonl(tsv_path, jsonl_path, output_path, threshold=3.6):
    # Load JSONL data into a dictionary with filename as the key
    jsonl_data = {}
    with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            record = json.loads(line.strip())
            jsonl_data[record['filename']] = record

    # Process the TSV file and filter entries based on P808_MOS
    filtered_lines = []
    with open(tsv_path, 'r', encoding='utf-8') as tsv_file:
        for line in tsv_file:
            parts = line.strip().split('\t')
            if parts[0] in jsonl_data:
                record = jsonl_data[parts[0]]
                if record.get('P808_MOS', 0) > threshold:
                    filtered_lines.append(line.strip())

    # Write the filtered lines to the output TSV file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for line in filtered_lines:
            output_file.write(line + '\n')

    print(f"Filtered {len(filtered_lines)} lines and saved to {output_path}.")


def main():
    parser = argparse.ArgumentParser(description="Filter a TSV file based on JSONL data.")
    parser.add_argument("--tsv", required=True, help="Path to the input TSV file.")
    parser.add_argument("--jsonl", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output filtered TSV file.")
    parser.add_argument("--threshold", type=float, default=3.6, help="P808_MOS threshold for filtering (default: 3.6).")

    args = parser.parse_args()

    # Ensure the input files exist
    if not os.path.exists(args.tsv):
        print(f"Error: TSV file '{args.tsv}' does not exist.")
        return

    if not os.path.exists(args.jsonl):
        print(f"Error: JSONL file '{args.jsonl}' does not exist.")
        return

    # Run the filtering process
    filter_tsv_by_jsonl(args.tsv, args.jsonl, args.output, args.threshold)


if __name__ == "__main__":
    main()