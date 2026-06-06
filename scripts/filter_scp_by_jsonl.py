#!/usr/bin/python
# coding: utf-8

import argparse
import json
import os


def filter_scp_by_jsonl(jsonl_path, output_path, logic):
    # Load JSONL data into a dictionary with filename as the key
    filtered_lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            record = json.loads(line.strip())
            ovrl_pass = record.get('OVRL', 0) > 3.2
            p808_mos_pass = record.get('P808_MOS', 0) > 3.6
            if (logic == 'or' and (ovrl_pass or p808_mos_pass)) or (
                logic == 'and' and (ovrl_pass and p808_mos_pass)
            ):
                filtered_lines.append(record['filename'].replace('/data1/', '/data2/'))

    # Write the filtered lines to the output SCP file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for line in filtered_lines:
            output_file.write(f'{line}\n')

    print(f"Filtered {len(filtered_lines)} lines and saved to {output_path}.")


def main():
    parser = argparse.ArgumentParser(description="Create a SCP file based on JSONL data.")
    parser.add_argument("--jsonl", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output filtered SCP file.")
    parser.add_argument(
        "--logic",
        choices=["or", "and"],
        default="or",
        help="Filtering logic for OVRL and P808_MOS thresholds. Default: or.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"Error: JSONL file '{args.jsonl}' does not exist.")
        return

    # Run the filtering process
    filter_scp_by_jsonl(args.jsonl, args.output, args.logic)


if __name__ == "__main__":
    main()
