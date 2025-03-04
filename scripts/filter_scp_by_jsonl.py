#!/usr/bin/python
# coding: utf-8

import argparse
import json
import os


def filter_scp_by_jsonl(jsonl_path, output_path):
    # Load JSONL data into a dictionary with filename as the key
    filtered_lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            record = json.loads(line.strip())
            if record.get('MOS_OVRL', 0) > 3.0: #P.804 2.75-->P.808 3.2
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

    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"Error: JSONL file '{args.jsonl}' does not exist.")
        return

    # Run the filtering process
    filter_scp_by_jsonl(args.jsonl, args.output)


if __name__ == "__main__":
    main()