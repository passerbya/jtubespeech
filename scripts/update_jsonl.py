#!/usr/bin/python
# coding: utf-8

import json
import argparse
import shlex

# 解析 rename.sh 文件中的重命名映射
def parse_rename_script(rename_sh_path):
    rename_map = {}
    with open(rename_sh_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("mv "):
                parts = shlex.split(line.strip())
                if len(parts) == 3:
                    old_path = parts[1]
                    new_path = parts[2]
                    rename_map[old_path] = new_path
    return rename_map

# 更新 JSONL 文件中的 filename 字段
# 保持原来的顺序
def update_jsonl(dnsmos_jsonl_path, rename_map):
    updated_records = []
    seen_filenames = set()

    with open(dnsmos_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            original_filename = record["filename"]
            if original_filename in rename_map:
                record["filename"] = rename_map[original_filename]

            # 根据 filename 去重，保持顺序
            if record["filename"] not in seen_filenames:
                updated_records.append(record)
                seen_filenames.add(record["filename"])

    return updated_records

# 保存更新后的记录到新的 JSONL 文件
def save_updated_jsonl(updated_records, output_jsonl_path):
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update JSONL filenames based on rename.sh mappings.")
    parser.add_argument("--rename_sh_path", type=str, help="Path to the rename.sh file.")
    parser.add_argument("--dnsmos_jsonl_path", type=str, help="Path to the dnsmos.jsonl file.")
    parser.add_argument("--output_jsonl_path", type=str, help="Path to save the updated JSONL file.")

    args = parser.parse_args()

    # 解析 rename.sh
    _rename_map = parse_rename_script(args.rename_sh_path)

    # 更新 JSONL 文件
    _updated_records = update_jsonl(args.dnsmos_jsonl_path, _rename_map)

    # 保存结果
    save_updated_jsonl(_updated_records, args.output_jsonl_path)

    print(f"Updated JSONL file saved to {args.output_jsonl_path}")

