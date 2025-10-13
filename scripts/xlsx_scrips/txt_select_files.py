#!/usr/bin/env python3
import shutil
import os
import sys

TXT_FILE   = './'   # 输入 txt 文件
OUTPUT_DIR = 'output'     # 输出根目录

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TXT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            folder = line.strip()
            if not folder:          # 跳过空行
                continue
            if not os.path.isdir(folder):
                print(f"[WARN] 目录不存在: {folder}")
                continue

            base_name = os.path.basename(folder.rstrip(os.sep))
            dest = os.path.join(OUTPUT_DIR, base_name)
            shutil.copytree(folder, dest, dirs_exist_ok=True)
            print(f"[COPY] {folder}  ->  {dest}")

    print("[DONE] 全部完成！")

if __name__ == '__main__':
    main()