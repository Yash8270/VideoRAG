import json
import sys

log_path = r"C:\Users\Yash Limbachiya\.gemini\antigravity\brain\6e00a6ac-bec7-43c6-a713-7be22eee2a83\.system_generated\logs\transcript.jsonl"
with open(log_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "tuple index out of range" in line:
        start = max(0, i - 15)
        for j in range(start, i + 5):
            print(lines[j])
        break
