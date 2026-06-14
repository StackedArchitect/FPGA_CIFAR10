import json

transcript_path = r"C:\Users\ADMIN\.gemini\antigravity\brain\b162c280-c2d6-4926-a6ad-9842958a6db5\.system_generated\logs\transcript.jsonl"

with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if "hysteresis testbench" in line.lower():
            try:
                data = json.loads(line)
                content = data.get("content", "")
                for l in content.split("\n"):
                    if "active count" in l.lower() or "mask_gen" in l.lower() or "done at" in l.lower():
                        print(f"Line {i}: {l.strip()}")
            except:
                pass
