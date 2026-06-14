import json
import re

transcript_path = r"C:\Users\ADMIN\.gemini\antigravity\brain\b162c280-c2d6-4926-a6ad-9842958a6db5\.system_generated\logs\transcript.jsonl"

print("Searching transcript...")
with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if "accuracy" in line.lower() or "acc" in line.lower() or "percent" in line.lower() or "accuracy_tolerance" in line.lower():
            # Find lines containing float baseline, TTQ + BN baseline, Hysteresis or Thresholding and percentage signs or numbers
            if any(term in line.lower() for term in ["accuracy", "acc", "drop", "accuracy_tolerance"]):
                # Let's print out lines that might contain the final results
                if any(x in line for x in ["90.", "89.", "88.", "87.", "86.", "85.", "84.", "83.", "82.", "81.", "80."]) and "%" in line:
                    # Parse json content to see
                    try:
                        data = json.loads(line)
                        content = data.get("content", "")
                        for l in content.split("\n"):
                            if any(x in l for x in ["90.", "89.", "88.", "87.", "86.", "85.", "84.", "83.", "82.", "81.", "80."]) and "%" in l:
                                print(f"Line {i}: {l.strip()}")
                    except:
                        pass
