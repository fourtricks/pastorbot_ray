# convert_to_chat_finetune.py

import json

INPUT_FILE = "pastor_ray_training.jsonl"
OUTPUT_FILE = "pastor_ray_chat_finetune.jsonl"

with open(INPUT_FILE, "r", encoding="utf8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf8") as fout:

    for line in fin:
        example = json.loads(line)
        prompt = example["prompt"]
        completion = example["completion"]

        # Build the chat-completions style example
        chat_example = {
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": completion}
            ]
        }
        fout.write(json.dumps(chat_example, ensure_ascii=False) + "\n")

print(f"Wrote chat‑formatted examples to {OUTPUT_FILE}")
