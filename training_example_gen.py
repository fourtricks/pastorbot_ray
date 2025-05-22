# training_example_gen.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Total number of examples to generate
TOTAL_EXAMPLES = 300

examples = []

for _ in tqdm(range(TOTAL_EXAMPLES), desc="Generating training examples"):
    # Single GPT-4 Turbo call per example
    res = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Rev. Ray Choi. Generate a user question and his scriptural, "
                    "pastoral answer as a pair. Separate question and answer with a newline."
                )
            }
        ],
        temperature=0.7,
        n=1
    )
    text = res.choices[0].message.content.strip()
    if "\n" in text:
        q, a = text.split("\n", 1)
    else:
        # Fallback parsing if newline missing
        parts = text.split("Pastor:")
        q = parts[0].replace("User:", "").strip()
        a = parts[1].strip() if len(parts) > 1 else ""

    examples.append({
        "prompt": f"User: {q}\nPastor:",
        "completion": f"  {a}"
    })

# Save to JSONL
with open("pastor_ray_training.jsonl", "w", encoding="utf8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"✅ Generated {len(examples)} examples and saved to pastor_ray_training.jsonl")
