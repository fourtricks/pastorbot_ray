# start_finetune.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1) Upload the JSONL file for fine‑tuning
file_resp = client.files.create(
    file=open("pastor_ray_chat_finetune.jsonl", "rb"),
    purpose="fine-tune"
)
training_file_id = file_resp.id
print("Uploaded file, id:", training_file_id)

# 2) Create the fine‑tune job, specifying n_epochs under hyperparameters
job = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model="gpt-3.5-turbo",
    hyperparameters={"n_epochs": 4}
)
print("Fine‑tune job started. ID:", job.id)
print("Initial status:", job.status)
