# list_finetune_events.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JOB_ID = "ftjob-cxUZDTHBrqkhRgcJcsGPPfKs"  # your job ID

# Fetch up to 200 events
events = client.fine_tuning.jobs.list_events(
    fine_tuning_job_id=JOB_ID,
    limit=200
)

print(f"Events for job {JOB_ID}:\n")
for evt in events.data:
    # timestamp, level (info/warning/error), and message
    ts = evt.created_at
    lvl = evt.level
    msg = evt.message
    print(f"[{ts}] {lvl.upper()}: {msg}")
