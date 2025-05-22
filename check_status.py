# check_status.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

job_id = "ftjob-cxUZDTHBrqkhRgcJcsGPPfKs"  # your fine‑tune job ID

# Call retrieve with the job ID as the first (and only) argument
status = client.fine_tuning.jobs.retrieve(job_id)

print("Status:", status.status)

if status.status == "succeeded":
    print("🎉 Finished! Your fine‑tuned model is:", status.fine_tuned_model)
elif status.status == "failed":
    print("⚠️ Fine‑tune failed. Check the logs on the OpenAI dashboard.")
