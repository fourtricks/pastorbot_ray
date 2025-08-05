import json
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

with open("sermons_metadata.json", encoding="utf-8") as f:
    data = json.load(f)

for sermon in data:
    supabase.table("sermons_metadata").insert(sermon).execute()

print("✅ All metadata uploaded to Supabase")
