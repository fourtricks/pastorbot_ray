# verify_supabase_updates.py
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
print("[info] SUPABASE_URL:", url)
print("[info] SUPABASE_KEY prefix:", (key or "")[:8])

sb = create_client(url, key)

# 1) Table stats
res_total = sb.table("sermons_metadata").select("sermon_id", count="exact").execute()
print("[info] sermons_metadata total rows:", res_total.count)

# 2) How many already point to bucket?
project_host = url.split("//")[1].split("/")[0]  # e.g. czucjlmdhrrglxdbyrif.supabase.co
prefix = f"https://{project_host}/storage/v1/object/public/sermons/"
res_bucket = sb.table("sermons_metadata").select("sermon_id", count="exact").ilike("transcript_file", f"{prefix}%").execute()
print("[info] rows with transcript_file in bucket:", res_bucket.count)

# 3) Peek a specific sermon
sid = "2023-02-02-two-natures-ishmael-vs-isaac"
row = sb.table("sermons_metadata").select("sermon_id,title,transcript_file").eq("sermon_id", sid).limit(1).execute().data
print("[info] sample row:", row)
