#!/usr/bin/env python3
from pathlib import Path


# --- edit only if your paths change ---
LOCAL_DIR = "/Users/duncanroepke/Desktop/preprocessed sermon dataset/sermons"
PROJECT_HOST = "czucjlmdhrrglxdbyrif.supabase.co"  # from your SUPABASE_URL
BUCKET = "sermons"
# --------------------------------------


def quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def main():
    p = Path(LOCAL_DIR).expanduser().resolve()
    files = sorted(p.glob("*.txt"))
    if not files:
        print("-- No .txt files found.")
        return

    rows = []
    for f in files:
        sermon_id = f.stem  # assumes {sermon_id}.txt
        url = f"https://{PROJECT_HOST}/storage/v1/object/public/{BUCKET}/{f.name}"
        rows.append((sermon_id, url))

    print("-- Paste everything below into Supabase > SQL Editor and run once")
    print("WITH data(sermon_id, transcript_file) AS (")
    print("  VALUES")
    vals = []
    for sid, url in rows:
        vals.append(f"    ({quote(sid)}, {quote(url)})")
    print(",\n".join(vals))
    print(")")
    print("UPDATE public.sermons_metadata AS t")
    print("SET transcript_file = d.transcript_file")
    print("FROM data d")
    print("WHERE t.sermon_id = d.sermon_id")
    print("  AND (t.transcript_file IS DISTINCT FROM d.transcript_file);")


if __name__ == "__main__":
    main()
