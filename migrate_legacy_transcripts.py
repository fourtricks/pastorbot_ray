#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client

# ================== CONFIG YOU PROVIDED ==================
LOCAL_DIR = "/Users/duncanroepke/Desktop/preprocessed sermon dataset/sermons"
BUCKET = "sermons"
TABLE = "sermons_metadata"

# Behavior:
DRY_RUN = False          # <-- set to False to actually upload/update
UPSERT_STORAGE = True   # upsert files in Storage so re-runs are idempotent
# ========================================================


def get_public_url_str(storage_client, bucket: str, path: str) -> Optional[str]:
    """Return a plain string public URL (client may return dict or str)."""
    res = storage_client.from_(bucket).get_public_url(path)
    if isinstance(res, dict):
        return res.get("publicUrl")
    return res


def upload_text_file(storage_client, bucket: str, path: str, data: bytes) -> None:
    """
    Uploads a file with correct headers. If it already exists, remove then upload (idempotent).
    """
    try:
        storage_client.from_(bucket).upload(
            path,
            data,
            {"contentType": "text/plain"}  # correct header name; do not pass booleans here
        )
    except Exception as e:
        # Many clients throw 409/conflict if the object exists; handle "upsert" manually.
        msg = str(e).lower()
        if "already exists" in msg or "409" in msg or "conflict" in msg:
            # remove then try once more
            storage_client.from_(bucket).remove([path])
            storage_client.from_(bucket).upload(
                path,
                data,
                {"contentType": "text/plain"}
            )
        else:
            raise


def fetch_row(sb, table: str, sermon_id: str) -> Optional[dict]:
    res = sb.table(table).select("*").eq("sermon_id", sermon_id).limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None


def update_transcript_file(sb, table: str, sermon_id: str, new_url: str) -> None:
    sb.table(table).update({"transcript_file": new_url}).eq("sermon_id", sermon_id).execute()


def main():
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("ERROR: SUPABASE_URL and/or SUPABASE_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)

    sb = create_client(supabase_url, supabase_key)

    src_dir = Path(LOCAL_DIR).expanduser().resolve()
    if not src_dir.exists():
        print(f"ERROR: local dir not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(src_dir.glob("*.txt"))
    print(f"[info] Found {len(txt_files)} .txt files in: {src_dir}")
    print(f"[info] Bucket: {BUCKET} | Table: {TABLE} | DRY_RUN={DRY_RUN} | UPSERT_STORAGE={UPSERT_STORAGE}")

    updated, skipped, missing, errors = 0, 0, 0, 0

    for file_path in txt_files:
        sermon_id = file_path.stem                 # assumes filename == {sermon_id}.txt
        storage_path = file_path.name              # store at bucket root with same filename

        try:
            # 1) Ensure DB row exists
            row = fetch_row(sb, TABLE, sermon_id)
            if not row:
                missing += 1
                print(f"[warn] No DB row for sermon_id={sermon_id} — skipping (you said there shouldn't be any).")
                continue

            # 2) Upload the file to Storage (idempotent if UPSERT_STORAGE)
            if DRY_RUN:
                print(f"[dry-run] Would upload {file_path} -> {BUCKET}/{storage_path}")
            else:
                with open(file_path, "rb") as fh:
                    upload_text_file(sb.storage, BUCKET, storage_path, fh.read())

            # 3) Compute public URL
            public_url = get_public_url_str(sb.storage, BUCKET, storage_path)
            if not public_url:
                raise RuntimeError(f"Could not obtain public URL for {storage_path}")

            # 4) Decide whether to update transcript_file
            current_url = (row.get("transcript_file") or "").strip()

            if current_url == public_url:
                # Already matches the bucket URL (e.g., your most recent 'fruit of love' row)
                skipped += 1
                print(f"[skip] {sermon_id} already has current public URL.")
                continue

            # If it's not equal, we update (per your rule: all others should be migrated to bucket)
            if DRY_RUN:
                print(f"[dry-run] Would update transcript_file for {sermon_id} -> {public_url}")
            else:
                update_transcript_file(sb, TABLE, sermon_id, public_url)
                updated += 1
                print(f"[ok] Updated transcript_file for {sermon_id}")

        except Exception as e:
            errors += 1
            print(f"[error] {sermon_id}: {e}")

    print("\n==== Summary ====")
    print(f"Updated: {updated}")
    print(f"Skipped (already current): {skipped}")
    print(f"Missing DB rows: {missing}")
    print(f"Errors: {errors}")
    print("Done.")


if __name__ == "__main__":
    main()
