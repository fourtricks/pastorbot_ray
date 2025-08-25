import os
import json
import openai
import streamlit as st
from slugify import slugify
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from supabase import create_client
from embed_sermons import chunk_text  # reuse your chunking function

# Load environment variables
load_dotenv()

# OpenAI + Pinecone clients
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pastor-ray-sermons")

# Supabase client (use service role key for secure inserts)
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

st.set_page_config(page_title="Sermon Uploader", layout="centered")
st.title("📖 Sermon Transcript Uploader")

# --- Session-based authentication ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.write("### Please enter password to continue")
    password = st.text_input(
        "Password",
        type="password",
        key="password_input",
        label_visibility="visible",
        placeholder=""
    )

    login_pressed = st.button("Login")
    if login_pressed or (password and not login_pressed and st.session_state.password_input != ""):
        if password == os.getenv("UPLOAD_PASSWORD"):
            st.session_state.authenticated = True
            st.success("✅ Logged in successfully!")
            st.rerun()
        elif password:
            st.error("❌ Incorrect password.")
            st.session_state.password_input = ""  # clear the field immediately
            st.rerun()

    st.stop()

# --- Upload form ---
with st.form("upload_form"):
    transcript = st.text_area("Paste the full sermon transcript", height=300)
    title = st.text_input("Sermon Title")
    date = st.date_input("Sermon Date")
    url = st.text_input("Sermon URL")
    submitted = st.form_submit_button("Submit Sermon")

if submitted:
    if not (transcript and title and date and url):
        st.error("Please fill in all fields.")
        st.stop()

    # Generate filename/id
    date_str = date.strftime("%Y-%m-%d")
    slug = slugify(title)
    sermon_id = f"{date_str}-{slug}"
    filename = f"{sermon_id}.txt"

    # --- Upload transcript to Supabase Storage ---
    try:
        file_bytes = transcript.encode("utf-8")

        supabase.storage.from_("sermons").upload(
            path=filename,
            file=file_bytes,
            file_options={
                "contentType": "text/plain; charset=utf-8",
                "cacheControl": "3600",   # must be a string
                "upsert": "true"          # must be "true"/"false", not True/False
            }
        )

        public_url_res = supabase.storage.from_("sermons").get_public_url(filename)
        public_url = public_url_res.get("publicUrl") if isinstance(public_url_res, dict) else public_url_res
        if not isinstance(public_url, str) or not public_url:
            raise RuntimeError("Could not obtain a public URL for the uploaded transcript.")

        st.success("✅ Transcript uploaded to Supabase Storage")
    except Exception as e:
        st.error(f"❌ Failed to upload transcript to Supabase Storage: {e}")
        st.stop()

    # --- Generate metadata using OpenAI ---
    prompt = f"""
You are a metadata extraction assistant.
Your task is to extract and return a JSON object with exactly the following fields:

{{
  "sermon_id": "{sermon_id}",
  "title": "{title}",
  "preacher": "Rev. Ray Choi",
  "date": "{date_str}",
  "series": null,
  "passages": [],
  "tags": [],
  "summary": "",
  "transcript_file": "{public_url}",
  "url": "{url}"
}}

Instructions:
    • Preacher is always "Rev. Ray Choi" (no need to infer).
    • Derive sermon_id and transcript_file using the date and a slugified version of the title.
    • If series is not mentioned, set it to null.
    • Extract all cited Scripture passages into the passages array.
    • Tags should be 3–5 concise themes from the sermon.
    • Summary should faithfully capture the heart of the message in 1–2 sentences.
    • Output only the JSON object—no extra commentary or text.

Sermon transcript:
{transcript.strip()}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        metadata_raw = response.choices[0].message.content.strip()
        metadata = json.loads(metadata_raw)

        # --- normalize schema (force arrays, ensure required keys) ---
        def as_list(x):
            if x is None or x == {} or x == "":
                return []
            return x if isinstance(x, list) else [x]

        # Canonicalize key fields we absolutely control
        metadata["sermon_id"] = sermon_id
        metadata["title"] = title
        metadata["date"] = date_str
        metadata["url"] = url
        metadata["preacher"] = "Rev. Ray Choi"
        # canonical transcript location in Storage
        metadata["transcript_file"] = public_url

        # Arrays (avoid {} vs [] drift)
        metadata["passages"] = as_list(metadata.get("passages"))
        metadata["tags"] = as_list(metadata.get("tags"))

        # Insert metadata into Supabase table (via your RPC)
        supabase.rpc("insert_sermon_metadata", metadata).execute()
        st.success("✅ Metadata inserted into Supabase table")

        # --- Embed and upsert into Pinecone ---
        chunks = chunk_text(transcript)
        vectors = []

        for idx, chunk in enumerate(chunks):
            res = client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            emb = res.data[0].embedding
            vectors.append((
                f"{sermon_id}_chunk{idx}",
                emb,
                {
                    "sermon_id":  sermon_id,
                    "title":      title,
                    "date":       date_str,               # helpful for debugging/filters
                    "passages":   ", ".join(metadata["passages"]),
                    "chunk_text": chunk
                }
            ))

        # Batch upserts (100 at a time)
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i: i+100])

        st.success("✅ Embedded and upserted to Pinecone")

        # Optional: show a quick summary to the uploader
        st.info(f"Uploaded **{sermon_id}** with {len(chunks)} chunks.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
