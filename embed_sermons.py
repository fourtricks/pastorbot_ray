import os
import requests
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm
from supabase import create_client

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pastor-ray-sermons")

# Initialize Supabase (anon key fine for reads, service key for bulk updates if needed)
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# Helper: chunk text into ~1500-character pieces
def chunk_text(text, max_length=1500):
    paras = text.split("\n\n")
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) < max_length:
            current += p + "\n\n"
        else:
            chunks.append(current.strip())
            current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks


if __name__ == "__main__":
    # 1) Load all sermon metadata from Supabase
    resp = supabase.table("sermons_metadata").select("*").execute()
    sermons = resp.data
    if not sermons:
        print("❌ No sermons found in Supabase.")
        exit()

    # 2) Embed & upsert
    for sermon in tqdm(sermons, desc="Embedding sermons"):
        transcript_url = sermon["transcript_file"]

        # Download transcript text from Supabase Storage
        try:
            r = requests.get(transcript_url)
            r.raise_for_status()
            text = r.text
        except Exception as e:
            print(f"❌ Failed to download transcript for {sermon['sermon_id']}: {e}")
            continue

        chunks = chunk_text(text)
        vectors = []

        for idx, chunk in enumerate(chunks):
            # Create a 1536-dim embedding
            res = client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            emb = res.data[0].embedding

            vectors.append((
                f"{sermon['sermon_id']}_chunk{idx}",
                emb,
                {
                    "sermon_id":  sermon["sermon_id"],
                    "title":      sermon["title"],
                    "passages":   ", ".join(sermon["passages"]),
                    "chunk_text": chunk
                }
            ))

        # Batch upsert in blocks of 100
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i: i+100])

    print("✅ All sermons embedded with OpenAI and uploaded to Pinecone.")
