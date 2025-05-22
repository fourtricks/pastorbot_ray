import os
import json
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# 1) Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2) Initialize Pinecone & select your index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pastor-ray-sermons")

# 3) Load metadata
with open("sermons_metadata.json", encoding="utf8") as f:
    sermons = json.load(f)


# 4) Helper to chunk text into ~1500-char pieces
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


# 5) Embed & upsert
for sermon in tqdm(sermons, desc="Embedding sermons"):
    # load transcript
    path = os.path.join("sermons", sermon["transcript_file"])
    with open(path, encoding="utf8") as f:
        text = f.read()

    chunks = chunk_text(text)
    vectors = []

    for idx, chunk in enumerate(chunks):
        # create a 1536-dim embedding
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

    # batch upsert in blocks of 100
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i: i+100])

print("✅ All sermons embedded with OpenAI and uploaded to Pinecone.")
