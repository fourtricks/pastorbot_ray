# recreate_index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# 1) Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 2) Delete old index if it exists
existing = [idx.name for idx in pc.list_indexes().indexes]
if "pastor-ray-sermons" in existing:
    pc.delete_index(name="pastor-ray-sermons")
    print("🗑️ Deleted old index")

# 3) Create new 1536-dim index on AWS us-east-1
pc.create_index(
    name="pastor-ray-sermons",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
print("✅ ‘pastor-ray-sermons’ recreated with dim=1536 on AWS us-east-1")
