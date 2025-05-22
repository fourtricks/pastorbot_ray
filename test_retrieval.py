import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# 1) Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pastor-ray-sermons")

# 2) Sample question
query = "How do I find hope in hard seasons?"

# 3) Embed the question (1536-dim)
res = client.embeddings.create(
    model="text-embedding-ada-002",
    input=query
)
query_embed = res.data[0].embedding

# 4) Retrieve top 3 chunks
results = index.query(
    vector=query_embed,
    top_k=3,
    include_metadata=True
)

# 5) Display
print(f"\n🔍 Question: {query}\n")
for match in results.matches:
    meta = match.metadata
    print(f"— Sermon: {meta['title']}  (score: {match.score:.4f})")
    print(f"  Excerpt: {meta['chunk_text'][:200].strip()}…\n")
