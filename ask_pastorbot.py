import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# — 1) Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pastor-ray-sermons")


def generate_answer(question: str, top_k: int = 3) -> str:
    print("⏳ Embedding question...")
    embed_res = client.embeddings.create(
        model="text-embedding-ada-002",
        input=question
    )
    q_vec = embed_res.data[0].embedding

    print("🔍 Retrieving relevant sermon chunks...")
    res = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True
    )

    context = "\n\n".join(f"• {m.metadata['chunk_text']}" for m in res.matches)
    print(f"📚 Retrieved {len(res.matches)} chunks. Building prompt...")

    system_message = {
        "role": "system",
        "content": (
            "You are Rev. Ray Choi—a warm, compassionate pastor. "
            "Answer the question below in his style, grounding every answer in Scripture.\n\n"
            "IMPORTANT: Whenever you use a name or pronoun for the Divine (God, Father, Son, Holy Spirit, He, Him, His, Me), capitalize it. "
            "When referring to humans, use lowercase (he, him, his, father, son)."
            "Apply this also to quoted passeges within the response."
        )
    }
    user_message = {
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer as Pastor Ray Choi:"
        )
    }

    print("💬 Sending to fine‑tuned model...")
    chat_res = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:easycloud::BStTh9Rf",
        messages=[system_message, user_message],
        temperature=0.7,
        max_tokens=512,       # cap the response size
        timeout=60            # fail if it takes over 60s
    )
    print("✅ Received response!")
    return chat_res.choices[0].message.content.strip()


if __name__ == "__main__":
    q = input("\nWhat’s your question for Pastor Ray? ")
    print("\nPastorBot says:\n")
    print(generate_answer(q))
