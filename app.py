import os
import json
from datetime import datetime
from flask import Flask, request, render_template_string
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from supabase import create_client, Client


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pc.Index("pastor-ray-sermons")

# Path for feedback storage
FEEDBACK_FILE = "feedback.jsonl"

# Load sermons metadata into a dict for quick lookup
with open("sermons_metadata.json", encoding="utf8") as f:
    sermon_meta = {s["sermon_id"]: s for s in json.load(f)}


# Helper: ask Pastor Ray using RAG + fine-tuned model
def ask_pastor_ray(question: str, top_k: int = 3) -> str:
    # Embed query
    e_res = oai.embeddings.create(
        model="text-embedding-ada-002",
        input=question
    )
    q_vec = e_res.data[0].embedding

    # Retrieve context
    res = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True
    )

    # Build the raw context for the model
    context = "\n\n".join(
        f"• {m.metadata['chunk_text']}" for m in res.matches
    )

    # Collect which sermons we pulled (de-duplicate by sermon_id)
    sermon_infos = {}
    for m in res.matches:
        sid = m.metadata["sermon_id"]
        if sid not in sermon_infos:
            sermon_infos[sid] = sermon_meta[sid]   # full metadata dict

    # Build messages
    system = {
        "role": "system",
        "content": (
            "You are Rev. Ray Choi—a warm, compassionate pastor. "
            "Answer the question below in his style, grounding every answer in Scripture. "
            "Always capitalize divine names/pronouns when referring to God, and use lowercase for human references."
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer as Pastor Ray Choi:"
        )
    }

    # Generate response
    chat = oai.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:easycloud::BStTh9Rf",
        messages=[system, user_msg],
        temperature=0.7,
        max_tokens=512,
        timeout=60
    )
    answer = chat.choices[0].message.content.strip()

    # Build a list of {link, chunk} for each of the top_k matches
    citations = []
    for m in res.matches:
        sid = m.metadata["sermon_id"]
        info = sermon_meta[sid]
        link_html = (
            f'<a href="{info["url"]}" target="_blank" rel="noopener noreferrer">'
            f'{info["title"]}</a> ({info["date"]})'
        )
        chunk = m.metadata["chunk_text"].strip()
        citations.append({"link": link_html, "chunk": chunk})

    return answer, citations


# Flask app setup
app = Flask(__name__)

template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pastor Ray Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f4f4f9; }
    h1 { color: #333; }
    form { margin-bottom: 1rem; }
    textarea { width: 100%; padding: 0.5rem; font-size: 1rem; border-radius: 0.25rem; border: 1px solid #ccc; }
    .btn-group { display: flex; gap: 1rem; margin-top: 0.5rem; }
    .btn-group button { background: none; border: none; cursor: pointer; padding: 0; }
    .response { margin-top: 1.5rem; padding: 1rem; background: white; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    #overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.8); display: none; align-items: center; justify-content: center; z-index: 1000; }
    .spinner { border: 8px solid #f3f3f3; border-top: 8px solid #4a90e2; border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    /* Rating icons: larger size simple line art */
    .icon { width: 48px; height: 48px; stroke-width: 3; fill: none; }
  </style>
</head>
<body>
  <div id="overlay"><div class="spinner"></div></div>
  <h1>🙏 Ask PastorBot Ray</h1>
  <form id="qa-form" method="post">
    <textarea name="question" rows="4" placeholder="Enter your question here..."></textarea><br>
    <button type="submit">Ask</button>
  </form>
  {% if answer %}
    <div class="response">
      <h2>PastorBot Ray says:</h2>
      {% for para in answer.split('\n\n') %}
        <p>{{ para }}</p>
      {% endfor %}

        {% if citations %}
        <h3><strong>CITATIONS:</strong></h3>
        <ul>
            {% for item in citations %}
            <li>
                {{ item.link|safe }}<br>
                <div style="white-space: pre-line; margin:0.5em 0; padding:0.5em; background:#f9f9f9; border-left:3px solid #ddd;">
                  {{ item.chunk }}
                </div>
            </li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>

      <h2>Rate this answer:</h2>
      <form method="post">
        <input type="hidden" name="question" value="{{ request.form.question }}">
        <input type="hidden" name="answer" value="{{ answer }}">
        <div class="btn-group">
          <!-- Poor: red frown -->
          <button type="submit" name="rating" value="1" title="Poor">
            <svg class="icon" viewBox="0 0 24 24" stroke="red">
              <circle cx="12" cy="12" r="10" />
              <path d="M8 16 C10 14,14 14,16 16" />
              <line x1="9" y1="9" x2="9.01" y2="9" />
              <line x1="15" y1="9" x2="15.01" y2="9" />
            </svg>
          </button>
          <!-- Okay: orange neutral line -->
          <button type="submit" name="rating" value="2" title="Okay">
            <svg class="icon" viewBox="0 0 24 24" stroke="orange">
              <circle cx="12" cy="12" r="10" />
              <line x1="8" y1="16" x2="16" y2="16" />
              <line x1="9" y1="9" x2="9.01" y2="9" />
              <line x1="15" y1="9" x2="15.01" y2="9" />
            </svg>
          </button>
          <!-- Good: green smile -->
          <button type="submit" name="rating" value="3" title="Good">
            <svg class="icon" viewBox="0 0 24 24" stroke="green">
              <circle cx="12" cy="12" r="10" />
              <path d="M8 16 C10 18,14 18,16 16" />
              <line x1="9" y1="9" x2="9.01" y2="9" />
              <line x1="15" y1="9" x2="15.01" y2="9" />
            </svg>
          </button>
        </div>
      </form>
    </div>
  {% endif %}
  <script>
    document.getElementById('qa-form').addEventListener('submit', () => {
      document.getElementById('overlay').style.display = 'flex';
      document.querySelector('button[type=submit]').disabled = true;
    });
  </script>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def home():
    answer = None
    citations = None
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        answer_text = request.form.get('answer', '').strip()
        rating = request.form.get('rating')
        # If rating provided, save feedback
        if rating and question and answer_text:
            feedback = {
                'timestamp': datetime.utcnow().isoformat(),
                'question': question,
                'response': answer_text,
                'rating': int(rating)
            }
            supabase.table("feedback").insert(feedback).execute()
            # Reset answer to avoid duplicate saves
            answer = None
        elif question:
            answer, citations = ask_pastor_ray(question)
    return render_template_string(template, answer=answer, citations=citations)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=True)
