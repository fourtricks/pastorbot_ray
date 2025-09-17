import os
from datetime import datetime
from flask import Flask, request, render_template_string
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from supabase import create_client, Client

# =========================================================
# ENV & CLIENTS
# =========================================================
load_dotenv()

# --- Services ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "pastor-ray-sermons")
index = pc.Index(INDEX_NAME)

print(f"[init] Pinecone Index: {INDEX_NAME}")
print(f"[init] Supabase URL: {SUPABASE_URL}")


# =========================================================
# ONE PLACE TO TUNE LENIENCY / GROUNDEDNESS
# (Override via env or tweak here while debugging)
# =========================================================
TUNING = {
    # Retrieval breadth
    "TOP_K":                    int(os.getenv("TOP_K", "8")),            # ★ how many chunks to fetch
    "MAX_CONTEXT_CHUNKS":       int(os.getenv("MAX_CONTEXT_CHUNKS", "4")),  # ★ how many chunks to keep after score sort

    # Score floors (groundedness)
    "SCORE_THRESHOLD":          float(os.getenv("SCORE_THRESHOLD", "0.80")),  # ★ typical accept cutoff
    "RETRIEVAL_FLOOR":          float(os.getenv("RETRIEVAL_FLOOR", "0.70")),  # ★ softer floor to allow slight expansion

    # Context adequacy guard (prevents thin/low-signal answers)
    "MIN_CONTEXT_CHARS":        int(os.getenv("MIN_CONTEXT_CHARS", "250")),   # ★ if combined context is too short, refuse

    # Generation controls (drift vs. strictness)
    "MODEL_ID":                 os.getenv("FT_MODEL_ID", "ft:gpt-4.1-mini-2025-04-14:easycloud::CA2f3fbc"),  # ★ main chat model
    "TEMPERATURE":              float(os.getenv("TEMPERATURE", "0.2")),  # ★ lower = stricter; higher = more flexible
    "MAX_TOKENS":               int(os.getenv("MAX_TOKENS", "450")),     # ★ longer answers vs. concise

    # Embeddings model
    "EMBEDDING_MODEL":          os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),  # ★ swap to text-embedding-3-large if desired
}

# Quick aliases to reduce clutter below
TOP_K = TUNING["TOP_K"]
MAX_CONTEXT_CHUNKS = TUNING["MAX_CONTEXT_CHUNKS"]
SCORE_THRESHOLD = TUNING["SCORE_THRESHOLD"]
RETRIEVAL_FLOOR = TUNING["RETRIEVAL_FLOOR"]
MIN_CONTEXT_CHARS = TUNING["MIN_CONTEXT_CHARS"]
MODEL_ID = TUNING["MODEL_ID"]
TEMPERATURE = TUNING["TEMPERATURE"]
MAX_TOKENS = TUNING["MAX_TOKENS"]
EMBEDDING_MODEL = TUNING["EMBEDDING_MODEL"]


# =========================================================
# METADATA HELPERS
# =========================================================

def load_all_meta():
    resp = supabase.table("sermons_metadata").select("*").execute()
    data = resp.data or []
    return {s["sermon_id"]: s for s in data}


sermon_meta = load_all_meta()
print(f"[init] Loaded {len(sermon_meta)} sermon metadata rows.")


def fetch_meta_by_id(sid: str):
    """Safe metadata fetch: check cache first, then Supabase (and cache it)."""
    meta = sermon_meta.get(sid)
    if meta:
        return meta
    try:
        res = supabase.table("sermons_metadata").select("*").eq("sermon_id", sid).limit(1).execute()
        rows = res.data or []
        if rows:
            sermon_meta[sid] = rows[0]
            return rows[0]
    except Exception as e:
        print(f"[warn] Supabase fetch failed for {sid}: {e}")
    print(f"[warn] Missing metadata for {sid}")
    return None


# =========================================================
# CORE RAG (NO PARAPHRASING)
# =========================================================

REFUSAL_SENTENCE = (
    "I have not covered that in the sermons yet, but thank you for bringing that question to my attention."
)

SYSTEM_PROMPT = (
    "You are a bot known as 'PastorBot Ray' emulating Rev. Ray Choi—a warm, compassionate pastor.\n"
    "GROUNDING (CLOSED-BOOK):\n"
    "• Use ONLY the provided Context for facts.\n"
    "• Do NOT add definitions, history, examples, names (e.g., philosophers, theologians), dates, or Scripture unless they appear verbatim in Context.\n"
    "• Every sentence in your answer must be anchored to Context via a short quoted phrase from Context (put it in double quotes). Do not invent quotes.\n"
    "• If the Context lacks the specific concept or definition requested, you must refuse.\n"
    "REFUSAL:\n"
    f"• If the Context does not EXPLICITLY explain the concept, reply EXACTLY:\n  {REFUSAL_SENTENCE}\n"
    "STYLE:\n"
    "• Be concise. 1–2 sentence direct answer, then 1 short paragraph tying to phrases from Context.\n"
    "• Warm pastoral tone. Capitalize divine names/pronouns; lowercase for human references."
)


def ask_pastor_ray(question: str, top_k: int = TOP_K):
    """
    Retrieval on the raw question only (no paraphrasing).
    All key leniency knobs live in TUNING (see top of file).
    Returns (answer, citations_list).
    """
    # --- Embed question ---
    e_res = oai.embeddings.create(model=EMBEDDING_MODEL, input=question)
    q_vec = e_res.data[0].embedding

    # --- Retrieve ---
    res = index.query(
        vector=q_vec,
        top_k=max(top_k, 5),  # ensure minimal breadth
        include_metadata=True,
        include_scores=True
    )

    matches = list(res.matches or [])
    # Sort, then keep those above softer floor; then trim to MAX_CONTEXT_CHUNKS
    matches.sort(key=lambda m: (m.score or 0.0), reverse=True)
    kept = [m for m in matches if (m.score or 0.0) >= RETRIEVAL_FLOOR][:MAX_CONTEXT_CHUNKS]

    if not kept:
        return REFUSAL_SENTENCE, []

    # --- Build context (verbatim bullets) ---
    context = "\n\n".join(f"• {m.metadata.get('chunk_text','')}" for m in kept)

    # --- Thin-context guard ---
    if len(context) < MIN_CONTEXT_CHARS:
        return REFUSAL_SENTENCE, []

    # --- Generate grounded answer ---
    user_msg = (
        f"Context (use ONLY this material for facts and NOTHING else):\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "If the Context is insufficient, use the REFUSAL sentence above. Otherwise, follow the STYLE and GROUNDING rules exactly as they are outlined."
    )

    chat = oai.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=60
    )
    answer = (chat.choices[0].message.content or "").strip()

    # --- Citations from kept chunks ---
    citations = []
    for m in kept:
        sid = m.metadata.get("sermon_id")
        info = fetch_meta_by_id(sid) if sid else None
        if not info:
            continue
        link_html = (
            f'<a href="{info.get("url", "#")}" target="_blank" rel="noopener noreferrer">'
            f'{info.get("title", sid)}</a> ({info.get("date", "")})'
        )
        chunk = (m.metadata.get("chunk_text") or "").strip()
        citations.append({"link": link_html, "chunk": chunk})

    return answer, citations


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


@app.before_request
def maybe_refresh_meta():
    # Auto-refresh metadata in debug so you see updates live.
    global sermon_meta
    if app.debug:
        sermon_meta = load_all_meta()


TEMPLATE = """
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
    .icon { width: 48px; height: 48px; stroke-width: 3; fill: none; }
    .tuning { font-size: 0.9rem; color: #555; background:#eef2f7; padding:0.5rem; border-left:3px solid #4a90e2; }

    /* Collapsed "Context" section — subtle & out of the way */
    .context-collapsible { margin-top: 0.75rem; }
    .context-collapsible > summary {
      cursor: pointer;
      list-style: none;            /* hide default marker in some browsers */
      color: #6b7280;              /* muted */
      font-size: 0.85rem;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      user-select: none;
    }
    .context-collapsible > summary::-webkit-details-marker { display: none; }
    .context-collapsible > summary:hover .ctx-summary-label { text-decoration: underline; }
    .context-collapsible[open] > summary { color: #374151; } /* darker when open */
    .ctx-summary-label {
      font-weight: 600;
      text-decoration: underline;
      text-underline-offset: 2px;     /* nicer spacing */
      text-decoration-thickness: 1.5px; /* optional */
    }
    .ctx-summary-hint  { font-weight: 400; opacity: 0.7; }

    .ctx-list { margin: 0.5rem 0 0 0; padding-left: 1.25rem; }
    .ctx-item { margin: 0.5rem 0; }
    .ctx-link { font-size: 0.9rem; margin-bottom: 0.25rem; }
    .ctx-chunk {
      white-space: pre-line;
      margin: 0.25rem 0 0.75rem 0;
      padding: 0.5rem 0.75rem;
      background: #f9fafb;
      border-left: 3px solid #e5e7eb;
      border-radius: 0.25rem;
      color: #374151;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <div id="overlay"><div class="spinner"></div></div>
  <h1>🙏 Ask PastorBot Ray</h1>

  <div class="tuning">
    <strong>Tuning (env or edit top of file):</strong>
    TOP_K={{top_k}}, MAX_CONTEXT_CHUNKS={{max_chunks}}, SCORE_THRESHOLD={{score_thr}},
    RETRIEVAL_FLOOR={{retrieval_floor}}, MIN_CONTEXT_CHARS={{min_ctx}},
    TEMPERATURE={{temperature}}, MAX_TOKENS={{max_tokens}}, MODEL_ID={{model_id}}, EMBEDDING_MODEL={{embedding_model}}
  </div>

  <form id="qa-form" method="post">
    <textarea name="question" rows="4" placeholder="Enter your question here..."></textarea><br>
    <button type="submit">Ask</button>
  </form>

  {% if answer %}
    <div class="response">
      <h2>Your Question:</h2>
      <p style="font-style: italic; background: #eef2f7; padding: 0.5rem; border-left: 3px solid #4a90e2;">
        {{ request.form.get('question') }}
      </p>
      <h2>PastorBot Ray says:</h2>
      {% for para in answer.split('\\n\\n') %}
        <p>{{ para }}</p>
      {% endfor %}

      {% if citations %}
        <details class="context-collapsible" aria-label="Context supporting this answer">
          <summary>
            <span class="ctx-summary-label">Context</span>
            <span class="ctx-summary-hint">(sermon excerpts used)</span>
          </summary>
          <ul class="ctx-list">
            {% for item in citations %}
              <li class="ctx-item">
                <div class="ctx-link">{{ item.link|safe }}</div>
                <div class="ctx-chunk">{{ item.chunk }}</div>
              </li>
            {% endfor %}
          </ul>
        </details>
      {% endif %}
    </div>

    <h2>Rate this answer:</h2>
    <form method="post">
      <input type="hidden" name="question" value="{{ request.form.question }}">
      <input type="hidden" name="answer" value="{{ answer }}">
      <div class="btn-group">
        <button type="submit" name="rating" value="1" title="Poor">
          <svg class="icon" viewBox="0 0 24 24" stroke="red">
            <circle cx="12" cy="12" r="10" />
            <path d="M8 16 C10 14,14 14,16 16" />
            <line x1="9" y1="9" x2="9.01" y2="9" />
            <line x1="15" y1="9" x2="15.01" y2="9" />
          </svg>
        </button>
        <button type="submit" name="rating" value="2" title="Okay">
          <svg class="icon" viewBox="0 0 24 24" stroke="orange">
            <circle cx="12" cy="12" r="10" />
            <line x1="8" y1="16" x2="16" y2="16" />
            <line x1="9" y1="9" x2="9.01" y2="9" />
            <line x1="15" y1="9" x2="15.01" y2="9" />
          </svg>
        </button>
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

        # Save rating feedback if provided
        if rating and question and answer_text:
            fb = {
                'timestamp': datetime.utcnow().isoformat(),
                'question': question,
                'response': answer_text,
                'rating': int(rating)
            }
            # Use a Postgres function (pre-created) to insert feedback atomically
            try:
                supabase.rpc("insert_feedback", {
                    "ts": fb["timestamp"],
                    "q": fb["question"],
                    "r": fb["response"],
                    "rate": fb["rating"]
                }).execute()
            except Exception as e:
                print(f"[warn] feedback save failed: {e}")
        elif question:
            answer, citations = ask_pastor_ray(question)

    return render_template_string(
        TEMPLATE,
        answer=answer,
        citations=citations,
        top_k=TOP_K,
        max_chunks=MAX_CONTEXT_CHUNKS,
        score_thr=SCORE_THRESHOLD,
        retrieval_floor=RETRIEVAL_FLOOR,
        min_ctx=MIN_CONTEXT_CHARS,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        model_id=MODEL_ID,
        embedding_model=EMBEDDING_MODEL
    )


if __name__ == '__main__':
    # Tip: set FLASK_DEBUG=1 to auto-refresh metadata per request
    app.run(host='0.0.0.0', port=8501, debug=True)
