import os
import shutil
import time

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from google import genai

# =========================
# APP
# =========================
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# GEMINI CLIENT
# =========================
client = genai.Client(api_key="AIzaSyBki14yGQOcknSzSgoA5vL9VgY-ypSVeeI")

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

def ask_gemini(prompt: str):
    for model in MODELS:
        for _ in range(2):
            try:
                res = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                if res and res.text:
                    return res.text.strip()
            except:
                time.sleep(1)
                continue
    return "❌ Model failed"

# =========================
# EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# PROMPTS (FIXED)
# =========================
full_prompt = """
You are a professional assistant.

Use the context below to answer in a detailed, structured, and explanatory way.

- Expand the answer as much as possible
- Use bullet points if needed
- Explain clearly and do not be overly short
- Use all relevant information in the context

Context:
{context}

Question:
{input}

Answer:
"""

light_prompt = """
Answer briefly and directly using only the context.

Context:
{context}

Question:
{input}

Answer:
"""

# =========================
# LOAD FILE
# =========================
def load_document(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(path)
    elif ext == ".docx":
        return Docx2txtLoader(path)
    elif ext == ".txt":
        return TextLoader(path)
    return None

# =========================
# FORMAT CONTEXT (IMPROVED)
# =========================
def format_docs(docs):
    if not docs:
        return "No relevant context found."

    return "\n\n".join(
        f"[Chunk {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )

# =========================
# UI
# =========================
@app.get("/rag-page", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <body style="background:#0f172a;color:white;text-align:center;font-family:Arial">

    <h2>📄 Full vs Light RAG (Improved)</h2>

    <input type="file" id="file"><br><br>
    <input id="q" placeholder="Ask..." style="padding:10px;width:300px"><br><br>

    <button onclick="run()">Run</button>

    <h3>🧠 Full RAG</h3>
    <pre id="full"></pre>

    <h3>⚡ Light RAG</h3>
    <pre id="light"></pre>

    <h3>⏱ Time</h3>
    <pre id="time"></pre>

    <script>
    async function run(){
        let file = document.getElementById("file").files[0];
        let q = document.getElementById("q").value;

        let form = new FormData();
        form.append("file", file);
        form.append("question", q);

        let res = await fetch("/rag", {
            method: "POST",
            body: form
        });

        let data = await res.json();

        document.getElementById("full").innerText = data.full_rag;
        document.getElementById("light").innerText = data.light_rag;
        document.getElementById("time").innerText = data.time;
    }
    </script>

    </body>
    </html>
    """

# =========================
# MAIN RAG ENDPOINT
# =========================
@app.post("/rag")
async def rag(file: UploadFile = File(...), question: str = Form(...)):
    try:
        path = os.path.join(UPLOAD_DIR, file.filename)

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = load_document(path)
        if not loader:
            return JSONResponse({"error": "Unsupported file type"})

        docs = loader.load()

        # =========================
        # SPLIT
        # =========================
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )

        chunks = splitter.split_documents(docs)

        # =========================
        # VECTOR DB
        # =========================
        db = Chroma.from_documents(chunks, embeddings)

        retriever_full = db.as_retriever(search_kwargs={"k": 10})
        retriever_light = db.as_retriever(search_kwargs={"k": 2})

        # =========================
        # FULL RAG
        # =========================
        t1 = time.time()

        docs_f = retriever_full.invoke(question)
        context_f = format_docs(docs_f)

        prompt_f = full_prompt.format(
            context=context_f,
            input=question
        )

        full_result = ask_gemini(prompt_f)
        t_full = time.time() - t1

        # =========================
        # LIGHT RAG
        # =========================
        t2 = time.time()

        docs_l = retriever_light.invoke(question)
        context_l = format_docs(docs_l)

        prompt_l = light_prompt.format(
            context=context_l,
            input=question
        )

        light_result = ask_gemini(prompt_l)
        t_light = time.time() - t2

        os.remove(path)

        return {
            "full_rag": full_result,
            "light_rag": light_result,
            "time": f"Full: {t_full:.2f}s | Light: {t_light:.2f}s"
        }

    except Exception as e:
        return {"error": str(e)}