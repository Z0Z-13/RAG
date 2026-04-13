import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# =========================
# Cache (use E drive)
# =========================
os.environ["HF_HOME"] = "E:\\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:\\hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "E:\\hf_cache"


# =========================
# 1. Select file
# =========================
Tk().withdraw()
file_path = askopenfilename()

print("Selected file:", file_path)

if not file_path:
    print("No file selected")
    exit()


# =========================
# 2. Load document
# =========================
ext = os.path.splitext(file_path)[1].lower()

if ext == ".pdf":
    loader = PyPDFLoader(file_path)
elif ext == ".docx":
    loader = Docx2txtLoader(file_path)
else:
    print("Unsupported file type")
    exit()

docs = loader.load()
print(f"Documents loaded: {len(docs)}")


# =========================
# 3. Chunking
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)
print(f"Chunks created: {len(chunks)}")


# =========================
# 4. Embeddings
# =========================
print("Loading embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# 5. Vector DB
# =========================
print("Building vector DB...")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="E:\\rag_project\\chroma_db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# =========================
# 6. Load LLM (TinyLlama)
# =========================
print("Loading TinyLlama...")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.float32
)


# =========================
# 7. Build context
# =========================
def build_context(question):
    docs = retriever.invoke(question)
    return "\n\n".join([d.page_content for d in docs])


# =========================
# 8. Generate answer
# =========================
def generate_answer(prompt):
    prompt = str(prompt)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            repetition_penalty=1.1
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# 9. Ask function (RAG)
# =========================
def ask(question):
    context = build_context(question)

    prompt = f"""
You are a smart assistant.

Context:
{context}

Question:
{question}

Answer:
"""

    return generate_answer(prompt)


# =========================
# 10. Chat loop
# =========================
print("\nChat ready (type 'exit' to quit)\n")

while True:
    q = input("You: ")

    if q.lower() == "exit":
        print("Bye 👋")
        break

    answer = ask(q)

    print("\nBot:\n")
    print(answer)
    print("-" * 50)