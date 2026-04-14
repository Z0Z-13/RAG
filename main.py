import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# =========================
# Cache
# =========================
os.environ["HF_HOME"] = "E:\\hf_cache"


# =========================
# 1. Select file
# =========================
Tk().withdraw()
file_path = askopenfilename()

print("Selected file:", file_path)

if not file_path:
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
    print("Unsupported file")
    exit()

docs = loader.load()


# =========================
# 3. Chunking
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)


# =========================
# 4. Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# 5. Vector DB
# =========================
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="E:\\rag_project\\chroma_db"
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# =========================
# 6. LLM (Pipeline)
# =========================
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.float32
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)


# =========================
# 7. Prompt
# =========================
template = """<|system|>
Answer the question based ONLY on the context.
If not found, say "I don't know".

Context:
{context}</s>
<|user|>
{input}</s>
<|assistant|>"""

prompt = PromptTemplate.from_template(template)


# =========================
# 8. RAG Chain
# =========================
rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# =========================
# 9. Chat
# =========================
print("\nChat ready (type 'exit' to quit)\n")

while True:
    q = input("You: ")

    if q.lower() == "exit":
        print("Bye 👋")
        break

    response = rag_chain.invoke(q)

    print("\nBot:\n")
    print(response)
    print("-" * 50)