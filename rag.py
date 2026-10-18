from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()
client = chromadb.PersistentClient(path="./data/")

client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"},   
)

def load_files(path):
    pdf_files = Path(path).glob("*.pdf")
    documents = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf.name}: {e}")
            continue
    return documents

def chunk_text(path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(load_files(path))
    return chunks

def load_chunks(chunks):
    collection.upsert(
        ids=[f"{c.metadata.get('source', 'unknown')}_p{c.metadata.get('page', 0)}_{i}" for i, c in enumerate(chunks)],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )

def retrieve(question,k=3):
    results = collection.query(query_texts=[question],n_results=k)
    return results["documents"][0]


def generate(question,chunks,model_name="nvidia/nemotron-3-super-120b-a12b:free"):
    context ="\n\n---\n\n".join(chunks)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer isn't there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client_llm.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response


def ask(question):
    chunk = retrieve(question)
    return generate(question,chunk)


if __name__ == "__main__":
    load_chunks(chunk_text("./data/test-data/"))
