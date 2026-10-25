from config import SIMILARITY_THRESHOLD
from chromadb.utils import embedding_functions
import chromadb
import hashlib
import re
from sentence_transformers import CrossEncoder
client = chromadb.PersistentClient("./cache_data/")
collection = client.get_or_create_collection(
        name="qa_cache",
        metadata={"hnsw:space": "cosine"},
        embedding_function= embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-base-en-v1.5"
        )
)

_ce = CrossEncoder("cross-encoder/quora-distilroberta-base")
CE_CUTOFF = 0.25


def add_qa(question,answer):
    collection.upsert(
        ids=[make_id(question)],
        documents=[question],
        metadatas=[{"answer" : answer}]
    )

def make_id(question):
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()

def lookup(question):
    if collection.count() == 0:
        return None, 0.0
    result = collection.query(query_texts=[question],n_results=1)
    best_sim = 1 - result["distances"][0][0]
    for doc,meta,dist in zip(result["documents"][0],result["metadatas"][0],result["distances"][0]):
        sim = 1 -dist
        if sim < SIMILARITY_THRESHOLD:
            break
        if doc.strip().lower() == question.strip().lower() or verify(doc,question):
            return meta["answer"],sim
    return None,best_sim

def clear_cache():
    all_ids = collection.get(include=[])["ids"]
    if all_ids:
        collection.delete(ids=all_ids)

def extract_tokens(q):
    return set(re.findall(r"\b[a-z]*\d+(?:\.\d+)?[a-z]*\b", q.lower()))

def numbers_guard(q1, q2):
    return extract_tokens(q1) == extract_tokens(q2)

def cross_encoder_score(q1, q2):
    return float(_ce.predict([(q1, q2)])[0])

def verify(q1, q2):
    return numbers_guard(q1, q2) and cross_encoder_score(q1, q2) >= CE_CUTOFF


    

if __name__ == "__main__":
    clear_cache()