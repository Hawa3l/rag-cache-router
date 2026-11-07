from config import SIMILARITY_THRESHOLD, CE_CUTOFF
from chromadb.utils import embedding_functions
import chromadb
import hashlib
import re
from sentence_transformers import CrossEncoder

client = chromadb.PersistentClient("./cache_data/")
collection = client.get_or_create_collection(
    name="qa_cache",
    metadata={"hnsw:space": "cosine"},
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5"
    ),
)

_ce = CrossEncoder("cross-encoder/quora-distilroberta-base")

# Word-to-number mapping for number normalization
WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
}
_NUM_WORD_PATTERN = re.compile(r"\b(" + "|".join(WORD_TO_NUM.keys()) + r")\b")


def add_qa(question, answer):
    collection.upsert(
        ids=[make_id(question)],
        documents=[question],
        metadatas=[{"answer": answer}],
    )


def make_id(question):
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()


def lookup(question, top_k=3):
    if collection.count() == 0:
        return None, 0.0

    k = min(top_k, collection.count())
    result = collection.query(query_texts=[question], n_results=k)

    docs = result["documents"][0] if result.get("documents") else []
    metas = result["metadatas"][0] if result.get("metadatas") else []
    dists = result["distances"][0] if result.get("distances") else []

    if not dists:
        return None, 0.0

    best_sim = max((1 - d) for d in dists)

    for doc, meta, dist in zip(docs, metas, dists):
        sim = 1 - dist
        if sim < SIMILARITY_THRESHOLD:
            continue
        if doc.strip().lower() == question.strip().lower() or verify(doc, question):
            return meta["answer"], sim

    return None, best_sim


def clear_cache():
    all_ids = collection.get(include=[])["ids"]
    if all_ids:
        collection.delete(ids=all_ids)


def extract_tokens(q):
    q_norm = re.sub(r"(?<=\d),(?=\d)", "", q.lower())
    q_norm = _NUM_WORD_PATTERN.sub(lambda m: WORD_TO_NUM[m.group(0)], q_norm)
    return set(re.findall(r"\b[a-z]*\d+(?:\.\d+)?[a-z]*\b", q_norm))


def numbers_guard(q1, q2):
    return extract_tokens(q1) == extract_tokens(q2)


def cross_encoder_score(q1, q2):
    return float(_ce.predict([(q1, q2)])[0])


def verify(q1, q2):
    return numbers_guard(q1, q2) and cross_encoder_score(q1, q2) >= CE_CUTOFF


if __name__ == "__main__":
    clear_cache()
