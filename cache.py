from config import SIMILARITY_THRESHOLD,BASE_DIR
from rag import ask
import chromadb
import hashlib
client = chromadb.PersistentClient(path=BASE_DIR+"/cache_data/")
collection = client.get_or_create_collection(
        name="qa_cache",
        metadata={"hnsw:space": "cosine"},   
)

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
    similarity = 1 - result["distances"][0][0] 
    if similarity >= SIMILARITY_THRESHOLD :
        return result["metadatas"][0][0]["answer"], similarity

    return None,similarity

def clear_cache():
    all_ids = collection.get(include=[])["ids"]
    if all_ids:
        collection.delete(ids=all_ids)




if __name__ == "__main__":
    print("Distance metric:", collection.metadata)  # should show 'cosine'
    clear_cache()

    original = "How many paid vacation days does a new employee get per year?"
    tests = {
        "exact repeat": original,
        "paraphrase": "What's the PTO allowance for someone just starting at the company?",
        "near-miss (sick days)": "How many paid sick days do employees receive per calendar year?",
        "unrelated": "Who is the CFO of Brightwave?",
    }

    answer, sim = lookup(original)
    print(f"empty cache -> hit={answer is not None}, similarity={sim:.3f}")

    add_qa(original, "FAKE ANSWER: 20 days")   # fake answer, no API call

    for label, q in tests.items():
        answer, sim = lookup(q)
        print(f"{label:24} similarity={sim:.3f}  hit={answer is not None}")