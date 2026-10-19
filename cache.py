from config import SIMILARITY_THRESHOLD
from rag import ask
import chromadb
client = chromadb.PersistentClient(path="./cache_data/")
collection = client.get_or_create_collection(
        name="qa_cache",
        metadata={"hnsw:space": "cosine"},   
)


def add_qa(question,answer):
    collection.upsert(
        ids=[f"Q{len(question)}"],
        documents=[question],
        metadatas=[{"answer" : answer}]
    )


def lookup(question):
    answer = None
    result = collection.query(query_texts=[question],n_results=1)
    print(result)
    similarity = 1 - result["distances"][0][0] 
    print("similarity : " + str(similarity))
    if (len(result["distances"][0]) != 0) and (similarity >= SIMILARITY_THRESHOLD) :
        answer = result["metadatas"][0]["answer"]

    return answer

def clear_cache():
    all_ids = collection.get(include=[])["ids"]
    if all_ids:
        collection.delete(ids=all_ids)

if __name__ == "__main__":
    question = "How many paid vacation days does a new employee get per year?"
    answer = lookup(question)
    if answer is None : 
        print("Not cached")
        response = ask(question)
        answer = response.choices[0].message.content
        add_qa(question,answer)
    print(answer)
