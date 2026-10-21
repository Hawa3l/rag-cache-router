import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]

DATA_FILE = Path(__file__).parent / "cache_test_questions.json"
with open(DATA_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

originals   = [q["original"]   for q in questions]
paraphrases = [q["paraphrase"] for q in questions]
near_misses = [q["near_miss"]  for q in questions]
N = len(questions)


def evaluate(model_name):
    model = SentenceTransformer(model_name)      

    start = time.perf_counter()
    orig_emb = model.encode(originals,   normalize_embeddings=True)
    para_emb = model.encode(paraphrases, normalize_embeddings=True)
    near_emb = model.encode(near_misses, normalize_embeddings=True)
    elapsed = time.perf_counter() - start

    para_sims = np.diag(orig_emb @ para_emb.T)
    near_sims = np.diag(orig_emb @ near_emb.T)


    orig_sims = orig_emb @ orig_emb.T
    np.fill_diagonal(orig_sims, -1)              

    return para_sims, near_sims, orig_sims, elapsed


if __name__ == "__main__":
    results = []
    thresholds = np.arange(0.40, 1.00, 0.01)

    for m in models:
        print(f"Evaluating {m} ...")
        para_sims, near_sims, orig_sims, elapsed = evaluate(m)

        gap = para_sims.min() - near_sims.max()
        near_beats_para = int(np.sum(near_sims > para_sims))
        max_unrelated = orig_sims.max()
        ms_per_sentence = elapsed * 1000 / (3 * N)

    
        caught, best_t = 0, None
        for t in thresholds:
            if np.sum(near_sims >= t) == 0 and max_unrelated < t:
                caught = int(np.sum(para_sims >= t))
                best_t = round(float(t), 2)
                break

        results.append({
            "model": m.split("/")[-1],
            "gap": gap,
            "nb": near_beats_para,
            "max_unrel": max_unrelated,
            "thr": best_t,
            "caught": caught,
            "ms": ms_per_sentence,
            "para": para_sims,
            "near": near_sims,
        })

    print(f"{'model':30}{'gap':>8}{'near>para':>11}{'max_unrel':>11}"
          f"{'thr':>7}{'caught':>9}{'ms/sent':>9}")
    for r in results:
        thr = "none" if r["thr"] is None else f"{r['thr']:.2f}"
        print(f"{r['model']:30}{r['gap']:8.2f}{r['nb']:>8}/{N}{r['max_unrel']:11.2f}"
              f"{thr:>7}{r['caught']:>6}/{N}{r['ms']:9.1f}")
        
    best = max(results, key=lambda r: (r["caught"], r["gap"]))
    print(f"\nWorst cases for {best['model']}:")
    print("Lowest-scoring paraphrases (should be high):")
    for i in np.argsort(best["para"])[:3]:
        print(f"  {best['para'][i]:.2f}  {originals[i]}  |  {paraphrases[i]}")
    print("Highest-scoring near-misses (should be low):")
    for i in np.argsort(best["near"])[::-1][:3]:
        print(f"  {best['near'][i]:.2f}  {originals[i]}  |  {near_misses[i]}")