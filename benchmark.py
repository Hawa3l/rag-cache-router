from cache import lookup, clear_cache, add_qa
from pipeline import answer_cached, answer_baseline, answer_routed
from logger import init_db, clear_db
import sys
import json
import time

with open("./questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)


def warmup():
    add_qa("warmup", "warmup")
    lookup("warmup")
    init_db()




pipelines = {
    "--cached": answer_cached,
    "--baseline": answer_baseline,
    "--routed": answer_routed,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in pipelines:
        print("Indicate the pipeline: --baseline, --cached, or --routed")
        sys.exit(1)

    p = sys.argv[1]
    warmup()
    total_q = len(questions)
    
    for i, q in enumerate(questions[22:]):
        answer = None
        for attempt in range(3):
            try:
                answer = pipelines[p](q["question"])
                break
            except Exception as e:
                err_str = str(e)
                if "401" in err_str or "404" in err_str:
                    print(f"[FATAL] {q['id']} failed with {err_str}. Stopping.")
                    raise

                wait_s = 15 * (attempt + 1)
                print(
                    f"[WARN] {q['id']} attempt {attempt + 1}/3 failed: {err_str}. Retrying in {wait_s}s..."
                )
                time.sleep(wait_s)

        if answer is None:
            print(f"[ERROR] {q['id']} failed all 3 attempts. Stopping benchmark.")
            break

        status = "HIT" if answer["cache_hit"] else f"MISS ({answer['model_used'].split('/')[-1]})"
        print(
            f"[{i + 1}/{total_q}] {q['id']} ({q['difficulty']}) --> {status} | {answer['latency_ms']:.1f} ms"
        )

        if not answer["cache_hit"]:
            time.sleep(4)
