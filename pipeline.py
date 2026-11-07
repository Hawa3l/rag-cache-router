from rag import ask
from logger import log_request, show_all, clear_db
from cache import lookup, add_qa, clear_cache
from router import route_query
from config import LARGE_MODEL
import time


def answer_cached(question):
    start = time.perf_counter()
    answer, _ = lookup(question)
    if answer is not None:
        cache_hit = 1
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_request(question, answer, "cache", cache_hit, 0, 0, elapsed_ms, pipeline="cached")
    else:
        response = ask(question)
        answer = response.choices[0].message.content
        add_qa(question, answer)
        cache_hit = 0
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_request(
            question,
            answer,
            LARGE_MODEL,
            cache_hit,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            elapsed_ms,
            pipeline="cached",
        )
    return {
        "answer": answer,
        "model_used": LARGE_MODEL if cache_hit == 0 else "cache",
        "cache_hit": False if cache_hit == 0 else True,
        "latency_ms": elapsed_ms,
    }


def answer_baseline(question):
    t0 = time.perf_counter()
    response = ask(question)
    latency_ms = (time.perf_counter() - t0) * 1000
    answer = response.choices[0].message.content
    cache_hit = 0
    log_request(
        question,
        answer,
        LARGE_MODEL,
        cache_hit,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        latency_ms,
        pipeline="baseline",
    )
    return {
        "answer": answer,
        "model_used": LARGE_MODEL,
        "cache_hit": False,
        "latency_ms": latency_ms,
    }


def answer_routed(question):
    start = time.perf_counter()
    answer, _ = lookup(question)
    if answer is not None:
        cache_hit = 1
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_request(question, answer, "cache", cache_hit, 0, 0, elapsed_ms, pipeline="routed")
        return {
            "answer": answer,
            "model_used": "cache",
            "cache_hit": True,
            "latency_ms": elapsed_ms,
        }

    chosen_model = route_query(question)
    response = ask(question, model_name=chosen_model)
    answer = response.choices[0].message.content
    add_qa(question, answer)
    cache_hit = 0
    elapsed_ms = (time.perf_counter() - start) * 1000
    log_request(
        question,
        answer,
        chosen_model,
        cache_hit,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        elapsed_ms,
        pipeline="routed",
    )
    return {
        "answer": answer,
        "model_used": chosen_model,
        "cache_hit": False,
        "latency_ms": elapsed_ms,
    }


if __name__ == "__main__":
    clear_cache()
    clear_db()
