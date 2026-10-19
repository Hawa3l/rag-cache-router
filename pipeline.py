from rag import ask
from logger import log_request,show_all
from cache import lookup
import time
def answer_baseline(question):
    answer,cache_hit = answer_cached(question)
    if cache_hit == 1:
        log_request(question,answer,"cache",cache_hit,0,0,0,pipeline="cache")
    else:
        start = time.perf_counter()
        response = ask(question)
        end = time.perf_counter()
        elapsed_ms = (end-start) * 1000
        answer = response.choices[0].message.content
        cache_hit = 0
        log_request(question,answer,response.model,cache_hit,response.usage.prompt_tokens,response.usage.completion_tokens,elapsed_ms,pipeline="base")
    return answer

if __name__ == "__main__":
    pass