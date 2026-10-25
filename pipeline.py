from rag import ask
from logger import log_request,show_all
from cache import lookup,add_qa
import time
def answer_cached(question):
    start = time.perf_counter()
    answer,sim = lookup(question)
    if answer is not None:
        cache_hit = 1
        elapsed_ms = (time.perf_counter()-start) * 1000
        log_request(question,answer,"cache",cache_hit,0,0,elapsed_ms,pipeline="cache")
    else:
        response = ask(question)
        elapsed_ms = (time.perf_counter()-start) * 1000
        answer = response.choices[0].message.content
        cache_hit = 0
        log_request(question,answer,response.model,cache_hit,response.usage.prompt_tokens,response.usage.completion_tokens,elapsed_ms,pipeline="cache")
        add_qa(question,answer)
    return answer

if __name__ == "__main__":
    res = answer_cached("How many paid sick days do employees receive per calendar year?")
    print(res)