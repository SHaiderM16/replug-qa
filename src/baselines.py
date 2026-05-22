import json
import random
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

from src.utils import normalize, make_cache_key, load_cache, save_cache_entry, cohere_api_call_with_retry


def build_prompt(question, passage=None, examples=[]):
    # Build 3-shot prompt with optional retrieved passage
    lines = []
    
    if passage:
        lines.append(f"Knowledge: {passage}")
    
    for ex in examples:
        lines.append(f"Question: {ex['question']}")
        lines.append(f"Answer: {ex['answer']}")
    
    lines.append(f"Question: {question}")
    lines.append("Answer (1-5 words only, no explanation, no extra text):")

    return "\n".join(lines)


def run_no_retrieval_baseline(queries, examples, client, cache_file):
    # Run LLM-only baseline (no retrieval)
    results = []
    cache = load_cache(cache_file)
    
    for query in queries:
        prompt = build_prompt(query['question'], examples=examples)
        cache_key = make_cache_key(prompt)
        
        if cache_key in cache:
            answer = cache[cache_key]['answer']
        else:
            response = cohere_api_call_with_retry(client, prompt)
            if response is None:
                answer = ""
            else:
                answer = response['answer']
                save_cache_entry(cache_file, {
                    'cache_key': cache_key,
                    'answer': answer
                })
                cache[cache_key] = {'answer': answer}
        
        results.append({
            'question_id': query['id'],
            'predicted_answer': answer
        })
    
    return results


def run_bm25_baseline(queries, passages, examples, client, cache_file):
    # Run BM25 + RAG baseline
    tokenized_passages = [p['text'].split() for p in passages]
    bm25 = BM25Okapi(tokenized_passages)

    results = []
    cache = load_cache(cache_file)

    for query in queries:
        tokenized_query = query['question'].split()
        scores = bm25.get_scores(tokenized_query)
        top_idx = np.argmax(scores)
        passage = passages[top_idx]['text']

        prompt = build_prompt(query['question'], passage=passage, examples=examples)
        cache_key = make_cache_key(prompt)
        
        if cache_key in cache:
            answer = cache[cache_key]['answer']
        else:
            response = cohere_api_call_with_retry(client, prompt)
            if response is None:
                answer = ""
            else:
                answer = response['answer']
                save_cache_entry(cache_file, {
                    'cache_key': cache_key,
                    'answer': answer
                })
                cache[cache_key] = {'answer': answer}
        
        results.append({
            'question_id': query['id'],
            'predicted_answer': answer
        })
    
    return results


def run_random_baseline(queries, passages, examples, client, cache_file):
    # Run random retrieval + RAG baseline
    random.seed(42)
    
    results = []
    cache = load_cache(cache_file)
    
    for query in queries:
        random_passages = random.sample(passages, min(10, len(passages)))
        passage = random_passages[0]['text']

        prompt = build_prompt(query['question'], passage=passage, examples=examples)
        cache_key = make_cache_key(prompt)
        
        if cache_key in cache:
            answer = cache[cache_key]['answer']
        else:
            response = cohere_api_call_with_retry(client, prompt)
            if response is None:
                answer = ""
            else:
                answer = response['answer']
                save_cache_entry(cache_file, {
                    'cache_key': cache_key,
                    'answer': answer
                })
                cache[cache_key] = {'answer': answer}
        
        results.append({
            'question_id': query['id'],
            'predicted_answer': answer
        })
    
    return results


def run_all_baselines(queries_file, passages_file, examples_file, client):
    # Run all baseline experiments
    with open(queries_file) as f:
        queries = json.load(f)
    
    with open(passages_file) as f:
        passages = json.load(f)
    
    with open(examples_file) as f:
        examples = json.load(f)
    
    cache_file = Path("data/baseline_cache.jsonl")
    results_file = Path("data/baseline_results.json")
    
    print("Running no retrieval baseline...")
    no_retrieval_results = run_no_retrieval_baseline(queries, examples, client, cache_file)
    
    print("Running BM25 baseline...")
    bm25_results = run_bm25_baseline(queries, passages, examples, client, cache_file)
    
    print("Running random baseline...")
    random_results = run_random_baseline(queries, passages, examples, client, cache_file)
    
    results = {
        'no_retrieval': no_retrieval_results,
        'bm25': bm25_results,
        'random': random_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {results_file}")
    return results


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv('COHERE_API_KEY'):
        print("Error: COHERE_API_KEY not found in environment")
        exit(1)

    results = run_all_baselines(
        'data/queries.json',
        'data/passages.json',
        'data/examples.json',
        None
    )
