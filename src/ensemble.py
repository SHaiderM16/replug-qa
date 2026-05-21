import json
import numpy as np
from scipy.special import logsumexp
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter

from src.utils import (
    normalize,
    normalize_str,
    make_cache_key,
    load_cache,
    save_cache_entry,
    groq_api_call_with_retry,
)
from src.dense_index import load_faiss_index


def compute_lambda_weights(cosine_scores):
    # Compute numerically stable log-softmax weights
    return cosine_scores - logsumexp(cosine_scores)


def get_query_embedding(query_text: str) -> np.ndarray:
    # Generate query embedding using Contriever
    model_name = "facebook/contriever"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(query_text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    query_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # L2-normalize
    query_emb = query_emb / np.linalg.norm(query_emb)
    return query_emb.astype(np.float32)


def generate_answers_for_passages(
    client, query, passages, cache_file, few_shot_examples
):
    # Generate LLM answers for each passage with caching
    passage_responses = []
    cache = load_cache(cache_file)

    for passage in passages:
        prompt = f"Knowledge: {passage}\nQuestion: {query}\nAnswer:"

        # Add few-shot examples
        for ex in few_shot_examples:
            prompt = f"Knowledge: {ex.get('passage', '')}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n{prompt}"

        cache_key = make_cache_key(prompt)

        if cache_key in cache:
            response = cache[cache_key]
            answer = response["answer"]
            if answer is None or not answer.strip():
                answer = ""

            passage_responses.append({"passage": passage, "answer": answer})
        else:
            api_response = groq_api_call_with_retry(client, prompt)
            if api_response is None:
                passage_responses.append({"passage": passage, "answer": ""})
            else:
                # Handle empty or None answers
                answer = api_response["answer"]
                if answer is None or not answer.strip():
                    answer = ""

                passage_responses.append({"passage": passage, "answer": answer})
                save_cache_entry(cache_file, {"cache_key": cache_key, "answer": answer})

    return passage_responses


def compute_weighted_scores(passage_responses, log_lambda, k_values):
    """Majority voting over top-k answers (approximation due to Groq logprobs limitation)."""
    results = {}
    for k in k_values:
        # Get indices of top-k passages by lambda (descending)
        top_k_idx = np.argsort(log_lambda)[-k:]   # ascending -> last k are highest lambda
        top_k_answers = [passage_responses[i]['answer'] for i in top_k_idx]

        # Normalize answers (skip empty/whitespace)
        norm_answers = [normalize_str(a) for a in top_k_answers if a and a.strip()]
        if not norm_answers:
            results[k] = ""
            continue

        # Majority vote
        winner_norm = Counter(norm_answers).most_common(1)[0][0]

        # Return the original answer (first matching winner)
        for a in top_k_answers:
            if normalize_str(a) == winner_norm:
                results[k] = a
                break
    return results


def run_replug_ensemble(queries_file, passages_file, index_file, examples_file, client):
    # Run REPLUG ensemble for all queries
    with open(queries_file) as f:
        queries = json.load(f)

    with open(passages_file) as f:
        passages = json.load(f)

    with open(examples_file) as f:
        examples = json.load(f)

    index = load_faiss_index()
    cache_file = Path("data/ensemble_cache.jsonl")
    results_file = Path("data/ensemble_results.json")

    all_results = {}
    k_values = [1, 2, 5, 10]

    for query in queries:
        query_text = query["question"]

        # Generate query embedding using Contriever
        query_embedding = get_query_embedding(query_text)

        # Retrieve top-10 passages
        scores, indices = index.search(query_embedding.reshape(1, -1), k=10)
        top_passages = [passages[i] for i in indices[0]]

        # Compute lambda weights
        log_lambda = compute_lambda_weights(scores[0])

        # Generate answers
        passage_responses = generate_answers_for_passages(
            client, query_text, top_passages, cache_file, examples
        )

        # Compute weighted scores for each k
        weighted_results = compute_weighted_scores(
            passage_responses, log_lambda, k_values
        )

        all_results[query["id"]] = weighted_results

    # Save results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved ensemble results to {results_file}")
    return all_results


if __name__ == "__main__":
    from groq import Groq
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    results = run_replug_ensemble(
        "data/queries.json",
        "data/passages.json",
        "data/faiss.index",
        "data/examples.json",
        client,
    )
