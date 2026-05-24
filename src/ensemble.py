import json
import numpy as np
from scipy.special import logsumexp as scipy_logsumexp
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

from src.utils import (
    normalize_str,
    make_cache_key,
    load_cache,
    save_cache_entry,
    cohere_api_call_with_retry,
)
from src.dense_index import load_faiss_index


# Refusal detection safeguard
REFUSAL_PHRASES = [
    "i'm sorry", "i don't have", "not enough information",
    "cannot answer", "no information", "knowledge not found"
]

def is_refusal(answer: str) -> bool:
    return any(phrase in answer.lower() for phrase in REFUSAL_PHRASES)


def compute_lambda_weights(cosine_scores):
    # Compute numerically stable log-softmax weights
    return cosine_scores - scipy_logsumexp(cosine_scores)


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
        passage_text = passage['text'] if isinstance(passage, dict) else passage

        # Build prompt: few-shot examples without Knowledge prefix, then actual query with passage
        lines = []
        for ex in few_shot_examples:
            lines.append(f"Question: {ex['question']}")
            lines.append(f"Answer: {ex['answer']}")
        lines.append(f"Knowledge: {passage_text}")
        lines.append(f"Question: {query}")
        lines.append("Answer (1-5 words only, no explanation, no extra text):")
        prompt = "\n".join(lines)

        cache_key = make_cache_key(prompt)

        if cache_key in cache:
            response = cache[cache_key]
            answer = response["answer"]
            total_logprob = response.get("total_logprob", float("-inf"))
            if answer is None or not answer.strip():
                answer = ""

            passage_responses.append({
                "passage": passage,
                "answer": answer,
                "total_logprob": total_logprob
            })
        else:
            api_response = cohere_api_call_with_retry(client, prompt)
            if api_response is None:
                passage_responses.append({"passage": passage, "answer": ""})
            else:
                # Handle empty or None answers
                answer = api_response["answer"]
                if answer is None or not answer.strip():
                    answer = ""

                passage_responses.append({
                    "passage": passage,
                    "answer": answer,
                    "total_logprob": api_response["total_logprob"]
                })
                save_cache_entry(cache_file, {
                    "cache_key": cache_key,
                    "answer": answer,
                    "total_logprob": api_response["total_logprob"]
                })

    return passage_responses


def compute_weighted_scores(passage_responses, log_lambda, k_values):
    # λ-weighted logprob scoring per REPLUG Section 3.2
    # Each answer's score = logsumexp(λ_i + logprob_i) over passages giving that answer
    results = {}
    for k in k_values:
        # Get indices of top-k passages by lambda (descending)
        top_k_idx = np.argsort(log_lambda)[-k:]

        # Collect weighted scores for each unique normalized answer
        answer_groups = {}

        for idx in top_k_idx:
            resp = passage_responses[idx]
            answer = resp['answer']
            logprob = resp.get('total_logprob', float('-inf'))

            # Skip refusal answers and empty answers
            if is_refusal(answer) or not answer or not answer.strip():
                continue

            # Normalize answer for deduplication
            norm_answer = normalize_str(answer)

            # Compute λ-weighted logprob (λ is already in log space)
            weighted_score = log_lambda[idx] + logprob

            if norm_answer not in answer_groups:
                answer_groups[norm_answer] = []
            answer_groups[norm_answer].append((weighted_score, answer))

        # Edge case: all answers filtered out
        if not answer_groups:
            results[k] = "I don't know"
            continue

        # Aggregate each group using logsumexp (proper probability sum in log space)
        final_scores = {}
        original_answers = {}
        for norm_answer, scores in answer_groups.items():
            weighted_scores_only = [s[0] for s in scores]
            final_scores[norm_answer] = scipy_logsumexp(weighted_scores_only)
            # Store the original answer (first occurrence for this normalized form)
            original_answers[norm_answer] = scores[0][1]

        # Select answer with highest aggregated score
        winner_norm = max(final_scores.items(), key=lambda x: x[1])[0]
        results[k] = original_answers[winner_norm]

    return results


def run_replug_ensemble(queries_file, passages_file, index_file, examples_file, client):
    # Run REPLUG ensemble for all queries with incremental checkpointing
    with open(queries_file) as f:
        queries = json.load(f)

    with open(passages_file) as f:
        passages = json.load(f)

    with open(examples_file) as f:
        examples = json.load(f)

    index = load_faiss_index()
    cache_file = Path("data/ensemble_cache.jsonl")
    results_file = Path("data/ensemble_results.json")

    # Load existing results if any (for resume capability)
    all_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Resuming from {len(all_results)} existing results")

    k_values = [1, 2, 5, 10, 15]
    checkpoint_interval = 5  # Save every 5 queries

    for idx, query in enumerate(queries):
        qid = query["id"]

        # Skip if already processed
        if qid in all_results:
            print(f"Skipping {qid} (already done)")
            continue

        query_text = query["question"]

        # Generate query embedding using Contriever
        query_embedding = get_query_embedding(query_text)

        # Retrieve top-15 passages
        scores, indices = index.search(query_embedding.reshape(1, -1), k=15)
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

        all_results[qid] = weighted_results
        print(f"Processed {idx+1}/{len(queries)}: {qid}")

        # Incremental checkpoint save
        if (idx + 1) % checkpoint_interval == 0 or (idx + 1) == len(queries):
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Checkpoint saved ({len(all_results)} queries)")

    # Final save
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved ensemble results to {results_file}")
    return all_results


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("COHERE_API_KEY"):
        print("Error: COHERE_API_KEY not found in environment")
        exit(1)

    results = run_replug_ensemble(
        "data/queries.json",
        "data/passages.json",
        "data/faiss.index",
        "data/examples.json",
        None,
    )
