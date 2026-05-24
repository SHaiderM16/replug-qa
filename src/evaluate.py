import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

from src.utils import normalize, normalize_str


def compute_exact_match(predicted: str, gold_answers: list) -> float:
    # Compute EM score (max over multiple gold answers) using string normalization
    pred_normalized = normalize_str(predicted)

    max_em = 0.0
    for gold_answer in gold_answers:
        gold_normalized = normalize_str(gold_answer)
        em = 1.0 if pred_normalized == gold_normalized else 0.0
        max_em = max(max_em, em)

    return max_em


def compute_token_f1(predicted: str, gold_answers: list) -> float:
    # Compute token-level F1 (max over multiple gold answers) using Counter for multiplicities
    pred_tokens = normalize(predicted)  # Returns Counter
    
    if not pred_tokens:
        return 0.0
    
    max_f1 = 0.0
    for gold_answer in gold_answers:
        gold_tokens = normalize(gold_answer)  # Returns Counter

        if not gold_tokens:
            continue

        # Counter & gives minimum counts (intersection with multiplicities)
        intersection = pred_tokens & gold_tokens
        precision = sum(intersection.values()) / sum(pred_tokens.values())
        recall = sum(intersection.values()) / sum(gold_tokens.values())

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        max_f1 = max(max_f1, f1)

    return max_f1


def evaluate_condition(predictions: dict, gold_data: dict) -> dict:
    # Evaluate EM and F1 for one condition using standard NQ normalization
    em_scores = []
    f1_scores = []

    for pred in predictions:
        question_id = pred['question_id']
        if question_id not in gold_data:
            continue

        predicted_answer = pred['predicted_answer']
        gold_answers = gold_data[question_id]['answers']

        em = compute_exact_match(predicted_answer, gold_answers)
        f1 = compute_token_f1(predicted_answer, gold_answers)

        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = np.mean(em_scores) if em_scores else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

    return {'em': avg_em, 'f1': avg_f1}


def evaluate_all_k_values(ensemble_results: dict, gold_data: dict, k_values: list):
    # Evaluate ensemble at different k values
    results = {}

    for k in k_values:
        k_predictions = []
        k_str = str(k)  # Convert to string to match ensemble_results keys
        for query_id, query_results in ensemble_results.items():
            if k_str in query_results:
                k_predictions.append({
                    'question_id': query_id,
                    'predicted_answer': query_results[k_str]
                })

        if k_predictions:
            results[f'replug_k{k}'] = evaluate_condition(k_predictions, gold_data)

    return results


def generate_k_variance_plot(all_results: dict, output_path: str):
    # Generate EM vs k plot
    k_values = []
    em_scores = []
    
    # Extract REPLUG scores at different k values
    for condition_name, scores in all_results.items():
        if condition_name.startswith('replug_k'):
            k = int(condition_name.split('_k')[1])
            k_values.append(k)
            em_scores.append(scores['em'])
    
    # Sort by k value
    sorted_pairs = sorted(zip(k_values, em_scores))
    k_values = [k for k, _ in sorted_pairs]
    em_scores = [em for _, em in sorted_pairs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, em_scores, marker='o', linewidth=2, label='REPLUG')
    
    # Add baseline lines if available
    if 'no_retrieval' in all_results:
        no_retrieval_em = all_results['no_retrieval']['em']
        plt.axhline(y=no_retrieval_em, linestyle='--', label='No Retrieval')
    
    if 'random' in all_results:
        random_em = all_results['random']['em']
        plt.axhline(y=random_em, linestyle=':', label='Random Retrieval')
    
    plt.xlabel('k (number of retrieved passages)')
    plt.ylabel('Exact Match Score')
    plt.title('REPLUG Performance vs. k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved k-variance plot to {output_path}")


def load_gold_answers(nq_data_path: str) -> dict:
    # Load NQ gold answers from JSON
    with open(nq_data_path) as f:
        data = json.load(f)
    
    gold_data = {}
    for item in data:
        gold_data[item['id']] = {
            'question': item['question'],
            'answers': item['answers']
        }
    
    return gold_data


def main_evaluation(baseline_results_file: str, ensemble_results_file: str, nq_data_file: str):
    # Orchestrate full evaluation pipeline
    gold_data = load_gold_answers(nq_data_file)

    # Load baseline results
    with open(baseline_results_file) as f:
        baseline_results = json.load(f)

    # Load ensemble results
    with open(ensemble_results_file) as f:
        ensemble_results = json.load(f)

    # Evaluate baselines
    all_results = {}

    for condition, predictions in baseline_results.items():
        all_results[condition] = evaluate_condition(predictions, gold_data)

    # Evaluate ensemble at different k values
    k_values = [1, 2, 5, 10, 15]
    ensemble_evaluated = evaluate_all_k_values(ensemble_results, gold_data, k_values)
    all_results.update(ensemble_evaluated)
    
    # Save results
    results_file = Path("data/results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved results to {results_file}")
    
    # Generate plot
    plot_file = Path("data/k_variance.png")
    generate_k_variance_plot(all_results, str(plot_file))
    
    return all_results


if __name__ == "__main__":
    results = main_evaluation(
        'data/baseline_results.json',
        'data/ensemble_results.json',
        'data/queries.json'
    )
