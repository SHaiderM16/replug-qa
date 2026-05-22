import json
import re
from pathlib import Path
import glob

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch

from src.utils import normalize_str


def normalize(text: str) -> str:
    # Simple normalize for test compatibility
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def load_nq_queries_with_gold_passages(passages_30k):
    nq = load_dataset("nq_open", split="validation")
    queries = []
    for i in range(50):
        ex = nq[i]
        answers = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
        queries.append(
            {
                "id": f"nq_dev_{i}",
                "question": ex["question"],
                "answers": answers,
                "gold_passage_id": None,
            }
        )
    for q in queries:
        norm_answers = [normalize_str(a) for a in q["answers"]]
        norm_answers = [a for a in norm_answers if len(a) >= 3]
        if not norm_answers:
            continue
        for p in passages_30k:
            norm_text = normalize_str(p["text"])
            if any(ans in norm_text for ans in norm_answers):
                q["gold_passage_id"] = p["id"]
                break
    found = sum(1 for q in queries if q["gold_passage_id"] is not None)
    print(f"Gold passages found: {found}/50")
    return queries


def stream_wiki_dpr_with_gold_first(gold_passage_ids, target_size=30000):
    # Single-pass corpus building - stream once, collect gold passages and others
    dataset = load_dataset(
        "wiki_dpr", "psgs_w100.nq.no_index", streaming=True, trust_remote_code=True
    )

    gold_passages = {}
    other_passages = []

    for example in dataset["train"]:
        passage_id = str(example["id"])
        passage_text = example["text"]

        if passage_id in gold_passage_ids and passage_id not in gold_passages:
            gold_passages[passage_id] = {"id": passage_id, "text": passage_text}
        elif len(other_passages) < target_size:
            other_passages.append({"id": passage_id, "text": passage_text})

        # Stop when we have all gold passages AND reached target size
        if len(gold_passages) == len(gold_passage_ids) and len(gold_passages) + len(other_passages) >= target_size:
            break

    # Check for missing gold passages
    missing_gold = set(gold_passage_ids) - set(gold_passages.keys())
    if missing_gold:
        print(f"Warning: Could not find {len(missing_gold)} gold passages")

    # Combine: gold passages first, then random to reach target
    passages = list(gold_passages.values())
    remaining = target_size - len(passages)
    passages.extend(other_passages[:remaining])

    print(f"Collected {len(passages)} passages ({len(gold_passages)} gold, {remaining} random)")
    return passages


def compute_contriever_embeddings(passages, batch_size=16, checkpoint_every=5000):
    # Compute L2-normalized Contriever embeddings with checkpointing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_size is None:
        batch_size = 64 if device == "cuda" else 16
    print(f"Device: {device}, batch_size: {batch_size}")

    # Prepare texts
    texts = [p["text"] if isinstance(p, dict) else p for p in passages]
    n = len(texts)

    # Checkpoint resume logic
    start_idx = 0
    existing_embeddings = []
    checkpoint_files = sorted(glob.glob("data/embeddings_checkpoint_*.npy"))
    if checkpoint_files:
        latest = checkpoint_files[-1]
        try:
            ckpt = np.load(latest)
            start_idx = ckpt.shape[0]
            existing_embeddings = [ckpt]
            print(f"Resuming from checkpoint: {latest} ({start_idx}/{n})")
        except Exception as e:
            print(f"Could not load checkpoint: {e}, starting fresh")

    if start_idx >= n:
        print("Embeddings already complete")
        return np.concatenate(existing_embeddings, axis=0) if existing_embeddings else np.zeros((n, 768), dtype=np.float32)

    print("Loading Contriever model...")
    model_name = "facebook/contriever"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    all_embeddings = []
    processed_count = start_idx  # Track total passages processed for checkpointing

    print(f"Computing embeddings for {n} passages...")
    for i in range(start_idx, n, batch_size):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        # Masked mean pooling - exclude padding tokens
        token_emb = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        batch_emb = (token_emb * mask).sum(dim=1) / mask_sum
        batch_embeddings = batch_emb.cpu().numpy()
        all_embeddings.append(batch_embeddings)

        processed_count += len(batch)

        # Save checkpoint periodically
        if processed_count % checkpoint_every == 0:
            checkpoint = np.concatenate(existing_embeddings + all_embeddings, axis=0)
            checkpoint_path = f"data/embeddings_checkpoint_{len(checkpoint)}.npy"
            np.save(checkpoint_path, checkpoint)
            print(f"Saved checkpoint: {checkpoint_path}")

        if processed_count % 1000 == 0:
            print(f"Processed {processed_count}/{n} passages")

    final_embeddings = np.concatenate(existing_embeddings + all_embeddings, axis=0)

    # L2-normalize for FAISS IndexFlatIP
    norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True)
    final_embeddings = final_embeddings / norms

    # Cast to float32 for memory efficiency
    final_embeddings = final_embeddings.astype(np.float32)

    # Verify embedding properties
    assert final_embeddings.shape == (n, 768), f"Wrong shape: {final_embeddings.shape}"
    assert final_embeddings.dtype == np.float32, f"Wrong dtype: {final_embeddings.dtype}"

    # Check L2 normalization
    norms = np.linalg.norm(final_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Not L2-normalized: min={norms.min():.6f}, max={norms.max():.6f}"

    # Check for NaN or Inf
    assert not np.isnan(final_embeddings).any(), "Embeddings contain NaN values"
    assert not np.isinf(final_embeddings).any(), "Embeddings contain Inf values"

    print(f"Verification passed: {final_embeddings.shape} embeddings, dtype={final_embeddings.dtype}")

    return final_embeddings


def save_passages_and_embeddings(passages, embeddings):
    # Save passages as JSON and embeddings as numpy array
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    passages_path = data_dir / "passages.json"
    with open(passages_path, "w") as f:
        json.dump(passages, f)

    embeddings_path = data_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    print(f"Saved {len(passages)} passages to {passages_path}")
    print(f"Saved embeddings shape {embeddings.shape} to {embeddings_path}")

    # Verify shape and dtype
    assert len(passages) == embeddings.shape[0], f"Passage count mismatch: {len(passages)} vs {embeddings.shape[0]}"
    assert embeddings.shape[1] == 768, f"Wrong embedding dimension: {embeddings.shape[1]}"
    assert embeddings.dtype == np.float32, f"Wrong dtype: {embeddings.dtype}"

    # Verify L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0), "Embeddings not L2-normalized"

    print("Verification passed!")


def main_data_prep():
    # Load NQ queries and extract answer strings for gold passage matching
    nq = load_dataset("nq_open", split="validation")
    queries = []
    all_answers = set()

    for i in range(50):
        ex = nq[i]
        answers = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
        queries.append(
            {
                "id": f"nq_dev_{i}",
                "question": ex["question"],
                "answers": answers,
                "gold_passage_id": None,
            }
        )
        # Pre-normalize answers for matching
        for a in answers:
            norm_a = normalize_str(a)
            if len(norm_a) >= 3:
                all_answers.add(norm_a)

    # Stream wiki_dpr once, collecting gold passages and random passages
    dataset = load_dataset(
        "wiki_dpr", "psgs_w100.nq.no_index", streaming=True, trust_remote_code=True
    )

    gold_passages = {}
    other_passages = []

    for example in dataset["train"]:
        passage_id = str(example["id"])
        passage_text = example["text"]
        norm_text = normalize_str(passage_text)

        # Check if any answer appears in this passage (gold passage)
        is_gold = (
            passage_id not in gold_passages
            and any(ans in norm_text for ans in all_answers)
        )

        if is_gold:
            gold_passages[passage_id] = {"id": passage_id, "text": passage_text}
        elif len(other_passages) < 30000:
            other_passages.append({"id": passage_id, "text": passage_text})

        # Stop when we have enough passages
        if len(gold_passages) + len(other_passages) >= 30000:
            break

    # Match gold passage IDs to queries
    passage_norm_texts = {p["id"]: normalize_str(p["text"]) for p in (list(gold_passages.values()) + other_passages)}
    for q in queries:
        norm_answers = [normalize_str(a) for a in q["answers"] if len(normalize_str(a)) >= 3]
        if not norm_answers:
            continue
        for pid in passage_norm_texts:
            if any(ans in passage_norm_texts[pid] for ans in norm_answers):
                q["gold_passage_id"] = pid
                break

    found = sum(1 for q in queries if q["gold_passage_id"] is not None)
    print(f"Gold passages found: {found}/50")

    # Build final corpus: gold passages first, then random
    passages = list(gold_passages.values())
    remaining = 30000 - len(passages)
    passages.extend(other_passages[:remaining])

    print(f"Collected {len(passages)} passages ({len(gold_passages)} gold, {remaining} random)")

    # Save queries
    queries_path = Path("data/queries.json")
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queries_path, "w") as f:
        json.dump(queries, f)
    print(f"Saved queries to {queries_path}")

    # Compute embeddings
    embeddings = compute_contriever_embeddings(passages)
    save_passages_and_embeddings(passages, embeddings)


if __name__ == "__main__":
    main_data_prep()
