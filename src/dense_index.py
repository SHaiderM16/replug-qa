import numpy as np
import faiss
from pathlib import Path


def build_faiss_index():
    # Build FAISS IndexFlatIP from Contriever embeddings
    embeddings_path = Path("data/embeddings.npy")
    index_path = Path("data/faiss.index")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)

    # Verify shape and dtype
    assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}"
    assert embeddings.shape[1] == 768, f"Wrong embedding dimension: {embeddings.shape[1]}"
    assert embeddings.dtype == np.float32, f"Wrong dtype: {embeddings.dtype}"

    # Verify L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings not L2-normalized"

    print(f"Loaded embeddings: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Build FAISS IndexFlatIP (inner product = cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(768)
    index.add(embeddings)
    print(f"Built FAISS index with {index.ntotal} vectors")

    # Save index
    faiss.write_index(index, str(index_path))
    print(f"Saved index to {index_path}")

    return index


def load_faiss_index():
    # Load saved FAISS index
    index_path = Path("data/faiss.index")
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index = faiss.read_index(str(index_path))
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index


if __name__ == "__main__":
    index = build_faiss_index()
    print("Index construction complete")
