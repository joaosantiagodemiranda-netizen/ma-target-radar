"""Step 6 - Text representations of each 10-K business description.

Two representations, so the paper can compare a classical and a deep-learning one:

1. TF-IDF cosine similarity (Hoberg & Phillips-style product-market similarity):
   per year, top-N most similar firms -> tfidf_neighbors.parquet
2. Transformer sentence embeddings (multilingual MiniLM): first EMBED_MAX_CHUNKS
   chunks of EMBED_CHUNK_WORDS words, mean-pooled, L2-normalised
   -> embeddings.npy (float16) aligned row-by-row with doc_index.parquet

Run on a GPU (Colab T4: ~15-20 min for all years). The multilingual model puts
Portuguese descriptions in the same space -> Brazil peer/buyer demo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


def load_docs() -> pd.DataFrame:
    files = sorted((config.PROCESSED / "text").glob("item1_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.reset_index(drop=True)


def chunks(text: str, n_words: int = config.EMBED_CHUNK_WORDS,
           max_chunks: int = config.EMBED_MAX_CHUNKS) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + n_words]) for i in range(0, min(len(w), n_words * max_chunks), n_words)]


def embed(docs: pd.DataFrame, model_name: str = config.EMBED_MODEL, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # heavy import, GPU step only
    model = SentenceTransformer(model_name)
    all_chunks, owner = [], []
    for i, t in enumerate(docs["item1"]):
        cs = chunks(t)
        all_chunks.extend(cs)
        owner.extend([i] * len(cs))
    print(f"encoding {len(all_chunks):,} chunks from {len(docs):,} documents")
    vecs = model.encode(all_chunks, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    owner = np.asarray(owner)
    out = np.zeros((len(docs), vecs.shape[1]), dtype=np.float32)
    np.add.at(out, owner, vecs)
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)
    return out


def tfidf_neighbors(docs: pd.DataFrame, k: int = config.N_NEIGHBORS) -> pd.DataFrame:
    """Within-year TF-IDF cosine neighbours (no cross-year information)."""
    rows = []
    for y, g in docs.groupby("year"):
        vec = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.25,
                              sublinear_tf=True, max_features=50_000,
                              token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
        X = vec.fit_transform(g["item1"])
        ciks = g["cik"].to_numpy()
        for start in range(0, X.shape[0], 1000):
            S = (X[start:start + 1000] @ X.T).toarray()
            for r in range(S.shape[0]):
                S[r, start + r] = -1                       # exclude self
            top = np.argpartition(-S, k, axis=1)[:, :k]
            for r in range(S.shape[0]):
                for j in top[r]:
                    rows.append((y, ciks[start + r], ciks[j], float(S[r, j])))
        print(f"  tfidf {y}: {X.shape[0]:,} docs, vocab {X.shape[1]:,}")
    return pd.DataFrame(rows, columns=["year", "cik", "nbr_cik", "sim"]).astype(
        {"sim": "float32"})


def build(skip_embeddings: bool = False) -> None:
    docs = load_docs()
    idx_cols = ["cik", "year", "filename", "accession", "n_words", "pending_deal"] + \
        [c for c in docs.columns if c.startswith("sig_")]
    docs[idx_cols].to_parquet(config.PROCESSED / "doc_index.parquet", index=False)
    docs.assign(desc=docs["item1"].str[:600])[["cik", "year", "desc"]].to_parquet(
        config.PROCESSED / "descriptions.parquet", index=False)
    tfidf_neighbors(docs).to_parquet(config.PROCESSED / "tfidf_neighbors.parquet", index=False)
    if not skip_embeddings:
        E = embed(docs)
        np.save(config.PROCESSED / "embeddings.npy", E.astype(np.float16))
        print(f"embeddings.npy: {E.shape}")


if __name__ == "__main__":
    build()
