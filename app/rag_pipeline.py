"""
rag_pipeline.py — shared RAG retrieval module

"""
 
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
 
class RAGPipeline:
    def __init__(self, index_path: str, metadata_path: str, k: int = 3):
        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        print(f"  {self.index.ntotal:,} vectors loaded")
 
        print("Loading metadata...")
        self.metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                self.metadata.append(json.loads(line.strip()))
 
        print("Loading embedder...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.k = k
        print("RAGPipeline ready ✓")
 
    def retrieve(self, query: str, k: int = None) -> list[dict]:
        k = k or self.k
        q_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
 
        distances, indices = self.index.search(q_emb, k)
 
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            meta = self.metadata[idx]
            results.append({
                "rank":         rank + 1,
                "distance":     float(dist),
                "chunk_id":     meta.get("chunk_id", str(idx)),
                "paper_id":     meta["paper_id"],
                "title":        meta.get("title", "N/A"),
                "url":          meta.get("url", ""),
                "year":         meta.get("year", ""),
                "text_preview": meta.get("text_preview", "")[:500],
            })
        return results
 
    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        context_parts = []
        for c in chunks:
            context_parts.append(
                f"[Source {c['rank']}: {c['title']} ({c['year']})]\n"
                f"{c['text_preview'][:400]}"
            )
        context_block = "\n\n".join(context_parts)
 
        return (
            f"You are a research assistant specializing in machine learning.\n"
            f"Use the following retrieved paper excerpts to answer the question.\n"
            f"If the answer is not in the context, say so.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"Question: {query}\nAnswer:"
        )
 
    def query(self, question: str, k: int = None) -> tuple[str, list[dict]]:
        """Convenience method: retrieve + build prompt in one call."""
        chunks = self.retrieve(question, k=k)
        prompt = self.build_prompt(question, chunks)
        return prompt, chunks
