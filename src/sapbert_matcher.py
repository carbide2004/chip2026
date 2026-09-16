"""[GPU 进阶模块] SapBERT 离线编码与 HPO 向量检索工具。

适合在配有 GPU / 显卡的机器上运行：
1. 对 hp.obo 的 1.7 万个表型概念构建 SapBERT 向量索引 (FAISS)；
2. 对文章中未被词典硬命中的生物医学候选短语进行近邻检索与标准化。
"""

from pathlib import Path
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import json

# Ensure domestic mirror by default if not specified
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import HPOntology

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class SapBERTRetriever:
    """Encodes HPO concepts and queries using SapBERT."""

    def __init__(
        self,
        obo_path: str,
        index_dir: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        self.ontology = HPOntology(obo_path)
        self.index_dir = Path(index_dir) if index_dir else repo_root / "data" / "processed" / "sapbert_index"

        # Lazy import of heavy GPU dependencies
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            import faiss

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
            self.faiss = faiss
        except ImportError as e:
            raise ImportError(
                f"SapBERT requires PyTorch, Transformers and FAISS: {e}. "
                "Run `pip install torch transformers sentence-transformers faiss-gpu faiss-cpu`."
            )

        self.ids_list: List[str] = []
        self.phrases_list: List[str] = []
        self.index = None

    def build_index(self, batch_size: int = 256) -> None:
        """Encode all HPO concept names and synonyms into a FAISS index."""
        print("Gathering HPO terms for SapBERT indexing...")
        terms_to_encode = []
        for term_id, term in self.ontology.terms.items():
            if term.is_obsolete or not term.is_sub_abnormality:
                continue
            if term.name:
                terms_to_encode.append((term_id, term.name))
            for syn in term.synonyms:
                terms_to_encode.append((term_id, syn))

        self.ids_list = [t[0] for t in terms_to_encode]
        self.phrases_list = [t[1] for t in terms_to_encode]
        print(f"Total phrases to encode: {len(self.phrases_list)}")

        all_embeddings = []
        with self.torch.no_grad():
            for i in range(0, len(self.phrases_list), batch_size):
                batch_texts = self.phrases_list[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                # CLS token embedding
                cls_rep = outputs.last_hidden_state[:, 0, :]
                # Normalize for cosine similarity
                cls_norm = cls_rep / cls_rep.norm(dim=1, keepdim=True)
                all_embeddings.append(cls_norm.cpu().numpy())

        import numpy as np

        mat = np.vstack(all_embeddings).astype("float32")
        dim = mat.shape[1]
        self.index = self.faiss.IndexFlatIP(dim)
        self.index.add(mat)
        print(f"FAISS index created with {self.index.ntotal} vectors.")
        self.save_index()

    def save_index(self) -> None:
        """Save FAISS index and metadata to index_dir."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        idx_file = self.index_dir / "sapbert_hpo.faiss"
        meta_file = self.index_dir / "sapbert_meta.json"
        print(f"Saving FAISS index to {idx_file}...")
        self.faiss.write_index(self.index, str(idx_file))
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "ids_list": self.ids_list,
                "phrases_list": self.phrases_list,
            }, f, ensure_ascii=False)
        print("Index saved successfully.")

    def load_index(self) -> bool:
        """Attempt to load existing FAISS index and metadata."""
        idx_file = self.index_dir / "sapbert_hpo.faiss"
        meta_file = self.index_dir / "sapbert_meta.json"
        if not idx_file.exists() or not meta_file.exists():
            return False
        print(f"Loading existing FAISS index from {idx_file}...")
        self.index = self.faiss.read_index(str(idx_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.ids_list = data["ids_list"]
            self.phrases_list = data["phrases_list"]
        print(f"FAISS index loaded ({self.index.ntotal} terms).")
        return True

    def search_candidates(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """Search Top-K closest HPO concepts for a given phrase query."""
        res = self.search_candidates_batch([query], top_k=top_k)
        return res[0] if res else []

    def search_candidates_batch(
        self, queries: List[str], top_k: int = 5, batch_size: int = 128
    ) -> List[List[Tuple[str, str, float]]]:
        """Search Top-K closest HPO concepts for a batch of queries."""
        if self.index is None:
            if not self.load_index():
                raise RuntimeError("Index not built. Call build_index() first.")

        if not queries:
            return []

        all_results = []
        with self.torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)
                cls_rep = outputs.last_hidden_state[:, 0, :]
                q_norm = (cls_rep / cls_rep.norm(dim=1, keepdim=True)).cpu().numpy().astype("float32")
                scores, indices = self.index.search(q_norm, top_k)
                for s_row, idx_row in zip(scores, indices):
                    row_res = []
                    for s, idx in zip(s_row, idx_row):
                        if idx >= 0:
                            row_res.append((self.ids_list[idx], self.phrases_list[idx], float(s)))
                    all_results.append(row_res)
        return all_results


if __name__ == "__main__":
    obo = repo_root / "data" / "raw" / "PatientPheX-V1-A" / "hp.obo"
    retriever = SapBERTRetriever(obo_path=str(obo))
    if not retriever.load_index():
        retriever.build_index(batch_size=512)

    test_queries = [
        "SNHL",
        "developmental delays",
        "diabetes",
        "high blood pressure",
        "intellectual disability",
        "short stature",
    ]
    print("\nTesting SapBERT Candidate Search:")
    for q in test_queries:
        res = retriever.search_candidates(q, top_k=3)
        print(f"\nQuery: '{q}'")
        for hpo_id, phrase, score in res:
            print(f"  -> {hpo_id} ({phrase}): score={score:.4f}")
