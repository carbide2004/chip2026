"""[GPU 进阶模块] SapBERT 离线编码与 HPO 向量检索工具。

适合在配有 GPU / 显卡的机器上运行：
1. 对 hp.obo 的 1.7 万个表型概念构建 SapBERT 向量索引 (FAISS)；
2. 对文章中未被词典硬命中的生物医学候选短语进行近邻检索与标准化。
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

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

    def search_candidates(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """Search Top-K closest HPO concepts for a given phrase query."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        import numpy as np

        with self.torch.no_grad():
            inputs = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            cls_rep = outputs.last_hidden_state[:, 0, :]
            q_norm = (cls_rep / cls_rep.norm(dim=1, keepdim=True)).cpu().numpy().astype("float32")

        scores, indices = self.index.search(q_norm, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.ids_list[idx], self.phrases_list[idx], float(score)))
        return results
