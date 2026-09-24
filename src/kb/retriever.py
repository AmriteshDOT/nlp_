import re
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self, db_path="artifacts/kb", col_name="essay_kb"):
        self.chroma = chromadb.PersistentClient(path=db_path)
        self.col = self.chroma.get_collection(name=col_name)
        self.emb = SentenceTransformer("all-MiniLM-L6-v2")

        all_data = self.col.get(include=["documents", "metadatas"])
        self.doc_ids = all_data["ids"]
        self.docs = all_data["documents"]
        self.metas = all_data["metadatas"]
        self.id_to_doc = {i: d for i, d in zip(self.doc_ids, self.docs)}

    def _dense_search(self, q_vec, n_res, where_filter=None):
        kwargs = {"query_embeddings": [q_vec], "n_results": n_res}
        if where_filter:
            kwargs["where"] = where_filter
        res = self.col.query(**kwargs)
        return res["ids"][0] if res["ids"] else []

    def _sparse_search(self, q_toks, n_res, where_filter=None):
        cand_indices = []
        for idx, meta in enumerate(self.metas):
            if where_filter:
                match = True
                if "$and" in where_filter:
                    for cond in where_filter["$and"]:
                        for k, v in cond.items():
                            if meta.get(k) != v:
                                match = False
                else:
                    for k, v in where_filter.items():
                        if meta.get(k) != v:
                            match = False
                if not match:
                    continue
            cand_indices.append(idx)

        if not cand_indices:
            return []

        corpus = [tokenize(self.docs[i]) for i in cand_indices]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(q_toks)

        top_sub_idx = np.argsort(scores)[::-1][:n_res]
        return [self.doc_ids[cand_indices[i]] for i in top_sub_idx if scores[i] > 0]

    def _rrf(self, dense_ids, sparse_ids, k=60):
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, doc_id in enumerate(sparse_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked]

    def retrieve(self, query, n_results=1, where=None, candidate_depth=10):
        q_vec = self.emb.encode(query).tolist()
        q_toks = tokenize(query)

        dense_ids = self._dense_search(q_vec, candidate_depth, where_filter=where)
        sparse_ids = self._sparse_search(q_toks, candidate_depth, where_filter=where)

        fused_ids = self._rrf(dense_ids, sparse_ids, k=60)[:n_results]
        return [
            self.id_to_doc[doc_id] for doc_id in fused_ids if doc_id in self.id_to_doc
        ]


if __name__ == "__main__":
    retriever = HybridRetriever()
    out = retriever.retrieve(
        query="industrial revolution machinery factory work",
        n_results=1,
        where={"type": "source"},
    )
    print("Retrieved Document:\n", out[0] if out else "None found")
