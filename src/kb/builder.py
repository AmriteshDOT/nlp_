import os
import sys
import numpy as np
import pandas as pd
import chromadb
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint

sys.path.append(".")


class KBBuilder:
    def __init__(self, csv_path, out_dir="artifacts/kb"):
        self.df = pd.read_csv(csv_path)
        self.out_dir = out_dir
        self.emb_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.chroma = chromadb.PersistentClient(path=out_dir)
        self.col = self.chroma.get_or_create_collection(name="essay_kb")

        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            max_new_tokens=300,
            temperature=0.2,
        )

    def _cluster_nearest(self, texts, k, n_near):
        if len(texts) < k:
            return texts
        vecs = self.emb_model.encode(texts)
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(vecs)

        reps = []
        for i, c in enumerate(km.cluster_centers_):
            idx = np.where(km.labels_ == i)[0]
            if len(idx) == 0:
                continue
            c_vecs = vecs[idx]
            dists = np.linalg.norm(c_vecs - c, axis=1)
            closest = idx[np.argsort(dists)[:n_near]]
            reps.extend([texts[j] for j in closest])
        return reps

    def build_k_anchor(self, k_clusters=2, n_near=5):
        docs, metas, ids = [], [], []
        doc_idx = 0

        scores = np.sort(self.df["score"].unique())
        for s in scores:
            sub = self.df[self.df["score"] == s]["essay"].tolist()
            reps = self._cluster_nearest(sub, k=k_clusters, n_near=n_near)

            for txt in reps:
                docs.append(txt)
                metas.append({"type": "anchor", "score": float(s)})
                ids.append(f"anchor_{doc_idx}")
                doc_idx += 1

        self.col.add(documents=docs, metadatas=metas, ids=ids)
        print(f"K_anchor indexed: {len(docs)} exemplars.")

    def build_k_rubric(self):
        docs, metas, ids = [], [], []
        scores = np.sort(self.df["score"].unique().astype(int))
        doc_idx = 0

        for i in range(len(scores) - 1):
            s_low, s_high = scores[i], scores[i + 1]
            e_low = self.df[self.df["score"] == s_low]["essay"].iloc[0][:500]
            e_high = self.df[self.df["score"] == s_high]["essay"].iloc[0][:500]

            p = (
                f"Compare these two essays.\n"
                f"Grade {s_low} sample: {e_low}\n"
                f"Grade {s_high} sample: {e_high}\n"
                f"List 3 decisive criteria distinguishing grade {s_high} from grade {s_low}."
            )
            rubric = self.llm.invoke(p)

            docs.append(rubric)
            metas.append(
                {
                    "type": "rubric",
                    "lower_score": float(s_low),
                    "upper_score": float(s_high),
                }
            )
            ids.append(f"rubric_{doc_idx}")
            doc_idx += 1

        self.col.add(documents=docs, metadatas=metas, ids=ids)
        print(f"K_rubric indexed: {len(docs)} transition boundaries.")

    def build_k_source(self, k_clusters=3, n_near=3):
        max_s = self.df["score"].max()
        top_essays = self.df[self.df["score"] == max_s]["essay"].tolist()
        reps = self._cluster_nearest(top_essays, k=k_clusters, n_near=n_near)

        docs, metas, ids = [], [], []
        doc_idx = 0

        for txt in reps:
            p = (
                f"Analyze this top-scoring essay:\n{txt[:700]}\n"
                f"Extract core factual themes, structural patterns, and domain arguments used."
            )
            facts = self.llm.invoke(p)

            docs.append(facts)
            metas.append({"type": "source", "source_score": float(max_s)})
            ids.append(f"source_{doc_idx}")
            doc_idx += 1

        self.col.add(documents=docs, metadatas=metas, ids=ids)
        print(f"K_source indexed: {len(docs)} factual/domain baselines.")


if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "*********"

    kb = KBBuilder("../../data/train.csv")
    kb.build_k_anchor(k_clusters=2, n_near=5)
    kb.build_k_rubric()
    kb.build_k_source(k_clusters=3, n_near=3)
