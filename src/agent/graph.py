import sys
import torch
import numpy as np
import os
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from transformers import DistilBertTokenizer

sys.path.append(".")
from src.models.scorer import EssayScorer
from src.agent.state import EssayState
from src.kb.retriever import HybridRetriever


class AESAgent:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = EssayScorer("artifacts/mlm_model").to(self.dev)
        self.model.load_state_dict(
            torch.load("artifacts/final_scorer.pt", map_location=self.dev)
        )
        self.model.eval()

        self.tok = DistilBertTokenizer.from_pretrained("artifacts/mlm_model")
        self.coef = np.load("artifacts/thresholds.npy")

        hf_llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            max_new_tokens=400,
            temperature=0.2,
        )
        self.llm = ChatHuggingFace(llm=hf_llm)
        self.retriever = HybridRetriever()

    def score_node(self, state: EssayState):
        enc = self.tok(
            state["essay"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            pred = self.model(
                enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
            ).item()

        score = float(np.clip(np.digitize(pred, self.coef) + 1, 1, 10))
        return {"score": score, "revision_count": 0, "eval_comments": ""}

    def retrieve_node(self, state: EssayState):
        s = state["score"]
        r_docs = self.retriever.retrieve(
            state["essay"],
            n_results=1,
            where={"$and": [{"type": "rubric"}, {"lower_score": s}]},
        )
        a_docs = self.retriever.retrieve(
            state["essay"],
            n_results=1,
            where={"$and": [{"type": "anchor"}, {"score": s}]},
        )
        s_docs = self.retriever.retrieve(
            state["essay"], n_results=1, where={"type": "source"}
        )

        r_txt = r_docs[0] if r_docs else "Max score reached."
        a_txt = a_docs[0] if a_docs else "No anchor found."
        s_txt = s_docs[0] if s_docs else "No source found."

        return {
            "context": f"Rubric:\n{r_txt}\n\nAnchor:\n{a_txt}\n\nBaseline:\n{s_txt}"
        }

    def critic_node(self, state: EssayState):
        prompt = (
            f"Essay: {state['essay']}\nScore: {state['score']}/10\nContext:\n{state['context']}\n"
            f"Evaluator Feedback (Fix these issues): {state.get('eval_comments', 'None')}\n\n"
            f"Write exactly 3 actionable bullet points explaining the score and how to improve."
        )
        res = self.llm.invoke(prompt)
        return {
            "feedback": res.content,
            "revision_count": state.get("revision_count", 0) + 1,
        }

    def eval_node(self, state: EssayState):
        prompt = (
            f"Feedback provided: {state['feedback']}\n"
            f"Does this feedback consist of exactly 3 actionable bullet points? "
            f"Respond with YES or NO, followed by a brief explanation of what is wrong if NO."
        )
        res = self.llm.invoke(prompt)
        return {"eval_comments": res.content}

    def route_evaluation(self, state: EssayState):
        if state["revision_count"] >= 3:
            return END
        if state["eval_comments"].strip().upper().startswith("YES"):
            return END
        return "critic"

    def compile(self):
        g = StateGraph(EssayState)

        g.add_node("scorer", self.score_node)
        g.add_node("retriever", self.retrieve_node)
        g.add_node("critic", self.critic_node)
        g.add_node("evaluator", self.eval_node)

        g.add_edge(START, "scorer")
        g.add_edge("scorer", "retriever")
        g.add_edge("retriever", "critic")
        g.add_edge("critic", "evaluator")
        g.add_conditional_edges("evaluator", self.route_evaluation)

        return g.compile()


if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "************"
    agent = AESAgent().compile()

    test_essay = "The industrial revolution changed everything. Factories were built and people moved to cities."
    res = agent.invoke(
        {
            "essay": test_essay,
            "score": 0.0,
            "context": "",
            "feedback": "",
            "eval_comments": "",
            "revision_count": 0,
        }
    )

    print(f"Final Score: {res['score']}/10")
    print(f"Total Revisions: {res['revision_count']}")
    print(f"Feedback:\n{res['feedback']}")
