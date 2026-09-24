
# Automated Essay Scoring & Feedback

End-to-end pipeline coupling a domain-adapted DistilBERT regressor with Nelder-Mead threshold optimization (0.76 QWK), layered with a self-correcting LangGraph RAG agent that uses hybrid search to retrieve grade-specific rubrics and exemplars for targeted feedback.


## What the project does ?

- Continues Masked Language Modeling (MLM) on DistilBERT after expanding the tokenizer with domain-specific Out-of-Vocabulary (OOV) tokens to adapt to essay-style language.
- Replaces standard classification with a continuous regression head, employing a gradient-free Nelder-Mead optimizer to learn non-uniform ordinal rounding thresholds that directly maximize Quadratic Weighted Kappa.
- Architected a stateful, agentic RAG workflow using LangGraph that manages a multi-node workflow including automated Scoring, Hybrid Retrieval, and an Actor-Critic reflection loop for iterative feedback refinement.
- Enforces output quality by dynamically querying a triple-asset ChromaDB knowledge base (K_anchor, K_rubric, K_source) to ground generation in actual grading rubrics and factual baselines, mitigating LLM hallucinations.
## Components & Flow

- **MLM (Continued Pretraining)**: Extracts top domain tokens, expands the DistilBERT vocabulary, and adapts the model via a freeze-and-thaw pretraining schedule.

- **Regressor (Fine-tune)**: Attaches a nn.Linear(768, 1) head, trains with MSE/AdamW, and discretizes raw float predictions via Nelder-Mead boundaries (0.76 val QWK).

- **Hybrid RAG Layer**: Combines dense HNSW vector search (Sentence Transformers) with sparse lexical search (BM25) via Reciprocal Rank Fusion (RRF) to retrieve the most contextually relevant rubrics and exemplar essays.

- **Self-Correcting Agent**: A LangGraph StateGraph routes the essay through a Scorer → Retriever → Critic → Evaluator loop. The Evaluator node strictly audits the Critic's feedback for formatting and rubric adherence, forcing revisions if criteria are not met.

- **Tech Stack**: Developed using PyTorch, SciPy (Optimization), LangGraph, LangChain, ChromaDB, Rank-BM25, and Hugging Face Serverless Inference APIs (Mistral-7B-Instruct & all-MiniLM-L6-v2).