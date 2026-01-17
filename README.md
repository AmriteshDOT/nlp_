
# Automated Essay Scoring & Feedback

End-to-end pipeline that adapts DistilBERT to essay text (continued MLM pretraining → fine-tuned regression) and a GenAI RAG layer that retrieves high-scoring essays to generate personalized feedback. Validation QWK = 0.79.


## What the project does ?

- Continue masked-language (MLM) pretraining of DistilBERT on domain essays to adapt the model to essay-style language.
- Fine-tune a regression head on top of DistilBERT to predict essay scores (score → grade mapping).
- Architected a stateful, agentic RAG workflow using LangGraph and Gemini 2.5 Flash that employs ChromaDB with dynamic metadata filtering to retrieve high-scoring exemplars and generates iterative, 5W1H-structured feedback via a Self-Correction (Coach-Critic) loop.
## Components & Flow

- **Preprocessing**: normalize text, replace newlines with [BR], split train/val and write plain text for MLM.
- **MLM (continued pretraining)**: distilbert (AutoModelForMaskedLM) on domain essays; save checkpoint + tokenizer.
- **Regressor (fine-tune)**: attach a linear head to DistilBERT, train with MSE/AdamW, evaluate with QWK (0.79 val).
- **RAG layer**: Agentic RAG Architecture: Orchestrated via LangGraph, utilizing a StateGraph to manage a multi-node workflow including automated Diagnosis, Score-Aware Retrieval, and a Coach-Critic reflection loop for iterative feedback refinement.
- **Intelligent Retrieval & Generation**: Features dynamic metadata filtering in ChromaDB to fetch high-scoring exemplars relative to user performance, paired with Gemini 2.5 Flash for comparative analysis based on the 5W1H framework.
- **Tech Stack**: Developed using LangChain, LangGraph, and Pydantic for structured validation, powered by Qwen3-0.6B embeddings and Google Generative AI models.