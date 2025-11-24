
# Automated Essay Scoring & Feedback

End-to-end pipeline that adapts DistilBERT to essay text (continued MLM pretraining → fine-tuned regression) and a GenAI RAG layer that retrieves high-scoring essays to generate personalized feedback. Validation QWK = 0.79.


## What the project does ?

- Continue masked-language (MLM) pretraining of DistilBERT on domain essays to adapt the model to essay-style language.
- Fine-tune a regression head on top of DistilBERT to predict essay scores (score → grade mapping).
- Build a GenAI retrieval-augmented-generation (RAG) pipeline using LangChain + Chroma + Gemini embeddings to fetch high-scoring essays and generate contextual, personalized writing feedback.
## Components & Flow

- **Preprocessing**: normalize text, replace newlines with [BR], split train/val and write plain text for MLM.
- **MLM (continued pretraining)**: distilbert (AutoModelForMaskedLM) on domain essays; save checkpoint + tokenizer.
- **Regressor (fine-tune)**: attach a linear head to DistilBERT, train with MSE/AdamW, evaluate with QWK (0.79 val).
- **RAG layer**: build Chroma vectorstore from essays using Gemini embeddings; use dynamic top-K retrieval filtered by predicted score.
- **Generation**: prompt-engineered Gemini (via LangChain) compares user essay with retrieved references and produces personalized feedback.