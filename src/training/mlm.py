import sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.append(".")
from src.vocab import build_oov_vocab


class MLMDataset(Dataset):
    def __init__(self, csv_file, tok, max_len=512):
        self.df = pd.read_csv(csv_file)
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["essay"])
        enc = self.tok(
            text, truncation=True, padding="max_length", max_length=self.max_len
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
        }


if __name__ == "__main__":
    csv_path = "data/train.csv"

    new_toks = build_oov_vocab(csv_path, top_n=500)
    tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tok.add_tokens(new_toks)

    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    model.resize_token_embeddings(len(tok))

    ds = MLMDataset(csv_path, tok)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=0.15
    )

    for name, param in model.named_parameters():
        if "embeddings" not in name:
            param.requires_grad = False

    args1 = TrainingArguments(
        output_dir="artifacts/mlm_phase1",
        learning_rate=1e-3,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=50,
    )

    t1 = Trainer(model=model, args=args1, train_dataset=ds, data_collator=collator)
    print("Phase 1: Warming up new embeddings...")
    t1.train()

    for param in model.parameters():
        param.requires_grad = True

    args2 = TrainingArguments(
        output_dir="artifacts/mlm_phase2",
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=50,
    )

    t2 = Trainer(model=model, args=args2, train_dataset=ds, data_collator=collator)
    print("Phase 2: Harmonizing full model...")
    t2.train()

    model.save_pretrained("artifacts/mlm_model")
    tok.save_pretrained("artifacts/mlm_model")
    print("Saved to artifacts/mlm_model")
