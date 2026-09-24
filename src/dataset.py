import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

class EssayDataset(Dataset):
    def __init__(self, csv_file, max_len=512, new_tokens=None):
        self.df = pd.read_csv(csv_file)
        self.tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
        if new_tokens:
            added = self.tok.add_tokens(new_tokens)
            print(f"Added {added} tokens to vocab.")
            
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["essay"])
        score = self.df.iloc[idx]["essay_score"]

        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(score, dtype=torch.float32)
        }

if __name__ == "__main__":
    from vocab import build_oov_vocab
    
    csv_path = "../data/train.csv"
    
    vocab_list = build_oov_vocab(csv_path, top_n=500)
    ds = EssayDataset(csv_path, new_tokens=vocab_list)
    samp = ds[0]

    print("ids:", samp["input_ids"].shape)
    print("vocab size:", len(ds.tok))