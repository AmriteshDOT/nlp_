import re
from collections import Counter
import pandas as pd
from transformers import DistilBertTokenizer


def build_oov_vocab(csv_path, top_n=500):
    
    df = pd.read_csv(csv_path)
    tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    raw_text = " ".join(df["essay"].astype(str).tolist()).lower()
    words = re.findall(r"\b[a-z]+\b", raw_text)

    cnt = Counter(words)
    oov_tokens = []

    # Frequent ones
    for w, _ in cnt.most_common():
        # Multiple pieces = OOV
        if len(tok.tokenize(w)) > 1:
            oov_tokens.append(w)

        if len(oov_tokens) >= top_n:
            break

    return oov_tokens


if __name__ == "__main__":
    
    csv = "../data/train.csv"
    new_vocab = build_oov_vocab(csv, top_n=100)
    print("Found OOV tokens:", len(new_vocab))
    print("Top 10:", new_vocab[:10])
