import torch
import torch.nn as nn
from transformers import DistilBertModel


class EssayScorer(nn.Module):
    def __init__(self, model_path="artifacts/mlm_model"):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_path)
        self.reg = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        cls_out = out.last_hidden_state[:, 0, :]
        return self.reg(cls_out).squeeze(-1)


if __name__ == "__main__":
    model = EssayScorer(model_path="artifacts/mlm_model")

    d_ids = torch.randint(0, 31022, (2, 512))
    d_mask = torch.ones(2, 512)

    preds = model(d_ids, d_mask)
    print("embed shape:", model.bert.embeddings.word_embeddings.weight.shape)
    print("preds shape:", preds.shape)
