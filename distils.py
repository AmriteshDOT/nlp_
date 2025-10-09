import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from scipy.optimize import minimize
import joblib


def thresholds(preds, y_true, initial=[1.5, 2.5, 3.5, 4.5], method="Powell"):

    preds = np.asarray(preds)
    y_true = np.asarray(y_true)

    def qwk_loss(th):
        th = np.sort(th)
        labels = np.digitize(preds, th) + 1
        return -cohen_kappa_score(y_true, labels, weights="quadratic")

    res = minimize(qwk_loss, initial, method=method)
    best_th = np.sort(res.x)
    pred_labels = np.digitize(preds, best_th) + 1
    qwk_val = cohen_kappa_score(y_true, pred_labels, weights="quadratic")
    return best_th, pred_labels, qwk_val


def simple_ensemble(pred_list):
    return np.mean(np.stack(pred_list, axis=0), axis=0)


def fit_transformer_cv(
    df,
    text_col="full_text",
    label_col="grade",
    model_name="distilbert-base-uncased",
    max_len=128,
    n_splits=3,
    batch_size=4,
    epochs=2,
    lr=2e-5,
    out_dir="./models",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = Dataset.from_pandas(df[[text_col, label_col]].reset_index(drop=True))

    def tok_fn(batch):
        return tokenizer(
            batch[text_col], truncation=True, padding="max_length", max_length=max_len
        )

    ds = ds.map(tok_fn, batched=True)
    ds = ds.remove_columns([text_col])
    ds = ds.rename_column(label_col, "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    base_nm = model_name.split("/")[-1]
    n = len(ds)
    oof_preds = []
    oof_labels = []
    saved_dirs = []
    thres = []
    qwks = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(n))):
        train_ds = ds.select(tr_idx)
        val_ds = ds.select(val_idx)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, problem_type="regression"
        )
        run_dir = os.path.join(out_dir, f"{base_nm}_f{fold}")
        os.makedirs(run_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=run_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            num_train_epochs=epochs,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_mae",
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
        )

        trainer.train()

        preds = trainer.predict(val_ds).predictions.flatten()
        oof_ytrue = np.array(val_ds["labels"])
        th, predictions, qwk = thresholds(preds, oof_ytrue)
        thres.append(th)
        qwks.append(qwk)
        oof_preds.append(predictions)
        oof_labels.append(np.array(val_ds["labels"]))
        trainer.save_model(run_dir)
        saved_dirs.append(run_dir)

    mae = mean_absolute_error(oof_labels, oof_preds)
    mse = mean_squared_error(oof_labels, oof_preds)
    th_array = np.array(thres)
    med_th = np.median(th_array, axis=0)
    joblib.dump(med_th, f"medianthres_oof_{base_nm}.joblib")
    # print(f"{mae} {mse}")
    return oof_preds, oof_labels, saved_dirs


def example_run(df):
    # DistilBERT
    oof_dbert, _, _ = fit_transformer_cv(df, model_name="distilbert-base-uncased")
    # DistilRoBERTa
    oof_drbt, _, _ = fit_transformer_cv(df, model_name="distilroberta-base")
    avg = simple_ensemble([oof_dbert, oof_drbt])
    th_distilBERT = joblib.load(f"medianthres_oof_distilbert-base-uncased.joblib")
    th_distilRoBERTa = joblib.load(f"medianthres_oof_distilroberta-base.joblib")
    ths = []
    ths.append(th_distilBERT)
    ths.append(th_distilRoBERTa)
    th_array = np.array(ths)
    final_th = np.median(th_array, axis=0)
    joblib.dump(final_th, "net_threshold.joblib")


def predict_test_set_simple(texts_test, fold_dirs, model_name, max_len=128):

    final_threshold = joblib.load("net_threshold.joblib")
    all_fold_preds = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(
        texts_test,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    for fold_dir in fold_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(
            fold_dir, num_labels=1, problem_type="regression"
        )
        model.eval()

        preds = model(**inputs).logits.numpy()
        all_fold_preds.append(preds)

    ensemble_preds = np.mean(all_fold_preds, axis=0)
    final_pred_labels = np.digitize(ensemble_preds, final_threshold) + 1
    return final_pred_labels
