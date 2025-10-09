import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from scipy import sparse
from scipy.optimize import minimize
import xgboost as xgb
import optuna
import joblib


def load_and_prep(path, text_col="full_text"):

    df = pd.read_csv(path)
    df[text_col] = df[text_col].fillna("")
    df["n_chars"] = df[text_col].apply(len)
    df["n_words"] = df[text_col].apply(lambda t: len(t.split()))
    return df


def stratified_split(df, label="score"):

    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    for tr_idx, rest_idx in sss.split(df, df[label]):
        train = df.iloc[tr_idx].reset_index(drop=True)
        rest = df.iloc[rest_idx].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    for v_rel, te_rel in sss2.split(rest, rest[label]):
        val = rest.iloc[v_rel].reset_index(drop=True)
        test = rest.iloc[te_rel].reset_index(drop=True)
    return train, val, test


def make_features(train, val, test, text_col="full_text", max_features=20000):

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    xtr_t = vec.fit_transform(train[text_col])
    xvl_t = vec.transform(val[text_col])
    xte_t = vec.transform(test[text_col])

    xtr_n = train[["n_chars", "n_words"]].values
    xvl_n = val[["n_chars", "n_words"]].values
    xte_n = test[["n_chars", "n_words"]].values

    xtr = sparse.hstack([xtr_t, sparse.csr_matrix(xtr_n)], format="csr")
    xvl = sparse.hstack([xvl_t, sparse.csr_matrix(xvl_n)], format="csr")
    xte = sparse.hstack([xte_t, sparse.csr_matrix(xte_n)], format="csr")
    return vec, xtr, xvl, xte


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


def trainEval(X_train, y_train, X_val, y_val, n_trials=100, early_stopping_rounds=30):

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_estimators": 2000,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbosity": 0,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        preds_val = model.predict(X_val)
        best_th, _, best_qwk = thresholds(preds_val, y_val)
        return float(best_qwk)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_params["objective"] = "reg:squarederror"
    best_params["random_state"] = 42

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    pred_val = final_model.predict(X_val)
    best_thresholds, pred_labels_val, qwk_val = thresholds(pred_val, y_val)
    mse_val = mean_squared_error(y_val, pred_val)
    joblib.dump(final_model, "fin_mod.joblib")

    return final_model, best_params, best_thresholds, pred_labels_val, qwk_val, mse_val
