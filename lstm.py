# lstm_qwk.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Input,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize


def thresholds(preds, y_true, initial=[1.5, 2.5, 3.5, 4.5, 5.5], method="Powell"):
    preds = np.asarray(preds)
    y_true = np.asarray(y_true)

    def qwk_loss(th):
        th = np.sort(th)
        labels = np.digitize(preds, th) + 1
        return -cohen_kappa_score(y_true, labels, weights="quadratic")

    res = minimize(qwk_loss, initial, method=method)
    best_th = np.sort(res.x)
    labels = np.digitize(preds, best_th) + 1
    qwk_val = cohen_kappa_score(y_true, labels, weights="quadratic")
    return best_th, labels, qwk_val


def train_bilstm_regressor(
    texts_train,
    y_train,
    texts_val,
    y_val,
    max_words=20000,
    max_len=200,
    embed_dim=100,
    glove_path="glove.6B.100d.txt",
    epochs=10,
    batch_size=32,
):
    # Tokenize
    tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tok.fit_on_texts(texts_train)
    seq_train = tok.texts_to_sequences(texts_train)
    seq_val = tok.texts_to_sequences(texts_val)
    X_train = pad_sequences(
        seq_train, maxlen=max_len, padding="post", truncating="post"
    )
    X_val = pad_sequences(seq_val, maxlen=max_len, padding="post", truncating="post")

    vocab_size = min(max_words, len(tok.word_index) + 1)  ## OOV -> +1

    emb_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vect = np.asarray(parts[1:], dtype="float32")
            emb_index[word] = vect
    embed_dim = len(next(iter(emb_index.values())))
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    for word, i in tok.word_index.items():
        if i >= vocab_size:
            continue
        vec = emb_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False,
    )

    # model
    inp = Input(shape=(max_len,), dtype="int32")
    x = embedding_layer(inp)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(1, activation="linear")(x)  # regression output
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    es = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )
    model.fit(
        X_train,
        np.asarray(y_train, dtype="float32"),
        validation_data=(X_val, np.asarray(y_val, dtype="float32")),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=2,
    )

    preds_val = model.predict(X_val).ravel()
    best_th, pred_labels_val, qwk_val = thresholds(preds_val, y_val)

    return {
        "model": model,
        "tokenizer": tok,
        "best_thresholds": best_th,
        "preds_val": preds_val,
        "pred_labels_val": pred_labels_val,
        "qwk_val": qwk_val,
    }
