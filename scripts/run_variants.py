"""Diagnose and fix the CNN head's immediate divergence.

The original head ends in Flatten -> Dense(128), which is ~5.94M parameters
trained on 697 samples. It memorises within one epoch and validation loss
climbs from epoch 2 onward, so EarlyStopping restores a one-epoch model.

This compares the original against minimal, standard alternatives on the
cached feature matrix, 5 seeds each, so the choice is made on evidence rather
than on which run happened to look best.
"""
import os
import sys
import json
import time
import pickle

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense, Dropout,
                                     Flatten, GlobalAveragePooling1D, Input,
                                     MaxPooling1D, SimpleRNN)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)

OUT = os.path.join(REPO, "processed_data")
RESULTS = os.path.join(REPO, "results")

X = np.load(os.path.join(OUT, "train_combined_features.npy"))
with open(os.path.join(OUT, "train_paths.pkl"), "rb") as f:
    paths = pickle.load(f)
y = np.array([0 if (os.sep + "lie" + os.sep) in p else 1 for p in paths])
Y = to_categorical(y, 2)
SHAPE = (X.shape[1], 1)
print("features", X.shape, "| counts lie/truth", np.bincount(y).tolist(), flush=True)


def cnn_flatten(lr=1e-4):
    return Sequential([
        Input(SHAPE),
        Conv1D(64, 3, activation="relu"), BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Conv1D(128, 3, activation="relu"), BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Flatten(), Dense(128, activation="relu"), BatchNormalization(), Dropout(0.5),
        Dense(2, activation="softmax")]), lr


def cnn_gap(lr=1e-4):
    """Same stack, but global average pooling instead of Flatten."""
    return Sequential([
        Input(SHAPE),
        Conv1D(64, 3, activation="relu"), BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Conv1D(128, 3, activation="relu"), BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        GlobalAveragePooling1D(), Dense(128, activation="relu"), BatchNormalization(), Dropout(0.5),
        Dense(2, activation="softmax")]), lr


def cnn_gap_lr3():
    """GAP head with a higher LR, affordable now that capacity is sane."""
    return cnn_gap(lr=1e-3)[0], 1e-3


def mlp(lr=1e-3):
    """Reference point: does the convolution earn its place at all?"""
    return Sequential([
        Input((X.shape[1],)),
        Dense(256, activation="relu"), BatchNormalization(), Dropout(0.5),
        Dense(64, activation="relu"), BatchNormalization(), Dropout(0.3),
        Dense(2, activation="softmax")]), lr


def rnn(lr=1e-4):
    return Sequential([
        Input(SHAPE),
        SimpleRNN(128, return_sequences=True), BatchNormalization(), Dropout(0.3),
        SimpleRNN(64), BatchNormalization(), Dropout(0.3),
        Dense(128, activation="relu"), BatchNormalization(), Dropout(0.5),
        Dense(2, activation="softmax")]), lr


VARIANTS = [
    ("CNN-flatten (original)", cnn_flatten, False),
    ("CNN-GAP", cnn_gap, False),
    ("CNN-GAP lr1e-3", cnn_gap_lr3, False),
    ("MLP baseline", mlp, True),
    ("RNN (original)", rnn, False),
]
SEEDS = [42, 1, 2, 3, 4]

for name, fn, flat_in in VARIANTS:
    m, _ = fn()
    print("%-24s params=%s" % (name, format(m.count_params(), ",")), flush=True)
    tf.keras.backend.clear_session()
print("", flush=True)

all_res = {}
T0 = time.time()

for name, fn, flat_in in VARIANTS:
    accs, aucs, f1s, rl, rt, eps = [], [], [], [], [], []
    for seed in SEEDS:
        Xin = X if flat_in else X.reshape(-1, X.shape[1], 1)
        Xa, Xb, Ya, Yb = train_test_split(Xin, Y, test_size=0.15,
                                          random_state=seed, stratify=Y)
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model, lr = fn()
        model.compile(optimizer=Adam(lr), loss="categorical_crossentropy",
                      metrics=["accuracy"])
        h = model.fit(Xa, Ya, validation_data=(Xb, Yb), epochs=50, batch_size=32,
                      verbose=0,
                      callbacks=[EarlyStopping(monitor="val_loss", patience=7,
                                               restore_best_weights=True),
                                 ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                   patience=3, verbose=0)])
        p = model.predict(Xb, verbose=0)
        pred, true = np.argmax(p, 1), np.argmax(Yb, 1)
        accs.append(accuracy_score(true, pred))
        aucs.append(roc_auc_score(Yb, p, multi_class="ovr"))
        f1s.append(f1_score(true, pred, average="weighted", zero_division=0))
        rl.append(recall_score(true, pred, pos_label=0, zero_division=0))
        rt.append(recall_score(true, pred, pos_label=1, zero_division=0))
        eps.append(int(np.argmin(h.history["val_loss"])) + 1)

    a, u, f = np.array(accs), np.array(aucs), np.array(f1s)
    all_res[name] = {
        "accuracy_mean": float(a.mean()), "accuracy_std": float(a.std(ddof=1)),
        "accuracy_min": float(a.min()), "accuracy_max": float(a.max()),
        "roc_auc_mean": float(u.mean()), "roc_auc_std": float(u.std(ddof=1)),
        "f1_mean": float(f.mean()), "f1_std": float(f.std(ddof=1)),
        "recall_lie_mean": float(np.mean(rl)), "recall_truth_mean": float(np.mean(rt)),
        "best_epochs": eps,
        "per_seed_accuracy": [float(v) for v in accs],
    }
    print("%-24s acc %.3f+/-%.3f (%.3f-%.3f)  auc %.3f+/-%.3f  f1 %.3f  best_ep %s"
          % (name, a.mean(), a.std(ddof=1), a.min(), a.max(),
             u.mean(), u.std(ddof=1), f.mean(), eps), flush=True)

with open(os.path.join(RESULTS, "variant_comparison.json"), "w") as fh:
    json.dump({"seeds": SEEDS, "variants": all_res}, fh, indent=2)
print("\nTOTAL %.0fs -> results/variant_comparison.json" % (time.time() - T0), flush=True)
