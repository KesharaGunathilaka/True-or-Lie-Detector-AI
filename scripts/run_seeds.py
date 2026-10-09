"""Repeat the notebook's training across several seeds.

The single-seed run showed the CNN peaking at epoch 1 then degrading, which
means one number is not a defensible estimate. This repeats the whole
train/evaluate step (varying both the split and the weight init) on the cached
feature matrix, and reports mean +/- std.
"""
import os
import sys
import json
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense, Dropout,
                                     Flatten, MaxPooling1D, SimpleRNN)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)
import pickle

OUT = os.path.join(REPO, "processed_data")
RESULTS = os.path.join(REPO, "results")

X = np.load(os.path.join(OUT, "train_combined_features.npy"))
with open(os.path.join(OUT, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
with open(os.path.join(OUT, "train_paths.pkl"), "rb") as f:
    paths = pickle.load(f)

y = np.array([0 if os.sep + "lie" + os.sep in p else 1 for p in paths])
Y = to_categorical(y, 2)
names = [str(c) for c in le.classes_]
print("features", X.shape, "| class counts", np.bincount(y).tolist(), names, flush=True)


def build_cnn(shape):
    m = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=shape), BatchNormalization(),
        MaxPooling1D(2), Dropout(0.3),
        Conv1D(128, 3, activation="relu"), BatchNormalization(),
        MaxPooling1D(2), Dropout(0.3),
        Flatten(), Dense(128, activation="relu"), BatchNormalization(), Dropout(0.5),
        Dense(2, activation="softmax")])
    m.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def build_rnn(shape):
    m = Sequential([
        SimpleRNN(128, return_sequences=True, input_shape=shape),
        BatchNormalization(), Dropout(0.3),
        SimpleRNN(64), BatchNormalization(), Dropout(0.3),
        Dense(128, activation="relu"), BatchNormalization(), Dropout(0.5),
        Dense(2, activation="softmax")])
    m.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return m


SEEDS = [42, 1, 2, 3, 4]
runs = {"CNN": [], "RNN": []}
T0 = time.time()

for seed in SEEDS:
    Xa, Xb, Ya, Yb = train_test_split(
        X.reshape(-1, X.shape[1], 1), Y, test_size=0.15,
        random_state=seed, stratify=Y)

    for name, builder in (("CNN", build_cnn), ("RNN", build_rnn)):
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = builder((Xa.shape[1], 1))
        hist = model.fit(
            Xa, Ya, validation_data=(Xb, Yb), epochs=50, batch_size=32, verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
            ])
        probs = model.predict(Xb, verbose=0)
        pred, true = np.argmax(probs, 1), np.argmax(Yb, 1)

        r = {
            "seed": seed,
            "accuracy": float(accuracy_score(true, pred)),
            "roc_auc": float(roc_auc_score(Yb, probs, multi_class="ovr")),
            "f1_weighted": float(f1_score(true, pred, average="weighted", zero_division=0)),
            "precision_weighted": float(precision_score(true, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(true, pred, average="weighted", zero_division=0)),
            "recall_lie": float(recall_score(true, pred, pos_label=0, zero_division=0)),
            "recall_truth": float(recall_score(true, pred, pos_label=1, zero_division=0)),
            "confusion": confusion_matrix(true, pred).tolist(),
            "epochs": len(hist.history["loss"]),
            "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
        }
        runs[name].append(r)
        print("%s seed=%-3d acc=%.4f auc=%.4f f1=%.4f epochs=%-3d best_ep=%-3d cm=%s"
              % (name, seed, r["accuracy"], r["roc_auc"], r["f1_weighted"],
                 r["epochs"], r["best_epoch"], r["confusion"]), flush=True)

summary = {"seeds": SEEDS, "n_eval": int(len(Y) * 0.15) + 1, "class_names": names, "runs": runs,
           "summary": {}}

print("\n" + "=" * 72, flush=True)
for name in ("CNN", "RNN"):
    rs = runs[name]
    s = {}
    for k in ("accuracy", "roc_auc", "f1_weighted", "precision_weighted",
              "recall_weighted", "recall_lie", "recall_truth"):
        v = np.array([r[k] for r in rs])
        s[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}
    s["epochs_mean"] = float(np.mean([r["epochs"] for r in rs]))
    summary["summary"][name] = s
    print("%s over %d seeds:" % (name, len(rs)))
    print("   accuracy  %.3f +/- %.3f   (range %.3f - %.3f)"
          % (s["accuracy"]["mean"], s["accuracy"]["std"], s["accuracy"]["min"], s["accuracy"]["max"]))
    print("   ROC AUC   %.3f +/- %.3f   (range %.3f - %.3f)"
          % (s["roc_auc"]["mean"], s["roc_auc"]["std"], s["roc_auc"]["min"], s["roc_auc"]["max"]))
    print("   F1 (wtd)  %.3f +/- %.3f" % (s["f1_weighted"]["mean"], s["f1_weighted"]["std"]))
    print("   recall lie   %.3f +/- %.3f" % (s["recall_lie"]["mean"], s["recall_lie"]["std"]))
    print("   recall truth %.3f +/- %.3f" % (s["recall_truth"]["mean"], s["recall_truth"]["std"]))
    print("   mean epochs  %.1f" % s["epochs_mean"])
    print("", flush=True)

with open(os.path.join(RESULTS, "seed_variance.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("TOTAL %.0fs -> results/seed_variance.json" % (time.time() - T0), flush=True)
