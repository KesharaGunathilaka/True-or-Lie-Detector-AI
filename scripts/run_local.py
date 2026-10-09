"""Run the full Truth-or-Lie pipeline locally and capture real metrics.

Mirrors the notebook exactly: same 30-frame sampling, same frozen MobileNetV2,
same 173-d librosa descriptor, same mean-pooling, same scalers, same RANDOM
15% stratified split (seed 42), same two heads and hyperparameters.

The one deviation is mechanical, not mathematical: frames are streamed straight
into MobileNetV2 in chunks rather than materialised as an 11 GB
train_frames.npy, because this machine has 7 GB free. Identical inputs reach
the encoder either way.
"""
import os
import sys
import json
import time
import pickle
import random
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.preprocessing import (
    extract_frames,
    extract_audio_features,
    NUM_FRAMES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUT = os.path.join(REPO, "processed_data")
RESULTS = os.path.join(REPO, "results")
DATA = os.path.join(REPO, "data", "TruthOrLie")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

CHUNK = 16
AUDIO_WORKERS = 10
FRAME_WORKERS = 8


def log(*a):
    print(*a, flush=True)


def collect(split):
    paths, labels = [], []
    for cls in ("lie", "truth"):
        d = os.path.join(DATA, split, cls)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if n.lower().endswith((".mp4", ".avi", ".mov")):
                paths.append(os.path.join(d, n))
                labels.append(cls)
    return paths, labels


def audio_pass(paths, tag):
    log("")
    log("[audio] %s: %d clips, %d workers" % (tag, len(paths), AUDIO_WORKERS))
    t0 = time.time()
    feats = [None] * len(paths)

    def one(i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return i, extract_audio_features(paths[i], output_dir=OUT)

    with ThreadPoolExecutor(max_workers=AUDIO_WORKERS) as ex:
        for i, f in tqdm(ex.map(one, range(len(paths))), total=len(paths), ncols=78):
            feats[i] = f

    arr = np.asarray(feats, dtype=np.float32)
    silent = int((np.abs(arr).sum(axis=1) == 0).sum())
    log("[audio] %s done in %.0fs  shape=%s  all-zero rows=%d"
        % (tag, time.time() - t0, arr.shape, silent))
    return arr


def visual_pass(paths, tag, encoder, preprocess_input):
    log("")
    log("[visual] %s: %d clips -> MobileNetV2 (streamed, chunk=%d)" % (tag, len(paths), CHUNK))
    t0 = time.time()
    out = np.zeros((len(paths), NUM_FRAMES, 1280), dtype=np.float32)
    failed = []

    with ThreadPoolExecutor(max_workers=FRAME_WORKERS) as ex:
        for start in tqdm(range(0, len(paths), CHUNK), ncols=78):
            batch_paths = paths[start:start + CHUNK]
            frames = list(ex.map(extract_frames, batch_paths))

            stack, keep = [], []
            for j, f in enumerate(frames):
                if f is None:
                    failed.append(batch_paths[j])
                    continue
                stack.append(f)
                keep.append(start + j)
            if not stack:
                continue

            batch = np.stack(stack)
            n = batch.shape[0]
            flat = batch.reshape(-1, FRAME_HEIGHT, FRAME_WIDTH, 3).astype(np.float32)
            emb = encoder.predict(preprocess_input(flat), verbose=0, batch_size=64)
            emb = emb.reshape(n, NUM_FRAMES, -1)
            for k, idx in enumerate(keep):
                out[idx] = emb[k]
            del batch, flat, emb

    log("[visual] %s done in %.0fs  shape=%s  unreadable=%d"
        % (tag, time.time() - t0, out.shape, len(failed)))
    return out, failed


def main():
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                         Dropout, Flatten, MaxPooling1D, SimpleRNN)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score,
                                 accuracy_score)

    tf.random.set_seed(SEED)
    gpus = tf.config.list_physical_devices("GPU")
    log("TensorFlow %s | GPUs: %s" % (tf.__version__, gpus if gpus else "none (CPU)"))

    train_paths, train_cls = collect("train")
    test_paths, test_cls = collect("test")
    log("")
    log("Dataset: %d train clips, %d test clips" % (len(train_paths), len(test_paths)))

    le = LabelEncoder().fit(["lie", "truth"])
    log("Label mapping: %s" % {c: int(i) for i, c in enumerate(le.classes_)})
    y_train = le.transform(train_cls)
    y_test = le.transform(test_cls)

    T0 = time.time()

    a_train = audio_pass(train_paths, "train")
    a_test = audio_pass(test_paths, "test")

    encoder = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                          input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3))
    encoder.trainable = False
    log("")
    log("MobileNetV2 loaded: %s params (frozen)" % format(encoder.count_params(), ","))

    d_train, fail_tr = visual_pass(train_paths, "train", encoder, preprocess_input)
    d_test, fail_te = visual_pass(test_paths, "test", encoder, preprocess_input)

    # --- fuse exactly as combine_features() does --------------------------
    dtr = d_train.mean(axis=1)
    dte = d_test.mean(axis=1)

    deep_scaler = StandardScaler().fit(dtr)
    audio_scaler = StandardScaler().fit(a_train)
    Xtr = np.hstack([deep_scaler.transform(dtr), audio_scaler.transform(a_train)])
    Xte = np.hstack([deep_scaler.transform(dte), audio_scaler.transform(a_test)])
    log("")
    log("Fused feature matrix: train %s, test %s" % (Xtr.shape, Xte.shape))

    Ytr = to_categorical(y_train, 2)
    Yte = to_categorical(y_test, 2)

    np.save(os.path.join(OUT, "train_combined_features.npy"), Xtr)
    np.save(os.path.join(OUT, "test_combined_features.npy"), Xte)
    np.save(os.path.join(OUT, "train_deep_sequence.npy"), d_train)
    with open(os.path.join(OUT, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(OUT, "train_paths.pkl"), "wb") as f:
        pickle.dump(train_paths, f)

    # --- the notebook's split: shipped test set is 8 clips (<20) ----------
    log("")
    log("Shipped test set has %d clips (<20) -> notebook falls back to a random "
        "stratified 15%% validation split, seed %d." % (Xte.shape[0], SEED))
    Xa, Xb, Ya, Yb = train_test_split(
        Xtr.reshape(-1, Xtr.shape[1], 1), Ytr,
        test_size=0.15, random_state=SEED, stratify=Ytr)
    log("  fit on %d, evaluate on %d" % (Xa.shape[0], Xb.shape[0]))
    log("  eval class counts: %s"
        % dict(zip(le.classes_, np.bincount(np.argmax(Yb, 1)).tolist())))

    class_names = list(le.classes_)
    results = {}

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

    for name, builder in (("CNN", build_cnn), ("RNN", build_rnn)):
        log("")
        log("=" * 70)
        log("Training %s" % name)
        log("=" * 70)
        t0 = time.time()
        model = builder((Xa.shape[1], 1))
        hist = model.fit(
            Xa, Ya, validation_data=(Xb, Yb), epochs=50, batch_size=32, verbose=2,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
            ])

        probs = model.predict(Xb, verbose=0)
        pred = np.argmax(probs, 1)
        true = np.argmax(Yb, 1)

        rep = classification_report(true, pred, target_names=class_names,
                                    output_dict=True, zero_division=0)
        r = {
            "accuracy": float(accuracy_score(true, pred)),
            "roc_auc": float(roc_auc_score(Yb, probs, multi_class="ovr")),
            "precision_weighted": float(precision_score(true, pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(true, pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(true, pred, average="weighted", zero_division=0)),
            "per_class": {c: {k: float(v) for k, v in rep[c].items()} for c in class_names},
            "confusion_matrix": confusion_matrix(true, pred).tolist(),
            "epochs_run": len(hist.history["loss"]),
            "best_val_acc": float(max(hist.history["val_accuracy"])),
            "train_seconds": round(time.time() - t0, 1),
        }
        results[name] = r

        log("")
        log("%s RESULTS" % name)
        log(classification_report(true, pred, target_names=class_names, zero_division=0))
        log("  accuracy=%.4f  roc_auc=%.4f  f1=%.4f"
            % (r["accuracy"], r["roc_auc"], r["f1_weighted"]))
        log("  confusion (rows=true %s): %s" % (class_names, r["confusion_matrix"]))
        log("  epochs=%d  time=%ss" % (r["epochs_run"], r["train_seconds"]))

        np.save(os.path.join(RESULTS, "%s_history.npy" % name.lower()),
                np.array([hist.history["accuracy"], hist.history["val_accuracy"],
                          hist.history["loss"], hist.history["val_loss"]]))

    meta = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": "local CPU (TensorFlow has no native-Windows GPU support)",
        "split": "random stratified 15%% validation, seed %d (shipped test set = %d clips, unused)"
                 % (SEED, Xte.shape[0]),
        "n_train_clips": len(train_paths),
        "n_test_clips": len(test_paths),
        "fit_n": int(Xa.shape[0]),
        "eval_n": int(Xb.shape[0]),
        "label_mapping": {c: int(i) for i, c in enumerate(le.classes_)},
        "feature_dim": int(Xtr.shape[1]),
        "unreadable_clips": len(fail_tr) + len(fail_te),
        "silent_audio_train": int((np.abs(a_train).sum(axis=1) == 0).sum()),
        "total_seconds": round(time.time() - T0, 1),
        "results": results,
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log("")
    log("=" * 70)
    log("TOTAL %ss -> results/metrics.json" % meta["total_seconds"])
    log("=" * 70)


if __name__ == "__main__":
    main()
