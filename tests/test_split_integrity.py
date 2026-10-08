"""Split-integrity tests.

The dataset is 829 short clips cut from only 109 source courtroom trials
(mean 7.6 clips per trial). Clips from one trial show the same person, in the
same room, saying related things. If any trial has clips on both sides of a
train/test split, the model can score highly by recognising the speaker rather
than by detecting deception.

These tests are the guard rail for that. They need nothing but the standard
library, numpy and scikit-learn -- no TensorFlow, no GPU, no video decoding.
"""

import os
from collections import Counter

import pytest

from src.preprocessing import group_split, trial_id

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data", "TruthOrLie")

# The video dataset is gitignored (328 MB), so these tests skip cleanly on a
# fresh clone rather than failing for the wrong reason.
requires_dataset = pytest.mark.skipif(
    not os.path.isdir(DATA_ROOT),
    reason="Video dataset not present; see README 'Data Setup'.",
)


def _collect_clips(split):
    """Return (paths, labels) for one of the on-disk 'train'/'test' splits."""
    paths, labels = [], []
    for cls in ("lie", "truth"):
        class_dir = os.path.join(DATA_ROOT, split, cls)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if name.lower().endswith(".mp4"):
                paths.append(os.path.join(class_dir, name))
                labels.append(cls)
    return paths, labels


# --- trial_id parsing -----------------------------------------------------

@pytest.mark.parametrize(
    "filename,expected",
    [
        ("trial_lie_001_000.mp4", "lie_001"),
        ("trial_truth_042_017.mp4", "truth_042"),
        ("/some/nested/path/trial_lie_109_003.mp4", "lie_109"),
    ],
)
def test_trial_id_parses_source_trial(filename, expected):
    assert trial_id(filename) == expected


def test_trial_id_rejects_unrecognised_names():
    with pytest.raises(ValueError):
        trial_id("some_other_video.mp4")


def test_clips_from_same_trial_share_an_id():
    """The whole point: different clips, same trial, same group key."""
    ids = {trial_id(f"trial_lie_002_{i:03d}.mp4") for i in range(16)}
    assert ids == {"lie_002"}


# --- group_split guarantees ------------------------------------------------

def test_group_split_keeps_trials_whole():
    # 10 trials x 8 clips. A random split would almost certainly straddle them.
    paths, labels = [], []
    for trial in range(10):
        label = "lie" if trial % 2 == 0 else "truth"
        for clip in range(8):
            paths.append(f"trial_{label}_{trial:03d}_{clip:03d}.mp4")
            labels.append(label)

    train_idx, test_idx = group_split(paths, labels, test_size=0.3, random_state=0)

    train_trials = {trial_id(paths[i]) for i in train_idx}
    test_trials = {trial_id(paths[i]) for i in test_idx}

    assert train_trials, "training split is empty"
    assert test_trials, "test split is empty"
    assert not (train_trials & test_trials), (
        f"trials leaked across the split: {sorted(train_trials & test_trials)}"
    )
    assert len(train_idx) + len(test_idx) == len(paths)


def test_group_split_is_deterministic():
    paths = [f"trial_lie_{t:03d}_{c:03d}.mp4" for t in range(12) for c in range(4)]
    labels = ["lie"] * len(paths)
    a = group_split(paths, labels, random_state=7)
    b = group_split(paths, labels, random_state=7)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_group_split_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        group_split(["trial_lie_001_000.mp4"], ["lie", "truth"])


# --- The real dataset on disk ---------------------------------------------

@requires_dataset
def test_dataset_shape_matches_documented_figures():
    """Guards the numbers quoted in the README against silent dataset drift."""
    train_paths, _ = _collect_clips("train")
    test_paths, _ = _collect_clips("test")

    assert len(train_paths) + len(test_paths) == 829
    all_trials = {trial_id(p) for p in train_paths + test_paths}
    assert len(all_trials) == 109

    per_class = Counter(t.split("_")[0] for t in all_trials)
    assert per_class == {"lie": 59, "truth": 50}


@requires_dataset
def test_group_split_on_the_real_dataset_leaks_nothing():
    paths, labels = _collect_clips("train")
    train_idx, test_idx = group_split(paths, labels, test_size=0.15, random_state=42)

    train_trials = {trial_id(paths[i]) for i in train_idx}
    test_trials = {trial_id(paths[i]) for i in test_idx}
    assert not (train_trials & test_trials)


@requires_dataset
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect in the dataset as shipped: trial lie_001 has clips in "
        "both train/ and test/. Documented as finding #3; kept as an xfail so "
        "the day the directories are re-split correctly, this test tells us."
    ),
)
def test_shipped_directory_split_does_not_share_trials():
    train_paths, _ = _collect_clips("train")
    test_paths, _ = _collect_clips("test")

    shared = {trial_id(p) for p in train_paths} & {trial_id(p) for p in test_paths}
    assert not shared, f"trials present in both train/ and test/: {sorted(shared)}"
