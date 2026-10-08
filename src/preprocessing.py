"""Canonical preprocessing helpers for the Truth-or-Lie detector.

These functions were extracted out of ``True_or_Lie_Detector_AI_Project.ipynb``
so that they can be imported and unit-tested without a GPU runtime or the
328 MB video dataset. The notebook imports from this module rather than
redefining the functions inline, so there is exactly one implementation of
each.

``librosa`` and ``sklearn`` are imported lazily inside the functions that need
them. That keeps ``extract_frames`` and the split helpers importable on a bare
environment that only has numpy and OpenCV installed.
"""

from __future__ import annotations

import os
import re
import subprocess
import uuid
import warnings

import cv2
import numpy as np

# --- Frame sampling -------------------------------------------------------

NUM_FRAMES = 30
FRAME_HEIGHT = 224
FRAME_WIDTH = 224

# --- Audio feature layout -------------------------------------------------
# The audio vector is the concatenation of the per-coefficient means of five
# librosa feature families. The widths are fixed by the librosa defaults we
# request, so the total is a compile-time constant rather than a magic number.
N_MFCC = 20
N_CHROMA = 12
N_MEL = 128
N_SPECTRAL_CONTRAST = 7
N_TONNETZ = 6
AUDIO_FEATURE_DIM = N_MFCC + N_CHROMA + N_MEL + N_SPECTRAL_CONTRAST + N_TONNETZ  # 173

SAMPLE_RATE = 22050

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

# Clip filenames follow ``trial_<class>_<trial>_<clip>.mp4``. The trial number
# identifies the source courtroom video (and therefore the person speaking),
# which is what we must not split across train and test.
_CLIP_NAME_RE = re.compile(r"trial_(?P<label>lie|truth)_(?P<trial>\d+)_(?P<clip>\d+)", re.IGNORECASE)


def check_ffmpeg() -> bool:
    """Return True if a working ffmpeg is on PATH.

    Checks the exit status as well as the binary's presence: an ffmpeg that is
    installed but broken returns non-zero and must not pass this check.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except FileNotFoundError:
        print("FFmpeg is not installed. Please install FFmpeg before continuing.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"FFmpeg is installed but returned a non-zero exit status: {exc.returncode}")
        return False


def trial_id(path: str) -> str:
    """Return the source-trial identifier for a clip path, e.g. ``'lie_001'``.

    Every clip cut from the same courtroom video shares a trial id. Grouping on
    this value is what prevents near-duplicate clips of the same speaker from
    landing on both sides of a train/test split.

    Raises:
        ValueError: if the filename does not match the expected clip pattern.
    """
    match = _CLIP_NAME_RE.search(os.path.basename(path))
    if match is None:
        raise ValueError(f"Unrecognised clip filename: {path!r}")
    return f"{match.group('label').lower()}_{match.group('trial')}"


def group_split(paths, labels, test_size=0.15, random_state=42):
    """Split clip indices so that no source trial appears on both sides.

    This replaces the ``train_test_split(..., stratify=labels)`` the notebook
    originally used. A plain random split scatters clips from the same trial
    across train and test, which inflates accuracy dramatically: the model can
    recognise the speaker rather than the deception.

    Args:
        paths: clip file paths (used only to derive trial ids).
        labels: per-clip labels, same length as ``paths``.
        test_size: fraction of *groups* to hold out.
        random_state: seed for reproducibility.

    Returns:
        ``(train_idx, test_idx)`` as numpy integer arrays.
    """
    from sklearn.model_selection import GroupShuffleSplit

    paths = list(paths)
    if len(paths) != len(labels):
        raise ValueError(f"paths and labels differ in length: {len(paths)} vs {len(labels)}")

    groups = np.array([trial_id(p) for p in paths])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.zeros(len(paths)), labels, groups))
    return train_idx, test_idx


def extract_frames(video_path: str, num_frames: int = NUM_FRAMES):
    """Extract ``num_frames`` evenly spaced RGB frames, resized to 224x224.

    Short or partially unreadable clips are zero-padded up to ``num_frames`` so
    that every video yields an identically shaped array and one bad file cannot
    break a batch of 800.

    Returns:
        ``(num_frames, 224, 224, 3)`` uint8 array, or None if the video could
        not be opened or reports no frames.
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"No frames found in video: {video_path}")
            cap.release()
            return None

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        while len(frames) < num_frames:
            frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))

        return np.array(frames)

    except Exception as exc:
        print(f"Error extracting frames from {video_path}: {exc}")
        return None


def extract_audio_features(video_path: str, output_dir: str = "processed_data"):
    """Extract a 173-d audio descriptor from a video's soundtrack.

    Demuxes the audio with ffmpeg into a uniquely named temporary WAV, then
    concatenates the per-coefficient means of five librosa feature families.
    Any failure at any stage returns a zero vector of the correct width so the
    caller always receives a consistently shaped result.

    The temp filename includes a uuid because callers run this across a
    ThreadPoolExecutor; a shared fixed filename would let concurrent workers
    overwrite and delete each other's audio mid-read.
    """
    temp_audio = os.path.join(output_dir, f"temp_audio_{uuid.uuid4().hex}.wav")

    try:
        os.makedirs(output_dir, exist_ok=True)
        # List-args (not a shell string) so paths containing spaces, quotes or
        # shell metacharacters are passed through literally.
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", temp_audio, "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"FFmpeg error for {video_path}: {exc.stderr.decode(errors='replace')}")
        return np.zeros(AUDIO_FEATURE_DIM)
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg.")
        return np.zeros(AUDIO_FEATURE_DIM)

    if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
        print(f"No audio extracted from {video_path}")
        return np.zeros(AUDIO_FEATURE_DIM)

    try:
        import librosa

        y, sr = librosa.load(temp_audio, sr=SAMPLE_RATE)

        # librosa's default n_fft is 2048; pad so very short clips still produce
        # a full spectrogram frame instead of warning and returning empties.
        if len(y) < 1024:
            print(f"Audio too short in {video_path}, padding with zeros")
            y = np.pad(y, (0, 1024 - len(y)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

            if len(y) > 0:
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            else:
                tonnetz = np.zeros((N_TONNETZ, 1))

        features = np.concatenate([
            np.mean(mfccs, axis=1) if mfccs.size > 0 else np.zeros(N_MFCC),
            np.mean(chroma, axis=1) if chroma.size > 0 else np.zeros(N_CHROMA),
            np.mean(mel, axis=1) if mel.size > 0 else np.zeros(N_MEL),
            np.mean(contrast, axis=1) if contrast.size > 0 else np.zeros(N_SPECTRAL_CONTRAST),
            np.mean(tonnetz, axis=1) if tonnetz.size > 0 else np.zeros(N_TONNETZ),
        ])

    except Exception as exc:
        print(f"Error processing audio from {video_path}: {exc}")
        features = np.zeros(AUDIO_FEATURE_DIM)

    finally:
        try:
            os.remove(temp_audio)
        except OSError:
            pass

    return features
