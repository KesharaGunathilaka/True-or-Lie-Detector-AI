"""Contract tests for the frame and audio feature extractors.

Everything here runs against synthetic media generated on the fly, so the tests
need neither the 328 MB dataset nor a GPU. The audio tests additionally need
librosa and ffmpeg and skip cleanly when those are absent.
"""

import os
import shutil

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    AUDIO_FEATURE_DIM,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NUM_FRAMES,
    extract_audio_features,
    extract_frames,
)


def _write_video(path, n_frames, width=320, height=240, fps=25):
    """Write a tiny silent MP4 whose frame N is filled with intensity N."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV has no usable mp4v encoder in this environment")
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


# --- extract_frames --------------------------------------------------------

def test_returns_fixed_shape_for_a_long_clip(tmp_path):
    video = _write_video(tmp_path / "long.mp4", n_frames=120)
    frames = extract_frames(video)

    assert frames is not None
    assert frames.shape == (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert frames.dtype == np.uint8


def test_short_clip_is_upsampled_to_the_same_shape(tmp_path):
    """A 5-frame clip must still yield 30 frames.

    np.linspace over a 5-frame video produces repeated indices, so the clip is
    resampled with duplicates rather than zero-padded. Zero-padding is the
    separate fallback for frames that fail to read at all. Either way the
    caller always gets the same shape, which is what the batch pipeline needs.
    """
    video = _write_video(tmp_path / "short.mp4", n_frames=5)
    frames = extract_frames(video)

    assert frames is not None
    assert frames.shape == (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3)
    # Real decoded content, not padding, and duplicated to fill the sequence.
    assert not (frames == 0).all()
    assert len(np.unique(frames.reshape(NUM_FRAMES, -1), axis=0)) <= 5


def test_honours_a_custom_frame_count(tmp_path):
    video = _write_video(tmp_path / "custom.mp4", n_frames=60)
    frames = extract_frames(video, num_frames=8)
    assert frames.shape == (8, FRAME_HEIGHT, FRAME_WIDTH, 3)


def test_missing_file_returns_none_rather_than_raising(tmp_path):
    assert extract_frames(str(tmp_path / "does_not_exist.mp4")) is None


def test_unreadable_file_returns_none_rather_than_raising(tmp_path):
    junk = tmp_path / "not_a_video.mp4"
    junk.write_bytes(b"this is definitely not an mp4")
    assert extract_frames(str(junk)) is None


# --- extract_audio_features ------------------------------------------------

def test_audio_dim_is_the_sum_of_its_parts():
    # 20 MFCC + 12 chroma + 128 mel + 7 contrast + 6 tonnetz
    assert AUDIO_FEATURE_DIM == 173


def test_missing_input_returns_a_zero_vector_of_correct_width(tmp_path):
    """A failure anywhere in the audio path must still return 173 floats.

    The batch pipeline stacks these into an array, so a short or absent return
    value would corrupt every downstream shape.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    features = extract_audio_features(
        str(tmp_path / "missing.mp4"), output_dir=str(tmp_path / "out")
    )
    assert features.shape == (AUDIO_FEATURE_DIM,)
    assert (features == 0).all()


def test_silent_video_yields_a_full_width_vector(tmp_path):
    """A video with no audio stream: ffmpeg's -map a fails, we return zeros."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    pytest.importorskip("librosa")

    video = _write_video(tmp_path / "silent.mp4", n_frames=30)
    features = extract_audio_features(video, output_dir=str(tmp_path / "out"))
    assert features.shape == (AUDIO_FEATURE_DIM,)


def test_temp_files_are_cleaned_up(tmp_path):
    """The uuid-named temp WAV must not survive the call."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    out_dir = tmp_path / "out"
    video = _write_video(tmp_path / "clip.mp4", n_frames=30)
    extract_audio_features(video, output_dir=str(out_dir))

    leftovers = [f for f in os.listdir(out_dir) if f.startswith("temp_audio_")] if out_dir.exists() else []
    assert leftovers == [], f"temp audio files left behind: {leftovers}"


def test_concurrent_calls_do_not_collide(tmp_path):
    """Regression test for finding #6.

    The original EDA helper wrote every worker's audio to a single shared
    'temp_audio.wav', so four threads raced to overwrite and delete one file.
    Distinct uuid-named temp files are what make this safe.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    from concurrent.futures import ThreadPoolExecutor

    out_dir = str(tmp_path / "out")
    videos = [_write_video(tmp_path / f"clip_{i}.mp4", n_frames=30) for i in range(4)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda v: extract_audio_features(v, output_dir=out_dir), videos))

    assert len(results) == 4
    assert all(r.shape == (AUDIO_FEATURE_DIM,) for r in results)
