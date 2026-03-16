"""
This module contains some utility functions for loading and processing videos in the dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def get_all_frames(video_root: Path) -> list[Path]:
    """
    Get all frames from the video root directory.

    Args:
        video_root: Path to the video directory.
    
    Returns:
        A list of ordered paths to the single video frames.
    """
    frames_root = video_root / "frames"
    return sorted(frames_root.glob("*.jpg"))


def get_all_cropped_frames(video_root: Path) -> list[Path]:
    """
    Get all cropped frames from the video root directory.
    The face has been detected with MediaPipe.
    Note that not all frames may be available (e.g. if no face was detected).

    Args:
        video_root: Path to the video directory.

    Returns:
        A list of ordered paths to the cropped video frames.
    """
    cropped_frames_root = video_root / "face_detection"
    return sorted(cropped_frames_root.glob("*.png"))


def get_ofiq_scores(video_root: Path) -> pd.DataFrame:
    """
    Get OFIQ quality scores for the video.
    Note that not all frames may be available (e.g. if no face was detected).

    Args:
        video_root: Path to the video directory.
    
    Returns:
        A Pandas DataFrame containing the OFIQ quality scores for the video frames, indexed by frame name (without extension).
    """
    ofiq_scores_root = video_root / "ofiq"
    ofiq_scores_files = sorted(ofiq_scores_root.glob("*.txt"))
    ofiq_scores = {}
    for ofiq_score_file in ofiq_scores_files:
        frame_name = ofiq_score_file.stem.replace("-OFIQ", "")
        df = pd.read_csv(ofiq_score_file, sep=",", index_col=0)
        frame_data = {}
        for measure, status, scalar, raw in df.itertuples():
            if status != "SUCCESS":
                continue
            frame_data[f"{measure}.scalar"] = scalar
            frame_data[f"{measure}.raw"] = raw
        ofiq_scores[frame_name] = frame_data
    return pd.DataFrame.from_dict(ofiq_scores, orient="index")


def get_mediapipe_annotations(video_root: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    Get MediaPipe face landmarks for the video.
    Note that not all frames may be available (e.g. if no face was detected).

    Args:
        video_root: Path to the video directory.
    
    Returns:
        A tuple containing:
        - A NumPy array of shape (num_frames, 478, 3) containing the 478 MediaPipe face landmarks for the video frames.
        - A dictionary mapping each blendshape name to a NumPy array of shape (num_frames,) containing the blendshape coefficients for each frame.
        - A NumPy array of shape (num_frames, 4, 4) containing the head pose transformation matrices for each frame.
    """
    mediapipe_root = video_root / "landmarks" / "landmarks_data"
    mediapipe_files = sorted(mediapipe_root.glob("*.txt"))
    all_landmarks = []
    blendshapes = {}
    head_poses = []
    for mediapipe_file in mediapipe_files:
        lines = mediapipe_file.read_text().splitlines()
        landmarks_data = lines[:478]
        blendshapes_data = lines[478:-4]
        head_pose_data = lines[-4:]
        landmarks = np.array([[float(coord) for coord in landmark.split(",") if coord != ""] for landmark in landmarks_data])
        all_landmarks.append(landmarks)
        for blendshape in blendshapes_data:
            name, value = blendshape.split(": ")
            blendshapes.setdefault(name, []).append(float(value))
        head_pose = np.array([[float(value) for value in line.split(",") if value != ""] for line in head_pose_data])
        head_poses.append(head_pose)
    all_landmarks = np.stack(all_landmarks)
    blendshapes = {name: np.array(values) for name, values in blendshapes.items()}
    head_poses = np.array(head_poses)
    return all_landmarks, blendshapes, head_poses
