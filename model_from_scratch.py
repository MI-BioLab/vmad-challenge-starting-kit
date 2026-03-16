"""
Sample script that wraps the submission for the challenge.

/!\ Rename this to model.py if you want to use it as your submission!
"""

from pathlib import Path

from utils import get_all_frames, get_all_cropped_frames, get_ofiq_scores, get_mediapipe_annotations


class Model:
    def __init__(self) -> None:
        """
        Initialize your model here. You can load pre-trained weights, set up any necessary data structures, etc.
        This method will be called once when the program starts, so you can perform any expensive setup operations here.
        The constructor must not take any arguments.
        """
        pass

    def predict(self, document_image_path: Path, video_root: Path) -> float:
        """
        Predict whether the document image is morphed or not by comparing it with the provided video frames.
        The prediction should be a float in the range [0, 1], where 0 means "bona fide" and 1 means "morphed".

        Args:
            document_image_path: Path to the document image, which may be either bona fide or morphed.
            video_root: Path to the video directory.
        
        Returns:
            A float in the range [0, 1] representing the prediction.
        """
        raise NotImplementedError()
