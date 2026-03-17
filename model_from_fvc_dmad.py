"""
Sample script that wraps the submission for the challenge, starting from a D-MAD model that has been submitted to FVC-onGoing.
You can plug here your D-MAD model file (dmad_detect.py), and it will be called with each frame, and the scores will be fused to produce the final prediction for the video.
This may be a good starting point if you already have a D-MAD model and want to adapt it for the challenge.
However, we also encourage you to implement your own model that directly takes as input the document image and the video frames, and produces a single prediction for the video.

/!\ Rename this to model.py if you want to use it as your submission!
"""

from pathlib import Path

import numpy as np

from utils import get_all_frames, get_all_cropped_frames

# You can copy the dmad_detect.py and frvt_common.py files from FVC-onGoing!
from frvt_common import ReturnCode, ImageLabel
from dmad_detect import initialize, detect_morph_differentially


class Model:
    def __init__(self) -> None:
        """
        Initialize your model here. You can load pre-trained weights, set up any necessary data structures, etc.
        This method will be called once when the program starts, so you can perform any expensive setup operations here.
        The constructor must not take any arguments.
        """
        self.use_cropped_frames = False  # Set this to True to use cropped frames instead of original frames for the prediction
        self.score_fusion_strategy = "mean"  # Set this to the desired score fusion strategy ("mean" | "max" | "min")
        initialize()

    def predict_dmad(self, document_image_path: Path, live_image_path: Path) -> float:
        """
        Predict whether the document image is morphed or not by comparing it with the provided trusted live-capture image.
        The prediction should be a float in the range [0, 1], where 0 means "bona fide" and 1 means "morphed".
        You can plug here your D-MAD model and it will be called with each frame, and the scores will be fused to produce the final prediction for the video.

        Args:
            document_image_path: Path to the document image, which may be either bona fide or morphed.
            live_image_path: Path to the trusted live-capture image of the subject.

        Returns:
            A float in the range [0, 1] representing the prediction.
        """
        return_code, _, score = detect_morph_differentially(str(document_image_path), ImageLabel.Unknown, str(live_image_path))
        if return_code != ReturnCode.Success:
            raise RuntimeError(f"D-MAD prediction failed with return code {return_code}")
        return score

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
        if self.use_cropped_frames:
            frames = get_all_cropped_frames(video_root)
        else:
            frames = get_all_frames(video_root)
        scores = []
        for frame in frames:
            try:
                score = self.predict_dmad(document_image_path, frame)
                scores.append(score)
            except RuntimeError as e:
                print(f"Warning: D-MAD prediction failed for frame {frame} with error: {e}")
                continue
        if self.score_fusion_strategy == "mean":
            return np.mean(scores)
        elif self.score_fusion_strategy == "max":
            return np.max(scores)
        elif self.score_fusion_strategy == "min":
            return np.min(scores)
        else:
            raise ValueError("Invalid score fusion strategy")
