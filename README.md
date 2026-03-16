# Starting Kit for the V-MAD Challenge 2026
This repository contains a starting kit for "Every Frame You Take: The Video-based Morphing Attack Detection Challenge", a challenge held in conjunction with the 2026 IEEE International Joint Conference on Biometrics.

Face morphing attacks represent a serious threat to biometric face recognition systems deployed in real-world operational environments, such as Automated Border Control (ABC) gates. Therefore, it is fundamental to develop accurate systems to counter this attack, i.e. Morphing Attack Detection (MAD) systems.

While most existing Morphing Attack Detection (MAD) approaches focus on single images (S-MAD) or image pairs (D-MAD), operational systems routinely acquire short video sequences, which are largely underexploited by current methods. 

This competition is motivated by the need to bridge this gap between research and practice by fostering the development of Video-based Morphing Attack Detection (V-MAD) approaches that explicitly leverage temporal information, multiple frames, and varying image quality.

The goal of the competition is to advance the state of the art in morphing attack detection by encouraging methods that can effectively operate on video data and improve the security and reliability of face recognition technologies in operational settings.

Notably, the competition introduces a new task definition (V-MAD) and new benchmarks, not yet available in the literature. Further details on V-MAD can be found in the original paper.

All participating teams will be required to submit their solution on the free and open-source Codabench platform, which supports the upload of models, their evaluation, and the display of results.

The authors of the best-performing models will be invited to contribute to a final paper of the competition.

## Submission Format

Submissions are expected to be in the form of a .zip file, which should contain a file named `model.py` defining a class named `Model` with a method `predict`.
This class will be instantiated exactly once, at the start of the evaluation, and the `predict` method will be called multiple times with different document-video attempts.
The `predict` method should take as input a potentially morphed document image and a video file, and output a binary label indicating whether the document image is the result of a morphing attack (1) or not (0).

The provided .zip file must be self-contained, meaning that it should include all necessary code and model weights to run the `predict` method.
To avoid sending large files, the compute workers that will run the provided code are equipped with several pre-installed libraries and weights of the most popular models.
Should participants use one of these pre-installed models, they can simply import it in their code without including the model weights in the .zip file.

## Preinstalled libraries

We provide the following pre-installed libraries, which can be used in your code without needing to include them in your submission:
* `numpy==2.3.5`
* `pandas==2.3.3`
* `scipy==1.16.3`
* `scikit-image==0.25.2`
* `scikit-learn==1.7.2`
* `pillow==12.0.0`
* `torch==2.10.0` (with GPU support)
* `torchvision==0.25.0`
* `transformers==5.3.0`
* `albumentations==2.0.8`
* `opencv-python==4.13.0.92`
* `timm==1.0.25`
* `face-alignment==1.4.1`
* `insightface==0.7.3`
* `facenet-pytorch==2.5.3`
* `tensorflow==2.20.0` (without GPU support)
* `deepface==0.0.99`

You can find the full list of dependencies in the `environment.yaml` file.


## Preinstalled models

We provide the following pre-trained model weights from HuggingFace Hub, which can be used in your code without needing to include them in your submission:
* `google/vit-base-patch16-224`
* `google/vit-large-patch16-384`
* `microsoft/swin-tiny-patch4-window7-224`
* `microsoft/swin-large-patch4-window7-224`
* `microsoft/swin-base-patch4-window7-224`
* `microsoft/beit-base-patch16-224`
* `facebook/dino-vitb8`
* `microsoft/resnet-50`
* `microsoft/resnet-101`
* `microsoft/resnet-18`
* `google/efficientnet-b0`
* `google/efficientnet-b2`
* `google/efficientnet-b4`
* `google/mobilenet_v2_1.4_224`
* `facebook/convnext-base-224-22k`
* `openai/clip-vit-base-patch32`
