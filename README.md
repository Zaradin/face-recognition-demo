# Face Recognition Demo (InsightFace + ONNX Runtime)

This project demonstrates a simple **face recognition pipeline** implemented in Python using pretrained models from **InsightFace**. The system detects faces in an image, extracts numerical embeddings representing each face, and compares them with stored embeddings to identify known individuals.

The project is implemented as a **Jupyter Notebook demo**, designed to illustrate how modern face recognition systems work using deep learning and similarity-based matching.

## Overview

The pipeline implemented in this project follows these steps:

1. **Face Detection**  
   Detect faces within an image and return bounding boxes.

2. **Face Embedding Extraction**  
   Convert each detected face into a **512-dimensional vector** representation.

3. **Similarity Comparison**  
   Compare the embedding with known embeddings stored in a gallery database.

4. **Identity Prediction**  
   The closest embedding (highest similarity score) determines the predicted identity.

## Example Output

The system detects faces in an image and labels them with the predicted identity and similarity score.

Bounding boxes are drawn around detected faces.

## Technologies Used

### InsightFace

This project uses **InsightFace**, an open-source deep learning library for face analysis.

InsightFace provides pretrained models for:

- face detection
- face recognition
- face alignment
- landmark detection
- age/gender estimation

GitHub:  
https://github.com/deepinsight/insightface

### ONNX

The models used in InsightFace are stored in **ONNX (Open Neural Network Exchange)** format.

ONNX is a standard format for machine learning models that allows models trained in frameworks like **PyTorch or TensorFlow** to be run in different environments.

Official site:  
https://onnx.ai/

### ONNX Runtime

The neural networks are executed using **ONNX Runtime**, a high-performance inference engine designed to run ONNX models efficiently on various hardware backends.

It supports multiple execution providers including:

- CPU
- CUDA (NVIDIA GPUs)
- CoreML (Apple hardware acceleration)

Docs:  
https://onnxruntime.ai/

## Model Pack: `buffalo_l`

This project uses the **`buffalo_l` model pack** provided by InsightFace.

The pack contains multiple pretrained models for different face analysis tasks.

Typical contents:

| Model            | Purpose                      |
| ---------------- | ---------------------------- |
| `det_10g.onnx`   | Face detection               |
| `w600k_r50.onnx` | Face recognition embeddings  |
| `2d106det.onnx`  | 2D facial landmark detection |
| `1k3d68.onnx`    | 3D facial landmarks          |
| `genderage.onnx` | Age and gender prediction    |

For this project, only the following models are used:

- **`det_10g.onnx`** – detects faces in an image
- **`w600k_r50.onnx`** – generates 512-dimensional face embeddings

Other models in the pack are ignored.

## Face Embeddings

Instead of directly classifying identities, modern face recognition systems represent faces using **embeddings**.

An embedding is a numerical vector that captures distinguishing facial features.

Example:
Amil → [0.12, -0.88, 0.41, ...]
Tom → [-0.71, 0.34, 0.55, ...]

<img src="./image_predictions/amil_predict.jpg" width="150"/>

Faces belonging to the same person produce **similar embeddings**, while different individuals produce **distant embeddings**.

Identity is determined by computing a similarity score between embeddings.

## PCA Visualization

Since embeddings typically have **512 dimensions**, they cannot be visualized directly.

To explore the structure of this high-dimensional space, **Principal Component Analysis (PCA)** is used to project embeddings into **2D or 3D space**.

This allows us to visualize how faces from different individuals cluster in embedding space.
