# Facial Emotion Recognition — CNN Optimization on FER2013

CNN-based facial emotion recognition system trained on FER2013, with a real-time inference pipeline using OpenCV and Haar Cascade for face detection. This project investigates how architecture depth, learning rate, and data augmentation strategy affect model performance in classifying seven facial expressions: anger, disgust, fear, happiness, sadness, surprise, and neutral.

The project is split across two notebooks, run in two different environments:
- **`emotion-project.ipynb`** — training pipeline, run on Kaggle (GPU-backed) against the FER2013 dataset.
- **`real-time-emotions.ipynb`** — real-time inference, run locally in an Anaconda environment, loading the model produced by the training notebook and classifying emotions live from the webcam.

---

## 📋 Overview

Facial emotion recognition (FER) is relevant for social robotics, human-computer interaction, and intelligent interfaces, since it gives machines a new layer of perception of human states. This project adapts a reference convolutional architecture and systematically evaluates:

- **Architecture depth** — adding an extra convolutional layer and additional Batch Normalization layers to a baseline CNN
- **Learning rate** — comparing `α = 0.001` and `α = 0.0001` with the Adam optimizer
- **Data augmentation strategy** — horizontal flip only vs. horizontal flip + width/height shift + zoom

Models are compared using validation accuracy/loss and per-class precision, recall, and F1-score, then validated in a real-time webcam scenario.

---

## 🗂️ Dataset

**FER2013** — 35,887 grayscale face images (48×48 px), scraped from the web and originally released as a Kaggle challenge dataset.

| Split | Images |
|---|---:|
| Train | 28,709 |
| Validation | 3,589 |
| Test | 3,589 |

Labels: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.

---

## 🧠 Architectures

Two architectures were compared, both trained with the Adam optimizer:

| Layer | Base Model | Proposed Model |
|---|---|---|
| conv2d | (48,48,32) | (48,48,32) |
| conv2d | – | (48,48,32) |
| batch norm. | – | (48,48,32) |
| max pooling | (24,24,32) | (24,24,32) |
| dropout | – | (24,24,32) |
| conv2d | (24,24,64) | (24,24,64) |
| max pooling | (12,12,64) | (12,12,64) |
| batch norm. | (12,12,64) | (12,12,64) |
| dropout | (12,12,64) | (12,12,64) |
| conv2d | (12,12,128) | (12,12,128) |
| batch norm. | (12,12,128) | (12,12,128) |
| max pooling | (6,6,128) | (6,6,128) |
| dropout | (6,6,128) | (6,6,128) |
| conv2d | (6,6,512) | (6,6,512) |
| batch norm. | (6,6,512) | (6,6,512) |
| max pooling | (3,3,512) | (3,3,512) |
| dropout | (3,3,512) | (3,3,512) |
| flatten | (4608) | (4608) |
| dense | (256) | (256) |
| batch norm. | – | (256) |
| dropout | (256) | (256) |
| dense (output) | (7) | (7) |

The **Proposed Model** adds a second conv2d layer to the first extraction block (32 filters), extra Batch Normalization layers throughout the network, and a dedicated normalization layer before the final Dropout.

Three training configurations were evaluated on the Proposed Model:
1. Horizontal flip, `α = 0.001` — same learning rate and augmentation as the base model
2. Horizontal flip, `α = 0.0001` — reduced learning rate, to test convergence stability
3. Horizontal flip + width/height shift + zoom, `α = 0.001` — combined augmentation techniques

---

## 📊 Results

### Architecture and Learning Rate Impact

| Configuration | Train Acc. | Val. Acc. | Train Loss | Val. Loss |
|---|---:|---:|---:|---:|
| Base Model (α = 0.001) | 64.15% | 60.41% | 1.2586 | 1.3669 |
| Base Model (α = 0.0001) | 77.25% | 62.78% | 0.8840 | 1.4262 |
| Proposed Model (α = 0.001) | 68.35% | 63.55% | 1.2583 | 1.4261 |
| **Proposed Model (α = 0.0001)** | **75.34%** | **64.73%** | 0.9131 | **1.2565** |

Adding filters and regularization layers raised validation accuracy from 60.41% to 63.55% at the same learning rate. Lowering `α` to 0.0001 on the proposed model gave the best validation performance overall (64.73%), with a more consistent drop in validation loss — indicating more stable convergence.

### Data Augmentation Impact

| Augmentation Strategy | Train Acc. | Val. Acc. | Train Loss | Val. Loss |
|---|---:|---:|---:|---:|
| Horizontal only (α = 0.001) | 68.35% | 63.55% | 1.2583 | 1.4261 |
| All methods (α = 0.001) | 61.05% | 64.73% | 1.3139 | **1.2284** |
| Horizontal only (α = 0.0001) | 75.34% | 64.73% | 0.9131 | 1.2565 |
| All methods (α = 0.0001) | 61.12% | 63.62% | 1.1769 | 1.1243 |

Combining multiple augmentation techniques at `α = 0.001` produced the lowest validation loss among all experiments. It also reduced training accuracy (from 75.34% to ~61%), consistent with augmentation's regularizing role: the model is forced to learn general expression patterns rather than memorizing specific training images.

### Per-Class Performance

**Proposed Model, horizontal flip only (α = 0.0001):**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Angry | 56.28% | 63.35% | 59.61% |
| Disgust | 63.16% | 54.55% | 58.54% |
| Fear | 63.64% | 37.75% | 47.38% |
| Happy | 78.87% | 86.44% | 82.48% |
| Sad | 59.92% | 58.94% | 59.43% |
| Surprise | 48.62% | 56.63% | 52.32% |
| Neutral | 79.62% | 75.30% | 77.40% |

**Proposed Model, full augmentation (α = 0.001):**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Angry | 66.45% | 53.93% | 59.54% |
| Disgust | 80.00% | 18.18% | 29.63% |
| Fear | 57.28% | 28.92% | 38.44% |
| Happy | 88.12% | 85.88% | 86.98% |
| Sad | 51.98% | 80.08% | 63.04% |
| Surprise | 52.94% | 50.60% | 51.75% |
| Neutral | 64.73% | 80.72% | 71.85% |

`Happy` and `Neutral` are consistently the best-recognized classes. `Fear` is the hardest to classify in both configurations — likely due to visual overlap with other negative expressions (`Surprise`, `Sad`). Under full augmentation, `Disgust` shows high precision (80.00%) but very low recall (18.18%): the model rarely predicts this class, but is usually right when it does — a pattern consistent with `Disgust`'s low representation in FER2013. `Sad` recall increases sharply (80.08%) under the same configuration, suggesting the model tends to default to `Sad` for underrepresented negative expressions like `Disgust` and `Fear`.

### Real-Time Validation

The trained model was validated with live webcam inference: Haar Cascade detects and crops the face region, which is then resized to 48×48 and classified in real time.

Observed confidence levels per emotion during live testing: **Neutral** (82.2%), **Surprise** (94.8%), **Happy** (100.0%), **Angry** (81.5%). Despite the lower offline scores for `Fear` and `Disgust`, the system performed reliably and with high confidence on the emotions most frequently expressed in a real interaction scenario — supporting its viability for social robotics applications.

---

## 🗂️ Repository Structure

```
.
├── emotion-project.ipynb          # Training pipeline (Kaggle, GPU): data loading,
│                                   # augmentation, model definition, training, evaluation
├── emotions.ipynb                 # Real-time webcam inference (local, Anaconda)
├── models/
│   ├── modelo_completo.keras
├── haarcascade_frontalface_default.xml
└── article/                       # IEEE-format article (LaTeX/PDF)
```

---

## ⚙️ Dependencies

- Python 3
- TensorFlow / Keras
- OpenCV (`cv2`) — face detection with Haar Cascade, real-time inference
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn (`classification_report`, `confusion_matrix`)

---

## ▶️ How to Use

### 1. Train the model (Kaggle)

Open `emotion-project.ipynb` on Kaggle (or any GPU-backed environment). Update the dataset paths (`train_dir`, `test_dir`) to point to your FER2013 directory structure (`train/` and `test/`, one subfolder per class). Run the cells in order, then save/export the trained model — e.g. `model.save('modelo_completo.keras')`.

Key training parameters:
```python
img_size = 48
epochs = 60
batch_size = 64
```

Download the resulting model file to run inference locally.

### 2. Real-time inference (local, Anaconda)

Copy the trained model file into the local project folder alongside `haarcascade_frontalface_default.xml`, then run `emotions.ipynb` in an Anaconda environment (a Conda env with TensorFlow/Keras + OpenCV — this notebook was run under `conda env:anaconda3-visao`). It loads the model and the Haar Cascade classifier, then opens the webcam and classifies emotions frame by frame:

1. Each frame is converted to grayscale.
2. `face_cascade.detectMultiScale` locates faces (`scaleFactor=1.1`, `minNeighbors=5`, `minSize=(60, 60)`).
3. Each detected face is cropped, resized to 48×48, and normalized to `[0, 1]` — the same preprocessing used during training.
4. The model predicts the emotion class; the predicted label and confidence are drawn on the frame as a bounding box + text.
5. Press `q` to close the window and release the webcam.

```python
MODEL_PATH = "modelo_completo.keras"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
IMG_SIZE = 48

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
```

> **Note on `.keras` loading:** models saved with a newer Keras version can include config fields (e.g. `quantization_config`, `lora_rank`, `lora_alpha`) that older Keras/TensorFlow installs don't recognize. `emotions.ipynb` includes a small helper that strips these incompatible fields from the model's `config.json` before deserializing, then loads the original weights into the cleaned architecture — this avoids re-saving the model just to fix a version mismatch.

---

## Conclusion

Adding regularization layers to the base architecture and lowering the learning rate to 0.0001 improved validation accuracy and convergence stability. Combining multiple data augmentation techniques produced the lowest validation loss among all experiments. `Happy` and `Neutral` were the best-recognized emotions; `Fear` and `Disgust` remain the hardest, reflecting challenges already reported in the literature regarding visual overlap between expressions and class imbalance in FER2013. Real-time validation confirmed the system's practical viability. Future work includes investigating class balancing techniques and exploring attention-based or Transformer architectures.

---

## References

- L. T. C. Ottoni, "Sistema para reconhecimento de emoção multimodal e multiclasse para a interação humano-robô," Doctoral Thesis, Escola Politécnica, Universidade Federal da Bahia, Salvador, 2024.
- F. Chollet, *Deep Learning with Python*, 2nd ed. Shelter Island, NY, USA: Manning Publications, 2021.

---

## Author

**Ludmila Nascimento dos Anjos**
Department of Electrical and Computer Engineering, Universidade Federal da Bahia
Salvador, Brasil
ludmila.n.anjos@gmail.com
