# CIFAR-10 Image Classification: DNN vs CNN

A practice project (AISV 802) building and comparing two image classifiers from scratch on the **CIFAR-10** dataset (10 classes, 60,000 32x32 color images) using **TensorFlow** and **Keras**.

## Results

| Model | API | Epochs | Test Accuracy |
|---|---|---|---|
| DNN (fully connected) | Sequential | 30 | **56.12%** |
| DNN (fully connected) | Subclassed | 5 (check) | 38.46% |
| CNN | Sequential | 40 | **82.89%** |
| CNN | Subclassed | 5 (check) | 61.71% |
| CNN (tuned) | Sequential | 40 | ~86.5% val. accuracy |

The CNN outperforms the plain DNN by ~27 points, since convolution + pooling exploit the spatial structure of images while a flattened DNN treats pixels as an unordered list of numbers.

## What's inside

- **`tf.data` pipeline** — proper train/validation/test split (45k / 5k / 10k), with shuffling, batching, and prefetching
- **Data augmentation** — random flip, rotation, zoom, translation, and contrast, applied only to the training split
- **DNN** — built with both the Keras `Sequential` API and `Model` subclassing
- **CNN** — same dual-API treatment, using stacked Conv2D + BatchNorm + MaxPool blocks
- **Hyperparameter tuning** — Keras Tuner `RandomSearch` over filter width, dropout, L2, and learning rate
- **TensorBoard** logging for every training run (scalars, histograms, learning-rate schedule)
- **Evaluation** — confusion matrix and per-class precision/recall/F1 on the CNN

## Files

- `CIFAR10_DNN_CNN.ipynb` — full notebook (code, training, evaluation)
- `CIFAR10_DNN_CNN_Report.pdf` — written report with results, charts, and discussion

## How to run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) (GPU runtime recommended: Runtime → Change runtime type → T4 GPU)
2. Run all cells top to bottom — `tf.keras.datasets.cifar10` downloads the dataset automatically on first run
3. Training + tuning takes roughly 20–30 minutes on a T4 GPU

## Tools

TensorFlow / Keras · Keras Tuner · scikit-learn · TensorBoard · Google Colab
