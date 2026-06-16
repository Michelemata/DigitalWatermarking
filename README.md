# Comparative Analysis of Watermarking Techniques via Machine Learning: SVD vs. Blue Channel Approach

This repository contains a MATLAB project aimed at comparing two different digital watermarking techniques applied to images, evaluating their robustness and detectability through Machine Learning models.

## 📋 Project Description

The project implements and compares two watermark insertion strategies:
1. **SVD (Singular Value Decomposition) Approach on Luminance**: Uses an object detector (YOLOv8) to identify a Region of Interest (ROI). The watermark is embedded by altering the singular values (SVD) of the luminance channel (Y) exclusively within the ROI.
2. **Blue Channel Approach**: A more direct and computationally lighter technique that injects the watermark by uniformly modifying the pixels of the blue channel (B) across the entire image.

To evaluate the presence of the watermark and test its robustness (even in the presence of Gaussian or salt & pepper noise), global and local descriptors (features) are extracted, such as SSIM, PSNR, MSE, and channel differences. Three classifiers are trained on these features:
* **Logistic Regression**
* **Support Vector Machine (SVM)** (with an RBF kernel and hyperparameter tuning)
* **Random Forest** (with hyperparameter tuning)

## 📊 Dataset

The project utilizes a dataset consisting of **652 images** in JPEG format depicting cars viewed from the rear. The dataset combines two collections characterized by heterogeneous real-world contexts (from Caltech parking lots to Southern California highways) and varying resolutions:

* **Caltech Cars 1999**: 896 x 592 ([Dataset 1 Details](https://data.caltech.edu/records/fmbpr-ezq86))
* **Caltech Cars 2001**: 360 x 240 ([Dataset 2 Details](https://data.caltech.edu/records/dvx6b-vsc46))

## 🔍 Feature Extraction Process

To bridge the gap between image processing and Machine Learning, a robust set of **20 statistical and structural features** is extracted by comparing the original image with its watermarked (and noisy) counterpart via `computeWatermarkScore.m`:

- **Global Structural Metrics:** Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE).
- **Absolute Error Statistics:** Comprehensive error distribution characteristics, including mean, standard deviation, median, maximum value, 90th percentile, and overall error energy.
* **Global Color & Error Metrics:** Mean and standard deviation of pixel-wise differences calculated across the Red, Green, and Blue channels independently.
* **Linear Correlation Metric:** Grayscale Correlation Coefficient (Pearson) between the two images.
* **Local ROI Features (SVD-specific):** SSIM, PSNR, mean absolute error, and standard deviation computed strictly within the YOLOv8-detected Bounding Box (Region of Interest).

## 📁 Project Structure

```text
.
├── cars_1999_2001/                      # Folder containing the original dataset images.
├── Results/                             # Directory (auto-generated) for exported results (plots, csv).
├── dataset_A_.../                       # Directory (auto-generated) for images modified with SVD.
├── dataset_B_.../                       # Directory (auto-generated) for images modified on the Blue Channel.
├── modelli_ml.m                         # Main script: pipeline orchestration.
├── runSweepAlphaIdx.m                   # Script for automatic parametric sweep execution.
├── watermarkSVD.m                       # Applies the watermark via SVD on luminance (ROI identified by YOLO).
├── watermarkBlu.m                       # Applies the watermark by modifying the Blue channel of the entire image.
├── addNoise.m                           # Applies noise for robustness testing.
├── computeWatermarkScore.m              # Feature Extraction process.
├── tuneSVMHyperparameters.m             # Performs Grid Search to optimize SVM hyperparameters.
├── tuneRFHyperparameters.m              # Performs Grid Search to optimize Random Forest hyperparameters.
├── metrics.m                            # Calculates classification metrics.
├── extractPositiveClassScores.m         # Extracts positive class scores for ROC curve computation.
├── selectBestBoundingBox.m              # Selects the best YOLO bounding box based on score and area.
├── normalizeImageResolution.m           # Standardizes resolution and channels for all dataset images.
├── generatePlots.m                      # Generates main plots (boxplots, feature importance, visual heatmaps).
├── plotHyperparameterTuningResults.m    # Generates heatmaps and line plots for tuning analysis.
├── createComplexityBarPlot.m            # Generates bar plots related to computational complexity.
├── makeTrialMetricsLongTable.m          # Formatted results into tables for aggregating multiple trials.
├── getModelSize.m                       # Calculates and estimates the memory footprint (MB) of trained models.
└── README.md                            # This file, providing project documentation.
```

## 🚀 Installation and Usage

### 1\. Clone the Repository
To get started, clone this project to your local environment:
```bash
git clone https://github.com/Michelemata/DigitalWatermarking.git
cd DigitalWatermarking
```

### 2\. Prerequisites
To run the code properly, you must have **MATLAB** installed (a recent version supporting YOLOv8 is recommended) along with the following official Toolboxes:
* **Deep Learning Toolbox**
* **Image Processing Toolbox**
* **Statistics and Machine Learning Toolbox**
* **Computer Vision Toolbox**

### 3\. Running the Analysis
* **Single Run**:
  Open `modelli_ml.m` in MATLAB. Ensure that the `sweepMode` parameter is set to `false`. Click **Run**. The script will generate the output folders with the modified images, extract the features, train the models and evaluates them.
* **Parametric Sweep Run**:
  Run the `runSweepAlphaIdx.m` file directly. This will automatically test various visibility levels (alpha) and contamination percentages (idx), aggregating the results into a tabular log.

## 📈 Generated Outputs

Running the scripts will generate several folders, including:
* **`dataset_A_...`** / **`dataset_B_...`**: Folders containing the physically modified images.
* **`Results/`**: Contains all exported results (ideal for presentations or reports). You will find:
  * `.csv` files containing the extracted features, aggregated metrics, and computational complexity measurements.
  * Confusion matrices, ROC curves, Feature Importance charts, and distribution Boxplots saved as `.png`.
  * Visual "heatmap" representations of the differences between the original and processed images.

## 📉 Evaluation

To evaluate and compare the performance of the three classification models (Logistic Regression, Support Vector Machine, and Random Forest), the following metrics were calculated:
* **Accuracy** (Global accuracy of the model)
* **Precision** (Ability to avoid classifying original images as watermarked)
* **Recall** (True positive detection rate)
* **F1-score** (Harmonic mean of Precision and Recall, used as the primary metric for tuning)
* **AUC (Area Under the Curve)** and **ROC Curves** (To analyze the balance between TPR and FPR at different thresholds)
* **Confusion Matrices** (For a visual breakdown of classification errors)

The results are printed to the console.

## 🎓 Acknowledgements

This project was carried out as part of the **Mathematical and Statistical Methods for Artificial Intelligence** course taught by Professor **Marina Popolizio** at the Politecnico di Bari. The project was developed in collaboration with **Colavitto Valeria**.

## 📄 License

This project is released under the MIT License. See the LICENSE file for more details.
