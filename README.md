# Human Activity Recognition Using a Gaussian Hidden Markov Model

**Python | NumPy | Signal Processing | Time-Series Analysis | Machine Learning**

## Project Overview

This project implements a **Gaussian Hidden Markov Model (HMM) from scratch using NumPy** to recognize four human activities from smartphone accelerometer and gyroscope data.

The project covers the complete machine learning workflow:

1. Collecting and validating real-world sensor recordings.
2. Synchronizing, filtering, and resampling sensor signals.
3. Extracting time-domain and frequency-domain features.
4. Implementing and training a Gaussian HMM.
5. Decoding activity sequences with the Viterbi algorithm.
6. Evaluating the model on previously unseen recordings.

The four activities are **Walking, Jumping, Standing, and Still**.

The implementation is organized into three Jupyter notebooks, supported by the processed dataset, extracted features, trained model, and academic report.

---

## Dataset

### Data Collection

The dataset was collected personally using a smartphone sensor-logging application.

Two sensors were recorded:

- **Accelerometer:** Linear acceleration along the X, Y, and Z axes (m/s²).
- **Gyroscope:** Angular velocity along the X, Y, and Z axes (rad/s).

Each recording lasted approximately **5–10 seconds**.

The activities were performed as follows:

| Activity | Collection Procedure |
|---|---|
| Walking | Walking at a consistent pace while carrying the phone |
| Jumping | Performing repeated jumps while carrying the phone |
| Standing | Standing while holding the phone steadily near waist level |
| Still | Placing the phone motionless on a flat surface |

### Dataset Distribution

| Activity | Training Recordings | Testing Recordings |
|---|---:|---:|
| Walking | 13 | 2 |
| Jumping | 12 | 2 |
| Standing | 12 | 2 |
| Still | 13 | 2 |
| **Total** | **50** | **8** |

The dataset contains **58 independent recordings**, with eight recordings reserved for testing.

The processed recordings are available in [`data/processed/`](data/processed/), organized by activity and training/testing split.

Each CSV contains the following columns:

`time_s`, `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`

The signals were synchronized and resampled to **25 Hz**. The preprocessing process also included quality validation and low-pass filtering.

### Data Collection Video Evidence

Four videos document the activity collection procedures.

**[View Data Collection Videos (Google Drive)](https://drive.google.com/drive/folders/17YPcBk_a7-3nDLnKP6GXnCP-e8ZgXAr9?usp=sharing)**

The folder contains:

- `Walking.mp4`
- `Jumping.mp4`
- `Standing.mp4`
- `Steady_phone_on_flat_surface.mp4`

The videos are maintained separately from the GitHub repository.

---

## Technical Implementation

### Data Preprocessing

The preprocessing pipeline includes sensor data validation, synchronization, Butterworth low-pass filtering, and resampling to a common sampling frequency of 25 Hz.

### Feature Extraction

Each recording was segmented into **2-second windows with 50% overlap**.

Twelve time-domain and frequency-domain features were initially extracted. Following correlation analysis, six were selected for HMM modeling:

- Acceleration RMS
- Gyroscope RMS
- Acceleration X–Y correlation
- Acceleration dominant frequency
- Acceleration spectral energy
- Gyroscope spectral energy

Features were standardized using statistics fitted exclusively on the training dataset.

### HMM Implementation

A four-state Gaussian HMM with diagonal covariance matrices was implemented using NumPy.

The implementation includes:

| Algorithm | Purpose |
|---|---|
| Forward–Backward | Compute posterior state probabilities and expected transitions |
| Baum–Welch | Estimate HMM parameters using Expectation–Maximization |
| Viterbi | Decode the most likely hidden-state sequence |

The implementation uses log-space computations for numerical stability and includes a convergence criterion for Baum–Welch training.

**The HMM algorithms were implemented from scratch without using `hmmlearn`.**

The complete implementation, evaluation, and interpretations are documented in the notebooks and academic report.

---

## Repository Structure

```text
Human-Motion-Recognition_HMM/
│
├── README.md
│
├── Human_Activity_Recognition_HMM_Report.docx.pdf
│
├── data/
│   ├── processed/
│   │   ├── Walking/
│   │   │   ├── Train/
│   │   │   └── Test/
│   │   ├── Jumping/
│   │   │   ├── Train/
│   │   │   └── Test/
│   │   ├── Standing/
│   │   │   ├── Train/
│   │   │   └── Test/
│   │   ├── Still/
│   │   │   ├── Train/
│   │   │   └── Test/
│   │   └── recordings_manifest.csv
│   │
│   └── features/
│       ├── train_features.csv
│       ├── test_features.csv
│       └── feature_selection.json
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_hmm_modeling.ipynb
│
├── models/
│   ├── gaussian_hmm_4state.npz
│   ├── test_activity_metrics.csv
│   ├── test_recording_predictions.csv
│   ├── walking_feature_comparison.csv
│   └── walking_feature_contributions.csv
│
└── figures/
    └── [Saved visualization files]
```

### Notebook Descriptions

| Notebook | Description |
|---|---|
| `01_data_preprocessing.ipynb` | Dataset inspection, quality validation, sensor synchronization, filtering, and resampling |
| `02_feature_extraction.ipynb` | Window segmentation, feature engineering, correlation analysis, and feature selection |
| `03_hmm_modeling.ipynb` | Gaussian HMM implementation, Baum–Welch training, Viterbi decoding, evaluation, and error analysis |

---

## Running and Reproducing the Project

### Requirements

- Python 3
- NumPy
- pandas
- SciPy
- Matplotlib
- scikit-learn
- Jupyter Notebook or Google Colab

Install the required Python libraries:

```bash
pip install numpy pandas scipy matplotlib scikit-learn jupyter
```

### 1. Clone the Repository

```bash
git clone https://github.com/Gaddy01/Human-Motion-Recognition_HMM.git
cd Human-Motion-Recognition_HMM
```

### 2. Open the Notebooks in Google Colab

The notebooks can be opened directly from GitHub:

1. [Data Preprocessing](https://colab.research.google.com/github/Gaddy01/Human-Motion-Recognition_HMM/blob/main/notebooks/01_data_preprocessing.ipynb)
2. [Feature Extraction](https://colab.research.google.com/github/Gaddy01/Human-Motion-Recognition_HMM/blob/main/notebooks/02_feature_extraction.ipynb)
3. [HMM Modeling](https://colab.research.google.com/github/Gaddy01/Human-Motion-Recognition_HMM/blob/main/notebooks/03_hmm_modeling.ipynb)

### 3. Configure the Google Drive Paths

The notebooks were originally developed in Google Colab using this project directory:

```text
/content/drive/MyDrive/Formative2 HMM/
```

They expect the following folder names:

```text
Formative2 HMM/
├── Raw Collected Data/
├── Processed Data/
├── Features/
├── Models/
└── Plots/
```

**Note:** The repository uses different directory names from the original Google Drive project. Therefore, the notebook paths must be configured before running them independently.

To prepare the publicly available processed data, features, model artifacts, and plots for Google Colab, run the following setup cell:

```python
from google.colab import drive
from pathlib import Path
import subprocess
import shutil

drive.mount("/content/drive")

repo = Path("/content/hmm_repo")
project = Path("/content/drive/MyDrive/Formative2 HMM")

if not repo.exists():
    subprocess.run([
        "git", "clone",
        "https://github.com/Gaddy01/Human-Motion-Recognition_HMM.git",
        str(repo)
    ], check=True)

project.mkdir(parents=True, exist_ok=True)

folder_mapping = {
    "data/processed": "Processed Data",
    "data/features": "Features",
    "models": "Models",
    "figures": "Plots"
}

for source, destination in folder_mapping.items():
    shutil.copytree(
        repo / source,
        project / destination,
        dirs_exist_ok=True
    )

print("Project files are ready.")
```

### 4. Execute the Notebooks

Once the project folders are prepared, execute:

**Step 1: Feature Extraction**

Open `02_feature_extraction.ipynb` and run its cells in order. This notebook reads the processed sensor CSV files and generates the feature datasets.

**Step 2: HMM Modeling**

Open `03_hmm_modeling.ipynb` and run its cells in order. This notebook trains the HMM, evaluates it using the held-out recordings, and generates the model outputs and visualizations.

The modeling notebook can also be executed directly using the provided feature datasets.

**Raw-data preprocessing:** The first notebook requires the original sensor ZIP archives, which are preserved separately and are not included in the public repository. The processed CSV files are provided so the feature extraction and modeling stages can be reproduced without the original archives.

---

## Model and Output Files

The [`models/`](models/) directory contains the trained HMM and its saved evaluation artifacts.

| File | Description |
|---|---|
| `gaussian_hmm_4state.npz` | Trained HMM parameters and fitted standardization statistics |
| `test_activity_metrics.csv` | Per-activity evaluation metrics |
| `test_recording_predictions.csv` | Recording-level predictions |
| `walking_feature_comparison.csv` | Feature comparison from the Walking error analysis |
| `walking_feature_contributions.csv` | Feature-level emission evidence for the Walking error analysis |

The [`figures/`](figures/) directory contains the visualizations generated during preprocessing, training, and evaluation.

The [`data/features/`](data/features/) directory contains the extracted training and testing features, along with the selected-feature configuration.

---

## Academic Report

The complete academic report includes the methodology, HMM formulation, results, visualizations, evaluation metrics, discussion, and conclusions.

**[Read the Full Academic Report (PDF)](Human_Activity_Recognition_HMM_Report.docx.pdf)**

---

## Author

**Gaddiel Irakoze**  
Bachelor of Software Engineering  
African Leadership University
