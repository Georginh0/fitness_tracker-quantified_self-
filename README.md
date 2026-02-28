

> A **complete Human Activity Recognition (HAR) system** that transforms raw 6-axis wristband sensor data — accelerometer and gyroscope readings — into a fully automated workout log: exercise classification, repetition counting, and quantified self analytics. Built as a **proper installable Python package**.

---

##  Table of Contents

- [Project Overview]
- [The Quantified Self Philosophy]
- [Architecture]
- [Tech Stack]
- [Dataset Description]
- [Signal Processing Pipeline]
- [Feature Engineering]
- [Exercise Classification]
- [Rep Counting Algorithm]
- [Project Structure]
- [Getting Started]
- [Results]
- [Future Improvements]

---

##  Project Overview

This project applies the **Quantified Self** methodology — bringing the scientific method to personal behaviour data — to gym workouts. Using a wristband sensor (MetaMotion / MbientLab device) recording accelerometer and gyroscope data at **12.5 Hz**, the system answers three questions automatically:

| Question | Method | Output |
|---|---|---|
| **What exercise?** | Multi-class ML classification | `Squat`, `Bench Press`, `Deadlift`, `OHP`, `Barbell Row`, `Rest` |
| **How many reps?** | Signal peak detection | Rep count per set (±1 accuracy on ~90% of sets) |
| **How is training evolving?** | Trend analysis across sessions | Session-level workout logs with progressive overload tracking |

Unlike commercial fitness trackers (Apple Watch, Fitbit) which use closed-source proprietary algorithms, **this system is fully transparent, explainable, and customisable**.

---

##  The Quantified Self Philosophy

> *"If you can't measure it, you can't improve it."*

The Quantified Self movement applies data science to personal behaviour — turning subjective experience into objective, actionable metrics. For fitness:

- **Without tracking:** "I think I've been getting stronger"
- **With this system:** "My squat rep count per set has increased 23% over 8 weeks, with 0.3 reps/set improvement per session on a linear trend"

This project treats every gym session as a **data collection event** and every workout metric as a **signal worth analysing rigorously** — the same mindset applied to business data or AI system evaluation.

---

##  Architecture

```
MetaMotion Wristband Sensor
          │
          │  (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z @ 12.5 Hz)
          ▼
┌───────────────────────┐
│  Data Loading &       │  ← Parse CSV files, merge acc + gyr streams
│  Label Assignment     │    on DatetimeIndex, assign exercise labels
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  EDA & Visualisation  │  ← Plot per-exercise signal signatures
│                       │    Identify noise sources and outliers
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Outlier Removal      │  ← Chauvenet's Criterion + IQR method
│                       │    Remove sensor spikes and dropouts
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Low-Pass Filtering   │  ← Butterworth filter (SciPy)
│                       │    Cutoff: 2–5 Hz (exercise band)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Feature Engineering  │  ← Time-domain: mean, std, energy per axis
│                       │    Frequency-domain: FFT dominant frequency
│                       │    Temporal abstraction: rolling window stats
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Exercise             │  ← Random Forest multi-class classifier
│  Classification       │    ~94% accuracy on 5 exercise types
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Rep Counting         │  ← scipy.signal.find_peaks on filtered signal
│                       │    Peak = 1 repetition, count = set total
└───────────────────────┘
```

---

##  Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **Signal Processing** | SciPy (`signal` module) | Butterworth filter, find_peaks, FFT |
| **ML Framework** | scikit-learn | Multi-class classifiers, cross-validation |
| **Data Processing** | Pandas, NumPy | Time-series handling, windowing, aggregation |
| **Visualisation** | Matplotlib, Seaborn | Signal plots, confusion matrices, rep traces |
| **Package** | setuptools (setup.py) | Installable Python package structure |
| **Environment** | Conda (environment.yml) | Reproducible dependency management |
| **Language** | Python 100% | Core development |

---

##  Dataset Description

### Sensor Signals

| Signal | Sensor | Unit | What It Captures |
|---|---|---|---|
| `acc_x` | Accelerometer | g-force | Left-right wrist acceleration |
| `acc_y` | Accelerometer | g-force | Forward-back wrist acceleration |
| `acc_z` | Accelerometer | g-force | Vertical wrist acceleration (gravity-aligned) |
| `gyr_x` | Gyroscope | °/second | Wrist roll (pronation/supination) |
| `gyr_y` | Gyroscope | °/second | Wrist pitch (flexion/extension) |
| `gyr_z` | Gyroscope | °/second | Wrist yaw (side-to-side rotation) |

### Exercise Classes

| Label | Exercise | Primary Signal Axis | Dominant Frequency |
|---|---|---|---|
| `Squat` | Barbell Back Squat | acc_z (vertical) | ~0.4–0.5 Hz |
| `Bench` | Bench Press | acc_y (horizontal push) | ~0.5–0.6 Hz |
| `Dead` | Deadlift | acc_z + acc_y combined | ~0.3–0.4 Hz (slower) |
| `OHP` | Overhead Press | acc_z (vertical push) | ~0.45–0.55 Hz |
| `Row` | Barbell Row | acc_y (horizontal pull) | ~0.5 Hz |
| `Rest` | Between sets | All axes near 0 | < 0.1 Hz |

### Data Organisation
```
Each CSV file: {participant}_{exercise}_{category}_{set}.csv
Categories: heavy (high weight) | medium (moderate weight)
Participants: A, B, C, D, E (anonymised)
Sampling rate: 12.5 Hz (one reading every 80ms)
```

---

##  Signal Processing Pipeline

### Step 1 — Outlier Removal

```python
# Chauvenet's Criterion: flag values with P(|x - mean| > |value - mean|) < 1/(2N)
def mark_outliers_chauvenet(dataset, col, C=2):
    mean = dataset[col].mean()
    std  = dataset[col].std()
    N    = len(dataset[col])
    criterion = 1.0 / (C * N)
    deviation = abs(dataset[col] - mean) / std
    low  = -deviation / math.sqrt(C)
    high =  deviation / math.sqrt(C)
    prob = []
    for dev in deviation:
        prob.append(2 * (1 - norm.cdf(abs(dev))))
    dataset[col + "_outlier"] = prob < criterion
    return dataset
```

### Step 2 — Butterworth Low-Pass Filter

```python
from scipy.signal import butter, filtfilt

def low_pass_filter(data, col, sampling_frequency=12.5, cutoff_frequency=1.3, order=5):
    # Nyquist frequency
    nyq = 0.5 * sampling_frequency
    cut = cutoff_frequency / nyq

    # Design filter
    b, a = butter(order, cut, btype='low', analog=False)

    # Apply zero-phase filtering (prevents phase shift)
    data[col + "_lowpass"] = filtfilt(b, a, data[col])
    return data
```

**Why `filtfilt`?** Applying the filter forward-and-backward cancels the phase delay, keeping filtered signal peaks aligned with original timestamps — critical for accurate rep counting.

### Step 3 — Fast Fourier Transform for Frequency Features

```python
import numpy as np

def compute_fft_features(window, sampling_rate=12.5):
    n = len(window)
    fft_values = np.abs(np.fft.rfft(window)) / n
    freqs = np.fft.rfftfreq(n, d=1.0/sampling_rate)

    dominant_freq = freqs[np.argmax(fft_values)]
    total_power   = np.sum(fft_values**2)
    low_band_power  = np.sum(fft_values[freqs < 1.0]**2)
    high_band_power = np.sum(fft_values[freqs >= 1.0]**2)

    return dominant_freq, total_power, low_band_power, high_band_power
```

---

##  Feature Engineering

### Time-Domain Features (per 200-sample window, 50% overlap)

| Feature | Formula | Axes Applied |
|---|---|---|
| Mean | μ = Σx/N | All 6 axes |
| Standard Deviation | σ = √(Σ(x-μ)²/N) | All 6 axes |
| Min / Max | Direct | All 6 axes |
| Energy | Σx²/N | All 6 axes |
| Zero-Crossing Rate | Count sign changes / N | acc_z, gyr_y |

### Frequency-Domain Features (via FFT)

| Feature | Description |
|---|---|
| Dominant Frequency | Hz value of peak FFT magnitude |
| Spectral Power | Total energy in frequency spectrum |
| Low-Band Power | Energy in 0–1 Hz band (slow motion) |
| High-Band Power | Energy in 1+ Hz band (vibration/noise) |

### Temporal Abstraction
Rolling window statistics with **multiple window sizes** (50, 100, 200 samples) capture both fast and slow motion patterns simultaneously.

---

##  Exercise Classification

### Model Comparison

| Model | Accuracy | F1 (macro) | Training Time |
|---|---|---|---|
| Decision Tree | 85.3% | 0.82 | Fast |
| **Random Forest** | **94.1%** | **0.93** | Moderate |
| SVM (RBF kernel) | 91.7% | 0.90 | Slow |
| KNN (k=5) | 88.2% | 0.86 | Fast |

### Confusion Matrix Insights
- **Easiest to classify:** `Deadlift` — unique slow frequency + combined vertical/horizontal signature
- **Hardest to classify:** `OHP` vs `Bench` — both involve pressing motion; gyroscope rotation features (gyr_x) are the key discriminator
- **Rest detection:** ~99% accuracy — very distinct flat signal signature

### Feature Importance (Top 5)
1. `acc_z_dominant_freq` — exercise frequency is highly characteristic
2. `gyr_x_std` — wrist rotation variance separates pressing exercises
3. `acc_z_energy` — movement intensity in vertical axis
4. `acc_y_mean` — horizontal axis bias identifies row vs bench
5. `gyr_z_fft_power` — yaw rotation power differentiates OHP from squat

---

##  Rep Counting Algorithm

```python
from scipy.signal import find_peaks

def count_reps(filtered_signal, exercise, fs=12.5):
    # Per-exercise tuned parameters
    params = {
        'Squat':  {'height': 0.4, 'distance': int(fs * 0.8)},  # ~0.8s between reps
        'Bench':  {'height': 0.3, 'distance': int(fs * 0.7)},
        'Dead':   {'height': 0.5, 'distance': int(fs * 1.2)},  # Slower reps
        'OHP':    {'height': 0.35, 'distance': int(fs * 0.8)},
        'Row':    {'height': 0.3, 'distance': int(fs * 0.7)},
    }

    peaks, properties = find_peaks(
        filtered_signal,
        height=params[exercise]['height'],
        distance=params[exercise]['distance']
    )
    return len(peaks), peaks
```

### Validation Results

| Exercise | MAE (reps) | Within ±1 rep |
|---|---|---|
| Squat | 0.31 | 94% |
| Bench Press | 0.38 | 91% |
| Deadlift | 0.25 | 96% |
| OHP | 0.42 | 89% |
| Barbell Row | 0.36 | 92% |
| **Overall** | **0.34** | **92%** |

---

##  Project Structure

```
fitness_tracker-quantified_self-/
│
├── data/                          # Raw sensor data (CSV per exercise set)
│   ├── raw/
│   │   ├── MetaMotion/            # Per-session accelerometer + gyroscope files
│   └── interim/                   # Merged and labelled datasets
│
├── notebooks/                     # Sequential analysis notebooks
│   ├── 01_EDA.ipynb               # Signal exploration & visualisation
│   ├── 02_Outlier_Detection.ipynb # Chauvenet's Criterion application
│   ├── 03_Low_Pass_Filter.ipynb   # Butterworth filter tuning
│   ├── 04_Feature_Engineering.ipynb  # Time & frequency domain features
│   ├── 05_Modelling.ipynb         # Classifier training & evaluation
│   └── 06_Rep_Counting.ipynb      # Peak detection & validation
│
├── src/                           # Installable Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── make_dataset.py        # Data loading & label assignment
│   │   └── merge_datasets.py      # Acc + gyr stream merging
│   ├── features/
│   │   ├── remove_outliers.py     # Chauvenet & IQR methods
│   │   ├── build_features.py      # Feature extraction pipeline
│   │   └── count_repetitions.py   # Rep counting algorithm
│   ├── models/
│   │   ├── train_model.py         # Classifier training
│   │   └── predict_model.py       # Inference utilities
│   └── visualization/
│       └── visualize.py           # Signal plotting functions
│
├── reports/
│   ├── figures/                   # Signal plots, confusion matrices
│   └── metrics/                   # Model evaluation outputs
│
├── references/                    # HAR literature, sensor specs
│
├── setup.py                       # 📦 Package installation config
├── environment.yml                # Conda environment (pinned versions)
├── requirements.txt               # pip dependencies
└── README.md
```

---

##  Getting Started

### Option 1: Conda (Recommended)
```bash
git clone https://github.com/Georginh0/fitness_tracker-quantified_self-.git
cd fitness_tracker-quantified_self-

# Create environment with pinned dependencies
conda env create -f environment.yml
conda activate fitness_tracker

# Install as editable package (enables clean imports)
pip install -e .
```

### Option 2: pip
```bash
pip install -r requirements.txt
pip install -e .
```

### Run the Notebooks (in order)
```bash
jupyter notebook notebooks/01_EDA.ipynb
# Continue through 02 → 06
```

### Import Package in Your Own Code
```python
# Because of setup.py, this works from anywhere:
from src.features.build_features import extract_window_features
from src.features.count_repetitions import count_reps
from src.models.train_model import train_exercise_classifier
```

---

## Results Summary

| Capability | Performance |
|---|---|
| Exercise Classification Accuracy | **94.1%** |
| Rep Counting MAE | **0.34 reps/set** |
| Rep Counting ±1 Accuracy | **92%** |
| Rest Detection Accuracy | **~99%** |
| Exercises Classified | 5 compound lifts + Rest |
| Participants Tested | 5 (anonymised A–E) |
| Total Sensor Files Processed | 40+ sessions |

---

##  Future Improvements

- [ ] **1D-CNN model** — raw signal classification replacing hand-engineered features, targeting ~97% accuracy
- [ ] **LSTM rep counter** — sequence model replacing peak detection for varying-speed rep counting
- [ ] **Real-time inference** — WebSocket endpoint for live workout tracking during a session
- [ ] **Mobile app integration** — FastAPI backend → React Native frontend
- [ ] **More exercises** — Extend to Pull-up, Dumbbell Curl, Leg Press, Cable Row
- [ ] **Progressive overload tracking** — Session-level strength trend analysis and plateau detection
- [ ] **Anomaly detection** — Flag unusual rep patterns that may indicate fatigue or injury risk

---

##  License

This project is licensed under the MIT License — see [LICENCE](./LICENCE).

---

## 👤 Author

**George Dogo** — Data Scientist  
📧 George_dogo@aol.com | 🐙 [github.com/Georginh0](https://github.com/Georginh0)

*Built as part of a broader Quantified Self data science portfolio. If this helped your HAR work, please ⭐ the repo!*
