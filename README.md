# 🔋 AI-Based Battery Management System (BMS)

> **End-of-Studies Internship Project @ Capgemini Engineering**  
> AI-powered State of Charge (SoC) prediction for Electric Vehicle batteries using time-series sensor data and deep learning.

---

## 📌 Overview

This project develops an AI solution to accurately predict the **State of Charge (SoC)** of Electric Vehicle (EV) lithium-ion batteries in real time. Traditional BMS systems rely on physics-based models that can degrade in accuracy under dynamic conditions. This solution replaces or augments them with machine learning and deep learning models trained on real driving cycle data — achieving near-perfect prediction accuracy.

The project covers the **full AI lifecycle**: data acquisition and preprocessing, model training and evaluation, and production deployment as an interactive dashboard and mobile application.

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| MAE    | **0.20%** |
| RMSE   | **0.44%** |
| R²     | **0.999** |

> Outperforms classical ML baselines (regression, gradient boosting) and achieves near-perfect fit on real-world driving cycle data.

---

## 🗂️ Project Structure

```
AI-Based-BMS/
│
├── data/                   # Raw and preprocessed driving cycle datasets
├── notebooks/              # Exploratory data analysis and model experimentation
├── models/                 # Trained model files and checkpoints
├── src/
│   ├── preprocessing/      # Data cleaning, normalization, augmentation, batching
│   ├── features/           # Feature engineering pipelines
│   ├── training/           # Model training and hyperparameter tuning scripts
│   └── evaluation/         # Metrics, plots, and model comparison
├── deployment/
│   ├── dashboard/          # Streamlit real-time SoC monitoring dashboard
│   └── mobile/             # Flutter mobile application
├── simulation/             # MATLAB/Simulink BMS model for validation
├── requirements.txt
└── README.md
```

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Loaded time-series sensor data from real EV driving cycles (current, voltage, temperature)
- Applied normalization, batching, and data augmentation to build high-quality training datasets
- Engineered temporal features to capture battery dynamics over driving sequences

### 2. Model Development & Evaluation
Trained and benchmarked multiple models:

| Model | Type |
|-------|------|
| **LSTM** | Deep Learning (best performer) |
| Gradient Boosting | Classical ML |
| Linear / Ridge Regression | Baseline |

All models evaluated on MAE, RMSE, and R² using held-out test sets from unseen driving cycles.

### 3. Deployment
- **Streamlit Dashboard** — real-time SoC monitoring with live sensor input and prediction visualization
- **Flutter Mobile App** — on-device SoC display with actionable battery health insights
- **MATLAB/Simulink** — physics-based BMS simulation used to cross-validate AI predictions

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Mobile App | Flutter (Dart) |
| Simulation | MATLAB / Simulink |
| Version Control | Git, GitHub |

---

## 🚀 Getting Started

### Prerequisites
```bash
Python >= 3.9
pip install -r requirements.txt
```

### Run the Dashboard
```bash
cd deployment/dashboard
streamlit run app.py
```

### Train a Model
```bash
cd src/training
python train.py --model lstm --epochs 100 --batch_size 32
```

### Evaluate Models
```bash
cd src/evaluation
python evaluate.py --model lstm
```

---

## 📊 Key Features

- ✅ Real-time SoC prediction from live sensor streams
- ✅ Multi-model comparison framework (DL vs classical ML)
- ✅ End-to-end preprocessing pipeline for time-series sensor data
- ✅ Interactive dashboard for live battery monitoring
- ✅ Mobile app for on-the-go insights
- ✅ Physics-based validation via MATLAB/Simulink BMS simulation

---




## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
