🫀 Sleep Apnea Detection using ECG Signals
Sleep apnea is a common but underdiagnosed disorder associated with cardiovascular risks. Traditional polysomnography is costly and time-consuming.
This project focuses on automated detection of sleep apnea from single-lead ECG signals using advanced deep learning architectures.

We propose a hybrid Temporal Convolutional Network (TCN) + Transformer model, which leverages dilated causal convolutions and multi-head self-attention to extract both local and global temporal dependencies in ECG signals.

🔹 Our best model (TCN+Transformer) achieved:

Accuracy: 90.59%

Precision: 88.9%

Recall: 87.2%

AUC-ROC: 0.9665

📂 Dataset

Source: PhysioNet Apnea-ECG Database

Records: 70 single-lead ECG signals (7–10 hrs each).

Sampling Rate: 100 Hz.

Annotations: Minute-wise apnea labels.

<p align="center"> <img src="images/Data Preprocessing Pipeline.png" width="650" alt="Data Preprocessing Pipeline"> </p>
⚙️ Preprocessing

Filtering: Bandpass FIR filter (3–45 Hz).

R-peak Detection: Hamilton algorithm (via BioSPPy
).

Segmentation: Sliding windows (5 min with 15s stride).

Multi-Scale Representation:

Full-scale (5 min → 900 samples)

Medium-scale (3 min → 540 samples)

Fine-scale (1 min → 180 samples)

🧠 Models Compared

We implemented and benchmarked the following models:

🌲 Random Forest (RF)

🔁 LSTM

🔁 Bi-LSTM

🌀 SE-MSCNN + Transformer

⚡ TCN + Transformer (proposed best model)

<p align="center"> <img src="images/TCN+Transformer Architecture.png" width="700" alt="TCN+Transformer Architecture"> </p>
📊 Results
Model	Accuracy	Precision	F1-Score	AUC-ROC
Random Forest	78.1%	69.1%	68.7%	0.833
LSTM	84.9%	81.2%	81.9%	0.923
Bi-LSTM	86.7%	82.5%	83.7%	0.938
SE-MSCNN + Transformer	90.2%	88.5%	87.6%	0.962
TCN + Transformer	90.59%	88.9%	87.7%	0.9665
<p align="center"> <img src="images/Confusion Matrix for TCN+Transformer Architecture.jpg" width="420" alt="Confusion Matrix - TCN+Transformer"> <img src="images/Confusion Matrix for SEMSCNN+Transformer Architecture.jpg" width="420" alt="Confusion Matrix - SE-MSCNN+Transformer"> </p>
🚀 Getting Started
🔧 Requirements

Python 3.8+

TensorFlow / PyTorch

NumPy, Pandas, Matplotlib

BioSPPy (for ECG preprocessing)

Scikit-learn

📥 Installation
git clone https://github.com/thrisha1217/apnea-ecg-analysis.git
cd apnea-ecg-analysis
pip install -r requirements.txt

▶️ Running the Notebook
jupyter notebook Sleep_Apnea_Project.ipynb


