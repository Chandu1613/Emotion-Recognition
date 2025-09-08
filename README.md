# 🎤 Speech Emotion Recognition (SER)

## 📌 Abstract  
This project focuses on **Speech Emotion Recognition (SER)** using deep learning techniques.  
The goal is to classify human emotions (*happy, sad, angry, neutral, fear, disgust, surprised*) from speech audio.  

The system integrates multiple **public emotion datasets**, extracts **spectrogram** and **prosodic features**, trains deep learning models (**CNN + Hybrid**), and finally provides a **Streamlit-based UI** for real-time predictions from microphone input or uploaded `.wav` files.  

---

## 🎯 Problem Statement  
Human emotions play a vital role in communication. Recognizing emotions from speech can enhance applications like:  

- Human–computer interaction  
- Call center monitoring  
- Virtual assistants  
- Mental health analysis  

The challenge is that emotions vary across speakers, accents, and recording conditions.  
This project aims to build a **robust SER system** that generalizes across multiple datasets.  

---

## ⚙️ Tools & Technologies Used  

- **Programming Language**: Python  
- **Libraries**:  
  - Data Processing → `pandas`, `numpy`, `os`, `joblib`  
  - Audio Processing → `librosa`, `parselmouth`, `sounddevice`, `soundfile`  
  - Machine Learning & Deep Learning → `scikit-learn`, `tensorflow/keras`  
  - Visualization → `matplotlib`, `seaborn`, `tqdm`  
  - Deployment → `streamlit`  

- **Datasets Used**:  
  - [RAVDESS](https://zenodo.org/record/1188976)  
  - [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)  
  - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)  
  - [EmoDB](http://emodb.bilderbar.info/)  
  - [IESC](https://zenodo.org/record/3716120)  

---

## 📂 Project Structure  

Speech-Emotion-Recognition/
│
├── data_loading.py        # Load & clean emotion datasets → cleaned_data.csv
├── preprocessing.py       # Feature extraction (spectrogram + prosodic)
├── model_training.py      # Normalization, encoding, training prep
├── models/                # Saved CNN & Hybrid models
│   ├── cnn_stage1_model.keras
│   ├── hybrid_ser_model.keras
├── artifacts/             # Saved scalers & encoders
│   ├── spec_mean.npy
│   ├── spec_std.npy
│   ├── pros_scaler.pkl
│   ├── label_encoder.pkl
├── app.py                 # Streamlit UI for real-time inference
├── requirements.txt       # Dependencies
└── README.md              # Project documentation


---

## 🔬 Methodology  

### 1. Data Preparation  
- Load multiple datasets (RAVDESS, CREMA-D, TESS, EmoDB, IESC).  
- Standardize and merge labels into a unified CSV.  

### 2. Feature Extraction  
- **Spectrogram Features**: Mel-spectrograms, MFCCs, Chroma.  
- **Prosodic Features**: Pitch, energy, formants, jitter, shimmer.  
- **Data Augmentation**: Pitch shift, time stretch, noise injection, SpecAugment.  

### 3. Model Training  
- Normalize spectrograms and prosodic features.  
- Encode labels with `LabelEncoder` → one-hot vectors.  
- Train **CNN model** on spectrograms.  
- Train **Hybrid model** combining CNN features + prosodic features.  
- Save preprocessing artifacts (scalers, encoders).  

### 4. Inference & Deployment  
- Load trained models & artifacts.  
- Accept audio from **file upload** or **microphone recording**.  
- Extract features → normalize → predict → display result with probabilities.  

---

## 📊 Results / Output  

- **Supported Emotions**:  
  `Neutral, Happy, Sad, Angry, Fear, Disgust, Surprised`  

- **Hybrid model** performed better than CNN-only model (due to inclusion of prosodic features).  

- **Streamlit UI Features**:  
  - Upload `.wav` files  
  - Record live audio via microphone  
  - Real-time prediction with probability chart  

*(Add screenshots here if available)*  

---

## 📈 Future Scope  

- Extend to multilingual emotion recognition.  
- Improve robustness with Transformer-based models (e.g., wav2vec2, HuBERT).  
- Deploy as a REST API or integrate into chatbots/virtual assistants.  
- Add support for **real-time streaming inference**.  

---

## 🙌 Acknowledgments  

- Datasets:  
  [RAVDESS](https://zenodo.org/record/1188976),  
  [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),  
  [TESS](https://tspace.library.utoronto.ca/handle/1807/24487),  
  [EmoDB](http://emodb.bilderbar.info/),  
  [IESC](https://zenodo.org/record/3716120)  

- Libraries:  
  `librosa`, `parselmouth`, `tensorflow`, `streamlit`, `scikit-learn`, `matplotlib`, `joblib`  

---

## 📜 License  
This project is open-source and available under the [MIT License](LICENSE).  
