import librosa
import numpy as np
import parselmouth
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=128, n_mfcc=40, max_len=200):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.max_len = max_len

    def pad_features(self, x):
        """Pad/Truncate features along time axis"""
        if x.shape[1] < self.max_len:
            return np.pad(x, ((0, 0), (0, self.max_len - x.shape[1])), mode='constant')
        else:
            return x[:, :self.max_len]

    def augment_audio(self, y, sr):
        """Optional data augmentation"""
        if np.random.rand() < 0.3:
            y = librosa.effects.pitch_shift(y, sr, n_steps=np.random.choice([-2, -1, 1, 2]))
        if np.random.rand() < 0.3:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
        if np.random.rand() < 0.3:
            y = y + 0.005 * np.random.randn(len(y))
        return y

    def extract_spectrogram_features(self, file_path):
        y, _ = librosa.load(file_path, sr=self.sample_rate)
        y = librosa.to_mono(y)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        y, _ = librosa.effects.trim(y)

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = self.pad_features(mel_db)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        mfcc = self.pad_features(mfcc)

        # Match MFCC and Mel dimensions
        if mfcc.shape[0] < mel_db.shape[0]:
            pad_rows = mel_db.shape[0] - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_rows), (0, 0)), mode='constant')
        elif mfcc.shape[0] > mel_db.shape[0]:
            mfcc = mfcc[:mel_db.shape[0], :]

        return np.stack([mel_db, mfcc], axis=-1)

    def extract_prosodic(self, file_path):
        snd = parselmouth.Sound(file_path)

        # Pitch
        pitch = snd.to_pitch()
        f0_vals = pitch.selected_array['frequency']
        f0_vals = f0_vals[f0_vals > 0]
        f0_mean, f0_std = np.mean(f0_vals), np.std(f0_vals)

        # Energy
        intensity = snd.to_intensity()
        energy_vals = intensity.values.T.flatten()
        energy_mean, energy_std = np.mean(energy_vals), np.std(energy_vals)

        # Formants
        formants = snd.to_formant_burg()
        times = np.linspace(0, snd.duration, 50)
        f1 = [formants.get_value_at_time(1, t) for t in times]
        f2 = [formants.get_value_at_time(2, t) for t in times]
        f3 = [formants.get_value_at_time(3, t) for t in times]

        return np.nan_to_num(np.array([
            f0_mean, f0_std, energy_mean, energy_std,
            np.nanmean(f1), np.nanmean(f2), np.nanmean(f3)
        ]))

    def process_dataframe(self, df, file_col="file_path", label_col="emotion"):
        """Extracts features for all rows in a DataFrame"""
        X_spec, X_prosodic, y = [], [], []

        print("Extracting features...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                spec_feat = self.extract_spectrogram_features(row[file_col])
                pros_feat = self.extract_prosodic(row[file_col])

                X_spec.append(spec_feat)
                X_prosodic.append(pros_feat)
                y.append(row[label_col])
            except Exception as e:
                print(f"Error: {row[file_col]} → {e}")

        return (
            np.array(X_spec, dtype=np.float32),
            np.nan_to_num(np.array(X_prosodic, dtype=np.float32)),
            np.array(y)
        )