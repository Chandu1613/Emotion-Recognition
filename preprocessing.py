import librosa
import numpy as np
import parselmouth
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=128, n_mfcc=40, max_len=200, seed=42):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)

    def pad_features(self, x, axis=1):
        if x.shape[axis] < self.max_len:
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, self.max_len - x.shape[axis])
            return np.pad(x, pad_width, mode="constant")
        else:
            return x.take(indices=range(self.max_len), axis=axis)

    def augment_audio_np(self, y, sr):
        """Lightweight waveform augmentations."""
        if self.rng.random() < 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=self.rng.choice([-3, -2, -1, 1, 2, 3]))
        if self.rng.random() < 0.5:
            rate = self.rng.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
        if self.rng.random() < 0.5:
            y = y + 0.005 * self.rng.standard_normal(len(y))
        return y

    def spec_augment_np(self, spec, max_f=12, max_t=24):
        """Simple SpecAugment (frequency + time masking)."""
        spec = spec.copy()
        f = self.rng.integers(0, max_f + 1)
        t = self.rng.integers(0, max_t + 1)
        if f > 0:
            f0 = self.rng.integers(0, max(1, spec.shape[0] - f))
            spec[f0:f0 + f, :] = 0
        if t > 0:
            t0 = self.rng.integers(0, max(1, spec.shape[1] - t))
            spec[:, t0:t0 + t] = 0
        return spec

    def extract_spectrogram_features(self, file_path, apply_aug=True, apply_specaug=True):
        """Extract mel, mfcc, chroma as stacked channels."""
        y, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        y, _ = librosa.effects.trim(y)

        if apply_aug:
            y = self.augment_audio_np(y, self.sample_rate)

        # Features
        mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = self.pad_features(mel_db)

        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        mfcc = self.pad_features(mfcc)

        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
        chroma = self.pad_features(chroma)

        # Align rows (freq bins) by padding
        max_rows = max(mel_db.shape[0], mfcc.shape[0], chroma.shape[0])
        mel_db = np.pad(mel_db, ((0, max_rows - mel_db.shape[0]), (0, 0)), mode="constant")
        mfcc = np.pad(mfcc, ((0, max_rows - mfcc.shape[0]), (0, 0)), mode="constant")
        chroma = np.pad(chroma, ((0, max_rows - chroma.shape[0]), (0, 0)), mode="constant")

        spec = np.stack([mel_db, mfcc, chroma], axis=-1)

        if apply_specaug and apply_aug:
            spec[..., 0] = self.spec_augment_np(spec[..., 0])  # mask only mel

        return spec.astype(np.float32)

    def extract_prosodic(self, file_path):
        """Extract pitch, energy, formants, jitter, shimmer."""
        snd = parselmouth.Sound(file_path)

        # Pitch (F0)
        pitch = snd.to_pitch()
        f0_vals = pitch.selected_array["frequency"]
        f0_vals = f0_vals[f0_vals > 0]
        f0_mean = float(np.mean(f0_vals)) if len(f0_vals) else 0.0
        f0_std = float(np.std(f0_vals)) if len(f0_vals) else 0.0

        # Intensity (Energy)
        intensity = snd.to_intensity()
        energy_vals = intensity.values.T.flatten()
        energy_mean = float(np.mean(energy_vals)) if len(energy_vals) else 0.0
        energy_std = float(np.std(energy_vals)) if len(energy_vals) else 0.0

        # Formants
        formants = snd.to_formant_burg()
        times = np.linspace(0, snd.duration, 50)
        f1 = [formants.get_value_at_time(1, t) for t in times]
        f2 = [formants.get_value_at_time(2, t) for t in times]
        f3 = [formants.get_value_at_time(3, t) for t in times]
        f1_mean, f2_mean, f3_mean = np.nanmean(f1), np.nanmean(f2), np.nanmean(f3)

        # Jitter & Shimmer
        try:
            pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
            jitter = float(parselmouth.praat.call(pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3))
            shimmer = float(parselmouth.praat.call((snd, pp), "Get shimmer (local)", 
                                                   0.0, 0.0, 0.0001, 0.02, 1.3, 1.6))
        except Exception:
            jitter, shimmer = 0.0, 0.0

        return np.nan_to_num(np.array([
            f0_mean, f0_std, energy_mean, energy_std,
            f1_mean, f2_mean, f3_mean, jitter, shimmer
        ], dtype=np.float32))

    def process_dataframe(self, df, file_col="file_path", label_col="emotion"):
        """Extract spectrogram + prosodic features from dataframe."""
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