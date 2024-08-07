import librosa
import numpy as np
from scipy.fftpack import dct

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mfcc=24, n_fft=1024, hop_length=256, n_mels=80, n_cqt=84):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_cqt = n_cqt
        
    def load_audio(self, file_path):
        """
        Load an audio file.
        
        Args:
        file_path (str): Path to the .wav file.
        
        Returns:
        tuple: (audio_time_series, sample_rate)
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr


    def extract_mfcc(self, audio):
        """
        Extract MFCC features from an audio time series.

        Args:
        audio (np.array): Audio time series.

        Returns:
        np.array: MFCC features with shape (126, n_mfcc * 3)
        """
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc,
                                    n_fft=self.n_fft, hop_length=self.hop_length)

        if mfcc.shape[1] < 2:
            mfcc = np.repeat(mfcc, 2, axis=1)

        delta_mfcc = librosa.feature.delta(mfcc, width=3)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=3)

        features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
        features = features.T

        target_frames = 126
        if features.shape[0] < target_frames:
            pad_width = ((0, target_frames - features.shape[0]), (0, 0))
            features = np.pad(features, pad_width, mode='constant')
        elif features.shape[0] > target_frames:
            features = features[:target_frames, :]

        return features

    def extract_cqcc(self, file_path):
        """
        Extract CQCC features from a .wav file.

        Args:
        file_path (str): Path to the .wav file.

        Returns:
        np.array: CQCC features with shape (126, n_cqcc * 3)
        """
        audio, _ = self.load_audio(file_path)

        C = np.abs(librosa.cqt(audio, sr=self.sample_rate, n_bins=self.n_cqt,
                               hop_length=self.hop_length))

        C_log = np.log(C + 1e-6)

        cqcc = dct(C_log, axis=0, type=2, norm='ortho')[:self.n_mfcc]

        delta_cqcc = librosa.feature.delta(cqcc)
        delta2_cqcc = librosa.feature.delta(cqcc, order=2)

        features = np.concatenate([cqcc, delta_cqcc, delta2_cqcc], axis=0)
        features = features.T

        target_frames = 126
        if features.shape[0] < target_frames:
            pad_width = ((0, target_frames - features.shape[0]), (0, 0))
            features = np.pad(features, pad_width, mode='constant')
        elif features.shape[0] > target_frames:
            features = features[:target_frames, :]

        return features

    def extract_lps(self, file_path):
        """
        Extract Log Power Spectrum (LPS) features from a .wav file.

        Args:
        file_path (str): Path to the .wav file.

        Returns:
        np.array: LPS features with shape (126, n_fft//2 + 1)
        """
        audio, _ = self.load_audio(file_path)

        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)

        S = np.abs(D)**2

        lps = librosa.power_to_db(S, ref=np.max)
        lps = lps.T

        target_frames = 126
        if lps.shape[0] < target_frames:
            pad_width = ((0, target_frames - lps.shape[0]), (0, 0))
            features = np.pad(lps, pad_width, mode='constant')
        elif lps.shape[0] > target_frames:
            features = lps[:target_frames, :]

        return features
