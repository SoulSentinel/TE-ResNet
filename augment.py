import numpy as np
import librosa
import random

class AudioAugmenter:
    def __init__(self, sr=16000):
        self.sr = sr

    def add_gaussian_noise(self, audio, noise_factor=0.005):
        """Add Gaussian noise to the audio"""
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio

    def add_snr_noise(self, audio, min_snr_db=5, max_snr_db=20):
        """Add noise with a random SNR"""
        snr_db = random.uniform(min_snr_db, max_snr_db)
        snr = 10 ** (snr_db / 10)
        audio_power = np.sum(audio ** 2) / len(audio)
        noise_power = audio_power / snr
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        augmented_audio = audio + noise
        return augmented_audio

    def time_shift(self, audio, shift_max=0.1):
        """Shift the audio in time"""
        shift = int(random.uniform(-shift_max, shift_max) * len(audio))
        augmented_audio = np.roll(audio, shift)
        if shift > 0:
            augmented_audio[:shift] = 0
        else:
            augmented_audio[shift:] = 0
        return augmented_audio

    def pitch_shift(self, audio, n_steps=4):
        """Shift the pitch of the audio"""
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)

    def time_stretch(self, audio, rate=1.2):
        """Stretch the time of the audio"""
        return librosa.effects.time_stretch(audio, rate=rate)

    def augment(self, audio):
        """Apply a random augmentation to the audio"""
        augmentation_functions = [
            self.add_gaussian_noise,
            self.add_snr_noise,
            self.time_shift,
            self.pitch_shift,
            self.time_stretch
        ]
        augmentation_function = random.choice(augmentation_functions)
        return augmentation_function(audio)

