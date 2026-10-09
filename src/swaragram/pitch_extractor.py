import librosa
import numpy as np
from scipy.signal import medfilt

class PitchExtractor:
    """
    Extracts the continuous Fundamental Frequency (F0) contour from raw audio 
    using the YIN algorithm with median filtering for temporal smoothing.
    """
    def __init__(self, sr=22050, fmin="C2", fmax="C7", hop_length=512):
        self.sr = sr
        self.fmin = librosa.note_to_hz(fmin)
        self.fmax = librosa.note_to_hz(fmax)
        self.hop_length = hop_length

    def extract(self, y):
        """
        Extracts the F0 contour.
        
        Args:
            y (np.ndarray): Audio time series
            
        Returns:
            np.ndarray: Array of F0 values in Hz. Unvoiced frames are interpolated.
        """
        # Swaragrams are Pitch-Class Chromagrams (octave invariant).
        # Therefore, the octave errors that standard YIN is prone to (compared to pYIN) 
        # DO NOT MATTER because they wrap to the exact same pitch-class bin (modulo 1200 cents).
        # Switching from pyin to yin gives a 10x-20x speedup.
        
        try:
            f0 = librosa.yin(
                y, 
                sr=self.sr, 
                fmin=self.fmin, 
                fmax=self.fmax, 
                hop_length=self.hop_length
            )
            
            # Since standard YIN doesn't return a voiced_flag, we mask unvoiced frames using RMS energy
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=self.hop_length)[0]
            # Threshold at 1% of max energy (approx -40dB)
            threshold = 0.01 * np.max(rms) if np.max(rms) > 0 else 0
            
            # Mask silent/unvoiced frames with NaN
            f0[rms < threshold] = np.nan
            
        except Exception as e:
            # Fallback
            return np.zeros(len(y) // self.hop_length + 1)
        
        # Interpolate unvoiced frames (NaNs) to maintain a continuous pitch contour
        if np.isnan(f0).any():
            # If the entire array is NaN (silence), return zeros
            if np.isnan(f0).all():
                return np.zeros_like(f0)
                
            # Otherwise interpolate
            valid_indices = np.where(~np.isnan(f0))[0]
            nan_indices = np.where(np.isnan(f0))[0]
            f0[nan_indices] = np.interp(nan_indices, valid_indices, f0[valid_indices])

        # Apply median filter to reduce frame-to-frame jitter (kernel=7 ≈ 160ms at hop=512, sr=22050)
        # This is a standard signal-processing refinement; it does not change musical content.
        f0 = medfilt(f0, kernel_size=7)
            
        return f0
