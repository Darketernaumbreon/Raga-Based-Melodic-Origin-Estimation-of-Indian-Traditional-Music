import numpy as np

class PitchNormalizer:
    """
    Normalizes the absolute F0 contour relative to the estimated Tonic (Sa).
    """
    def __init__(self):
        pass

    def normalize(self, f0, tonic_hz):
        """
        Converts absolute frequencies to relative cents above the tonic.
        Sa becomes exactly 0 cents.
        
        Args:
            f0 (np.ndarray): Array of F0 values in Hz
            tonic_hz (float): The estimated tonic frequency in Hz
            
        Returns:
            np.ndarray: Array of pitch values in cents relative to Sa
        """
        # Avoid log(0) for unvoiced frames (if any zeros remained)
        f0_safe = np.where(f0 == 0, 1e-5, f0)
        
        # 1200 cents in an octave
        cents = 1200 * np.log2(f0_safe / tonic_hz)
        
        # Wrap the cents to a single octave [0, 1200) 
        # because Pitch-Class Chromagrams are octave-invariant
        cents_wrapped = cents % 1200
        
        # If the original f0 was 0 (silence), keep it as NaN or handle it
        cents_wrapped[f0 == 0] = np.nan
        
        return cents_wrapped
