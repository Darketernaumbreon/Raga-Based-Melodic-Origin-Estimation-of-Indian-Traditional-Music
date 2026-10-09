import numpy as np
import math

class SwaragramGenerator:
    """
    Constructs the 2D Pitch-Class Chromagram (Swaragram) from relative cents.
    """
    def __init__(self, num_bins=12, sigma=25.0):
        # 12 bins = 12 Swaras (semitones)
        self.num_bins = num_bins
        # Sigma for Gaussian smoothing (in cents). 25 cents is a quarter-tone standard deviation.
        self.sigma = sigma

    def generate(self, cents):
        """
        Generates the 2D Swaragram matrix.
        
        Args:
            cents (np.ndarray): Array of relative pitch in cents (length T)
            
        Returns:
            np.ndarray: 2D matrix of shape (num_bins, T)
        """
        T = len(cents)
        swaragram = np.zeros((self.num_bins, T), dtype=np.float32)
        
        # Center of each bin in cents (0, 100, 200, ..., 1100)
        bin_centers = np.arange(self.num_bins) * (1200 / self.num_bins)
        
        for t in range(T):
            if np.isnan(cents[t]):
                continue # Leave as zeros
                
            c = cents[t]
            
            # Calculate distance to each bin center (accounting for octave wrap-around)
            for b in range(self.num_bins):
                center = bin_centers[b]
                # Shortest distance on a circle of 1200 cents
                dist = min(abs(c - center), 1200 - abs(c - center))
                
                # Apply Gaussian activation
                swaragram[b, t] = math.exp(- (dist ** 2) / (2 * self.sigma ** 2))
                
        return swaragram
