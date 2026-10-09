import numpy as np
import librosa

class TonicEstimator:
    """
    Estimates the Tonic (Sa) from the continuous F0 contour using a pitch histogram
    with octave-consistent refinement.
    """
    def __init__(self, bins_per_octave=36):
        # 36 bins per octave = 1/3 semitone resolution for finding the peak
        self.bins_per_octave = bins_per_octave

    def estimate(self, f0):
        """
        Estimates the fundamental frequency of the Tonic (Sa).

        Uses an octave-consistent refinement to prevent selecting a tonic
        that is one octave too high — a common error on high-register singers
        when using standard YIN.
        
        Args:
            f0 (np.ndarray): Array of F0 values in Hz
            
        Returns:
            float: Estimated tonic frequency in Hz
        """
        # Remove zeros and NaNs (silence/unvoiced frames)
        f0_valid = f0[~np.isnan(f0) & (f0 > 0)]
        if len(f0_valid) == 0:
            return 440.0 # Fallback

        # Convert to MIDI notes (logarithmic frequency)
        midi_notes = librosa.hz_to_midi(f0_valid)

        # Create a histogram of pitches (high resolution for accurate peak finding)
        hist, bin_edges = np.histogram(midi_notes, bins=100)

        # Find the primary peak (most frequently sung pitch)
        peak_bin = np.argmax(hist)
        tonic_midi = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0

        # --- Octave-Consistent Refinement ---
        # If a pitch one octave BELOW the detected peak has comparable energy (>80% of peak),
        # it is likely the true tonic (YIN sometimes locks onto the upper octave).
        tonic_midi_lower = tonic_midi - 12.0  # One octave below

        # Find histogram energy around the lower octave candidate (within ±0.5 semitones)
        lower_mask = (midi_notes >= tonic_midi_lower - 0.5) & (midi_notes < tonic_midi_lower + 0.5)
        upper_mask = (midi_notes >= tonic_midi - 0.5) & (midi_notes < tonic_midi + 0.5)

        energy_lower = np.sum(lower_mask)
        energy_upper = np.sum(upper_mask)

        if energy_upper > 0 and (energy_lower / energy_upper) >= 0.8:
            # The lower octave has comparable energy — prefer it as the true tonic
            tonic_midi = tonic_midi_lower

        # Convert back to Hz
        tonic_hz = librosa.midi_to_hz(tonic_midi)
        return float(tonic_hz)
