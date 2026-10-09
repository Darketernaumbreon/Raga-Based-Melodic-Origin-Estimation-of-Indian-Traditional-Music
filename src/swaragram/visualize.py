import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import os

def plot_swaragram_pipeline(y, sr, f0, tonic_hz, cents, swaragram, save_path="swaragram_pipeline.png"):
    """
    Generates a 4-step visualization of the Swaragram extraction pipeline for the Methods section.
    """
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    time_axis = librosa.times_like(f0, sr=sr, hop_length=512)
    
    # 1. Original F0
    axs[0].plot(time_axis, f0, color='blue', label='F0 Contour (Hz)')
    axs[0].set_title('Step 1: Original Fundamental Frequency (F0)')
    axs[0].set_ylabel('Frequency (Hz)')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # 2. Detected Sa
    axs[1].plot(time_axis, f0, color='blue', alpha=0.5)
    axs[1].axhline(y=tonic_hz, color='red', linestyle='--', linewidth=2, label=f'Detected Tonic (Sa) = {tonic_hz:.1f} Hz')
    axs[1].set_title('Step 2: Tonic Estimation (Sa)')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    # 3. Normalized Pitch (Relative Cents)
    axs[2].plot(time_axis, cents, color='green', marker='.', markersize=2, linestyle='none', label='Normalized Cents')
    axs[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axs[2].set_title('Step 3: Relative Pitch Normalization (Cents relative to Sa)')
    axs[2].set_ylabel('Cents')
    axs[2].set_ylim(-50, 1250)
    axs[2].set_yticks(np.arange(0, 1201, 200))
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()
    
    # 4. Final Swaragram
    img = librosa.display.specshow(swaragram, x_axis='time', y_axis='chroma', sr=sr, hop_length=512, ax=axs[3], cmap='magma')
    axs[3].set_title('Step 4: Final 12-Bin Pitch-Class Swaragram')
    axs[3].set_ylabel('Swara (Pitch Class)')
    
    # Adjust y-axis ticks for Indian Swaras (approximate mapping)
    swaras = ['Sa', 're', 'Re', 'ga', 'Ga', 'Ma', 'ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni']
    axs[3].set_yticks(np.arange(12) + 0.5)
    axs[3].set_yticklabels(swaras)
    
    fig.colorbar(img, ax=axs[3], format='%+2.0f')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
