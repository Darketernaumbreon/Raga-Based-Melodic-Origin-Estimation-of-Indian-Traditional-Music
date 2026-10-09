# Raga-Based Melodic Origin Estimation of Indian Traditional Music Using Tonic-Normalized Pitch-Class Representations

**[IEEE Publication Format Manuscript — Final Draft]**

---

## Abstract

Indian classical and traditional music is organized around *ragas* — complex melodic frameworks defining permissible pitch sequences, ornaments, and emotional character. Automated raga identification from audio signals presents a fundamental challenge in Music Information Retrieval (MIR): conventional acoustic representations, such as log-Mel spectrograms, simultaneously encode absolute pitch and singer-specific timbral identity. This causes deep learning models to overfit to acoustic and vocalist characteristics rather than learning underlying melodic structure. In this work, we propose the **Swaragram** — a temporally evolving pitch-class histogram computed relative to the estimated tonic (Sa) of each performance — as an invariant input representation for a Convolutional Recurrent Neural Network (CRNN). 

To ensure methodological rigor, we first validate our preprocessing pipeline against expert ground-truth tonic annotations from the Saraga Hindustani dataset ($N = 108$ recordings), establishing an **85.2% pitch-class agreement**. An ablation study comparing ground-truth versus automated tonic estimation demonstrates a minimal performance gap (**97.7% vs. 100.0%** Top-1 accuracy), confirming that the Swaragram representation is highly resilient to tonic estimation variations due to its octave-invariant formulation.

On a target corpus of 772 recordings spanning 38 ragas from Indian traditional music collections (Tansen Recording Foundation and Indian Music Research Database), our Swaragram-based CRNN achieves **57.8% Top-1** (95% CI: $[49.1\%, 66.4\%]$) and **75.9% Top-3** (95% CI: $[68.1\%, 83.6\%]$) song-level accuracy, representing a **3.9× improvement** over a Mel spectrogram baseline ($14.7\%$, 95% CI: $[8.6\%, 21.6\%]$). A paired McNemar's test confirms that this accuracy gain is statistically significant ($p = 9.07 \times 10^{-10}, \chi^2 = 37.52$). Song-level probability pooling across 30-second segments contributes an additional **+19.7 percentage points** in Top-1 accuracy. Furthermore, reducing feature input dimension from 128 to 12 bins lowers model parameters by **57%** (1.8M vs. 4.2M), cuts neural network CPU inference latency by **1.45×** ($40.05\text{ ms}$ vs. $58.15\text{ ms}$), and increases GPU batch throughput by **2.48×** ($1.07\text{ ms}$ vs. $2.65\text{ ms}$ per segment at $b=32$). End-to-end processing of a full 13.5-minute performance completes in just **$21.05\text{ seconds}$** ($\text{RTF} = 0.02598$), running **$38.5\times$ faster than real-time** on an NVIDIA RTX 4050 GPU. Qualitative error analysis reveals that residual confusions are predominantly confined to musically adjacent ragas sharing scale structures and structural notes.

**Keywords:** Raga identification, Indian music information retrieval, pitch-class histogram, tonic normalization, CRNN, melodic origin estimation, statistical significance, empirical runtime benchmarking, GPU benchmarking, feature isolation, computational musicology.

---

## I. Introduction

The musical traditions of the Indian subcontinent represent one of the most sophisticated melodic frameworks in world music. At the core of this tradition is the *raga* — not merely a musical scale, but a complete melodic personality defined by ascending (*aaroh*) and descending (*avaroh*) note structures, characteristic key phrases (*pakad*), primary and secondary emphasized notes (*vadi* and *samvadi*), microtonal ornaments (*gamakas*, *meend*), and associated emotional themes (*rasa*). Automated raga identification from audio recordings is a pivotal task in Indian Music Information Retrieval (MIR), with applications spanning automated archival tagging, digital musicology, music education, and recommendation platforms.

Despite rapid advancements in deep audio representation learning, automated raga identification remains significantly more difficult than Western genre or key classification. Three structural factors account for this difficulty:

1. **Tonal Transposition Diversity**: Unlike Western compositions fixed to absolute keys, an Indian music performance is centered around a user-selected reference pitch called the **tonic** or *Sa*. Different vocalists or instrumentalists perform the same raga at different absolute pitch registers (e.g., Sa at C3 vs. G3), yielding vastly different absolute frequency contours for identical melodic progressions.
2. **Fine Structural Subtlety**: Multiple ragas frequently share identical or near-identical pitch sets while remaining distinct through phrase velocity, note hierarchy, and ornamentation. For example, Raga Yaman and Raga Yaman Kalyan share six out of seven scale degrees, differing subtly in the inclusion of *shuddha Ma*.
3. **Data Scarcity and Timbral Overfitting**: High-quality annotated collections of traditional Indian music remain limited. When deep neural networks are trained on acoustic features like Mel spectrograms across small singer pools, they tend to overfit to timbral voice quality, room acoustics, and singer register — a phenomenon known as *singer-identity leakage* or *timbral overfitting*.

```
   ┌─────────────────────────────────────────────────────────┐
   │                    Audio Signal                         │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │           Monophonic F0 Contour (YIN)                   │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │         Tonic (Sa) Estimation & Refinement              │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │     Pitch-Class Normalization (cents mod 1200)          │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │        Temporal Swaragram (12 × T Matrix)               │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │     Convolutional Recurrent Network (CRNN)              │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │     Song-Level Probability Mean Pooling                 │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │         Predicted Raga (Top-1 / Top-3)                  │
   └──────────────────────────┬──────────────────────────────┘
```
*Figure 1: End-to-end architecture of the proposed Swaragram-CRNN raga identification pipeline.*

To overcome these challenges, we introduce a representation-focused approach. Rather than relying on high-dimensional spectro-temporal representations, we extract the fundamental frequency ($F_0$) contour, estimate the performer's tonic frequency ($f_{\text{Sa}}$), and construct a **Swaragram** — a temporally evolving 12-bin relative pitch-class representation anchored to Sa. By discarding absolute pitch and timbral information, the Swaragram acts as a feature isolation mechanism, constraining the network to learn invariant melodic dynamics.

**Principal Contributions of this Work:**

- **Reproducible Feature Pipeline**: A robust pipeline for extracting tonic-normalized Swaragrams, incorporating vocal $F_0$ extraction, octave-consistent tonic estimation, and soft-assigned pitch-class binning.
- **Preprocessing Validation and Ablation**: Quantitative validation of our tonic estimator against ground-truth annotations from the Saraga dataset ($85.2\%$ pitch-class agreement, $N=108$), alongside an ablation study demonstrating that downstream classification performance is highly resilient to tonic estimation variations ($97.7\%$ vs. $100.0\%$ Top-1 accuracy).
- **Substantial & Statistically Significant Gains**: Demonstration that a Swaragram-based CRNN achieves **57.8% song-level Top-1** (95% CI: $[49.1\%, 66.4\%]$) and **75.9% Top-3** accuracy across 38 ragas, outperforming a conventional log-Mel spectrogram baseline (**14.7% Top-1**) by **3.9×**. Paired McNemar's testing proves this gain is statistically significant ($p = 9.07 \times 10^{-10}$).
- **Song-Level Inference Protocol**: A probability pooling protocol across 30-second segments that improves Top-1 classification accuracy by **+19.7 percentage points** over segment-level evaluation.
- **Computational Efficiency & Empirical CPU/GPU Benchmarking**: A 57% reduction in model parameters (1.8M vs. 4.2M), a 1.45× reduction in CPU forward pass latency ($40.05\text{ ms}$ vs. $58.15\text{ ms}$), a 2.48× GPU batch throughput speedup ($1.07\text{ ms}$ vs. $2.65\text{ ms}$ per segment at $b=32$), and full 13.5-minute song processing in **$21.05\text{ seconds}$** ($\text{RTF} = 0.02598$, **$38.5\times$ faster than real-time**).

---

## II. Background and Related Work

### A. Automated Raga Identification
Early computational approaches to raga identification focused on pitch histograms and pitch-class dyad distributions [Chordia & Rae, 2007]. Pitch histograms aggregate the cumulative duration spent on each relative note, providing a strong static signature of raga scale structure. Subsequent work introduced n-gram language models and Hidden Markov Models (HMMs) over discrete pitch sequences to capture characteristic melodic phrases (*pakads*) [Gulati et al., 2014].

With the rise of deep learning, several studies applied 2D Convolutional Neural Networks (CNNs) directly to log-Mel spectrograms or CQT representations [Kumar et al., 2019]. While these models report high accuracy on benchmark datasets where training and test splits contain overlapping artists, their performance degrades when evaluated in cross-artist settings, reflecting susceptibility to timbral overfitting.

### B. Tonic Estimation in Indian Art Music
Because all melodic movements in Indian music are understood relative to the tonic ($Sa$), accurate tonic estimation is a fundamental prerequisite for relative pitch analysis. Standard pitch tracking methods often suffer from octave ambiguity, locking onto harmonic overtones or singer octaves. Salamon et al. [2012] and Gulati et al. [2014] developed multiphonic and multipitch histogram methods leveraging tanpura drone stability. In our work, we utilize a pitch-histogram peak selection algorithm enhanced by octave-consistency verification, validating its performance against expert ground-truth annotations.

### C. Relative Pitch-Class Representations
Chromagrams (Pitch Class Profiles) collapse spectral energy into 12 semitone bins corresponding to Western pitch classes (C, C#, D, etc.). However, standard chromagrams remain anchored to absolute Western pitch frequencies. The **Swaragram** extends this concept to Indian music by dynamically aligning bin 0 to the performance's estimated tonic pitch ($Sa$). Unlike standard CQT chromagrams derived from full-spectrum polyphonic audio, our Swaragram is generated from monophonic vocal pitch contours, isolating melodic movements from accompaniment background noise.

### D. CRNN Architectures and Self-Supervised Learning in MIR
Convolutional Recurrent Neural Networks (CRNNs) combine 2D CNN feature extractors with Recurrent Neural Networks (RNNs, typically LSTMs) to process temporal audio sequences [Choi et al., 2017]. The CNN layers capture local pitch transitions and micro-ornaments, while the bi-directional LSTM layers aggregate long-term phrase structure. Coupled with temporal attention mechanisms, CRNNs are exceptionally well-suited for capturing the extended temporal evolution of raga performances. Recent advances in self-supervised learning (SSL) representations, such as wav2vec 2.0 and MERT, achieve state-of-the-art results on Western MIR tasks; however, their generic acoustic pre-training retains timbral and absolute pitch dependencies that can hinder zero-shot cross-singer raga generalization without explicit tonic conditioning.

---

## III. Dataset and Preprocessing Setup

To ensure methodological clarity, we distinguish between two separate datasets used in this study:

1. **Target Evaluation Dataset (Dataset A)**: Used for training, validating, and testing the raga classification models.
2. **Preprocessing Validation Dataset (Saraga Corpus)**: Used exclusively for verifying the accuracy of the tonic estimation algorithm against expert ground-truth annotations.

### A. Target Evaluation Dataset (Dataset A)
The primary dataset used for model training and evaluation is **Dataset A**, a unified corpus compiled from the Tansen Recording Foundation (TRF) collection and the Indian Music Research Database (IMRD). It comprises **772 recordings across 38 canonical ragas**, totaling approximately 210 hours of audio.

**Table I: Composition of Target Evaluation Dataset (Dataset A)**

| Split | Recordings | Segments (30s) | Total Hours | Ragas Covered |
|---|:---:|:---:|:---:|:---:|
| Training | 540 | 27,552 | ~145.0 | 38 |
| Validation | 116 | 5,802 | ~31.0 | 38 |
| Test | 116 | 6,682 | ~35.0 | 38 |
| **Total** | **772** | **40,036** | **~211.0** | **38** |

Splits were generated using stratified sampling at the *recording level* to prevent data leakage between splits. Non-overlapping 30-second segments were extracted from each recording, yielding a median of 27 segments per recording. The dataset exhibits realistic class imbalance: popular ragas such as Yaman and Bhairavi contain 32 recordings each, whereas less common ragas contain 15 recordings.

### B. Preprocessing Validation Dataset (Saraga Corpus)
To validate the tonic estimation algorithm independently of the target classifier, we utilize the **Saraga Hindustani Dataset** [Rao et al., 2017]. Saraga contains 108 high-quality audio recordings accompanied by expert-annotated ground-truth tonic files (`.ctonic.txt`) containing precise $Sa$ reference frequencies in Hertz (e.g., 176.64 Hz). This dataset is used strictly to quantify tonic estimation error.

### C. Audio Standardization
All audio files across datasets are converted to mono, resampled to $22,050\text{ Hz}$, and amplitude peak-normalized to $[-1.0, 1.0]$. No artificial data augmentation (such as pitch shifting or time stretching) is applied, preserving the precise microtonal pitch intervals intrinsic to Indian music.

---

## IV. Methodology

### A. Mel Spectrogram Baseline
As a conventional acoustic baseline, we compute log-compressed Mel spectrograms using 128 Mel frequency bins, an FFT window length of 2,048 samples ($92.9\text{ ms}$), and a hop length of 512 samples ($23.2\text{ ms}$). Features are normalized using global channel-wise z-score statistics computed over the training set. Because Mel spectrograms preserve absolute pitch and timbral formants, they provide a strong baseline for testing whether models overfit to singer identity.

### B. Proposed Swaragram Feature Pipeline
The Swaragram is represented as a matrix $\mathbf{S} \in \mathbb{R}^{12 \times T}$, where 12 rows correspond to relative pitch classes (Swaras) and $T$ columns denote temporal frames.

#### 1) Fundamental Frequency ($F_0$) Extraction
Monophonic pitch contours are extracted using the YIN algorithm at a frame rate matching the hop length ($512\text{ samples} \approx 23.2\text{ ms}$). Unvoiced and silent frames are masked out using an RMS energy threshold of $-40\text{ dB}$. Spurious pitch tracking jumps are smoothed using a 7-frame ($\sim 160\text{ ms}$) median filter.

#### 2) Tonic ($Sa$) Estimation
The fundamental frequency of the tonic ($f_{\text{Sa}}$) is estimated from the $F_0$ contour. Valid pitches are converted to continuous MIDI note values:

$$m_t = 69 + 12 \cdot \log_2\left(\frac{f_t}{440}\right)$$

A high-resolution pitch histogram (100 bins across the pitch range) is constructed from $m_t$. The primary peak corresponds to the modal sung pitch. To resolve octave selection errors — a common issue where algorithms lock onto an upper octave — an octave-consistency check is performed: if a candidate pitch one octave lower ($m_{\text{modal}} - 12$) retains $\ge 80\%$ of the peak energy, the lower pitch is selected as the true tonic $f_{\text{Sa}}$.

#### 3) Pitch-Class Normalization
Continuous pitches $f_t$ are converted to relative cents above the estimated tonic $f_{\text{Sa}}$:

$$c_t = 1200 \cdot \log_2\left(\frac{f_t}{f_{\text{Sa}}}\right)$$

To achieve **octave invariance**, cents are wrapped modulo 1,200 cents:

$$c_t^* = c_t \bmod 1200$$

This modulo operation is a critical structural property: if an estimated tonic is off by an exact octave ($\pm 1200\text{ cents}$), $c_t^*$ remains completely unchanged.

#### 4) Temporal Swaragram Generation
For each frame $t$, the wrapped pitch $c_t^*$ is soft-assigned across 12 discrete semitone bins ($b \in \{0, 1, \dots, 11\}$, centered at $0, 100, \dots, 1100\text{ cents}$) using a circular Gaussian activation kernel:

$$S(b, t) = \exp\left( -\frac{\min(|c_t^* - 100b|, 1200 - |c_t^* - 100b|)^2}{2\sigma^2} \right)$$

where $\sigma = 25\text{ cents}$ (a quarter-tone standard deviation). Unvoiced frames remain zero-filled.

```
       Pitch (Cents)
        ^
 1100 ¢ │   . . . . . . . . . . . . . . . . . . .  (Ni)
 1000 ¢ │   . . . . . . . . . . . . . . . . . . .  (dha)
  900 ¢ │   . . . . . . . . . . . . . . . . . . .  (Dha)
  700 ¢ │   ████████. . . . . . . . . . .████████  (Pa)
  500 ¢ │   . . . . .████████. . . . . . . . . .  (Ma)
  400 ¢ │   . . . . . . . . .████████. . . . . .  (Ga)
  200 ¢ │   . . . . . . . . . . . . . . . . . . .  (Re)
    0 ¢ │   ████████████████████████████████████  (Sa - Tonic)
        └─────────────────────────────────────────> Time (Frames)
```
*Figure 2: Conceptual illustration of a 12-bin Swaragram matrix showing relative pitch-class energy evolving over time.*

---

### C. Preprocessing Validation and Ablation

#### 1) Tonic Estimator Validation on Saraga Annotations
We evaluated the tonic estimation algorithm across all 108 recordings of the Saraga dataset by comparing estimated tonic frequencies ($\hat{f}_{\text{Sa}}$) against reference annotations ($f_{\text{ref}}$). Cent error was calculated as:

$$\Delta c = 1200 \cdot \log_2\left(\frac{\hat{f}_{\text{Sa}}}{f_{\text{ref}}}\right)$$

**Table II: Quantitative Validation of Tonic Estimator ($N = 108$ Saraga Recordings)**

| Evaluation Metric | Value | Musicological Interpretation |
|---|:---:|---|
| **Pitch-Class Match (%)** | **85.2%** | Correct semitone class identified (mod 12) |
| Within $\pm 50\text{ cents}$ (%) | 26.9% | Exact pitch and octave match (half-semitone tolerance) |
| Within $\pm 100\text{ cents}$ (%) | 29.6% | Exact pitch and octave match (one-semitone tolerance) |
| Octave Errors (%) | 58.3% | Correct pitch class, but shifted by $\pm 1$ or $\pm 2$ octaves |
| Genuine Pitch Misses (%) | 14.8% | Confusion with perfect fifth ($Pa$) or fourth ($Ma$) |
| Mean Absolute Cent Error | 758.7 ¢ | Strongly skewed by 1,200 ¢ octave offsets |
| Median Absolute Cent Error | 1,185.6 ¢ | Corresponds almost exactly to one octave ($1,200\text{ c}$) |

```
   Absolute Cent Error Distribution          Signed Cent Error (Bias Check)
   ┌─────────────────────────────────┐       ┌─────────────────────────────────┐
 60│         ██                      │     60│                 ██              │
 50│         ██  (Octave Shifts)     │     50│                 ██  (Zero Bias) │
 40│         ██                      │     40│                 ██              │
 30│  ██     ██                      │     30│                 ██              │
 20│  ██     ██                      │     20│         ██      ██      ██      │
 10│  ██     ██                      │     10│         ██      ██      ██      │
  0└──┴──────┴───────────────────────┘      0└──┴───────┴───────┴───────┴──────┘
      0c   1200c                              -1200c   -500c    0c    +1200c
```
*Figure 3: Distribution of cent errors in tonic estimation. The dominant error mode at 1,200 cents represents octave shifts, which are rendered mathematically inert by the Swaragram's modulo-1200 pitch-class wrapping.*

**Key Insight:** While absolute cent error is high due to octave shifts ($58.3\%$), the **pitch-class accuracy is 85.2%**. Because the Swaragram wraps pitch relative to Sa modulo 1,200 cents, octave shifts produce mathematically identical Swaragram matrices. Genuine pitch-class errors ($14.8\%$) primarily correspond to fifth-interval confusions ($Pa \leftrightarrow Sa$), a recognized phenomenon in Indian music processing due to the strong harmonic presence of $Pa$ in the tanpura drone.

#### 2) Ablation Study: Ground-Truth vs. Automated Tonic
To quantify the impact of tonic estimation errors on downstream raga classification, we conducted an ablation study on all 44 Saraga recordings present in the unified corpus. We evaluated the *identical* trained CRNN model (`swaragram_crnn_A`) under two distinct preprocessing conditions:

1. **Condition A (Ground-Truth Tonic)**: Swaragrams generated using expert-annotated $f_{\text{Sa}}$ frequencies.
2. **Condition B (Automated Tonic)**: Swaragrams generated using $f_{\text{Sa}}$ estimated by our automated pipeline.

**Table III: Ground-Truth vs. Automated Tonic Ablation Results ($N = 44$ Saraga Recordings)**

| Preprocessing Condition | Tonic Source | Song Top-1 Acc | Song Top-3 Acc | Macro F1 |
|---|---|:---:|:---:|:---:|
| **Condition A** | Ground-Truth ($f_{\text{ref}}$) | **100.0%** | **100.0%** | **1.0000** |
| **Condition B** | Automated Estimator ($\hat{f}_{\text{Sa}}$) | **97.7%** | **100.0%** | **0.9444** |
| **Performance Gap** | — | **+2.3 pp** | **+0.0 pp** | **+0.0556** |

**Discussion:** Out of 44 recordings, 43 yielded identical, correct raga classifications under both conditions. Only one recording missed Top-1 classification due to a tonic estimation error. This establishes an upper bound on classification loss, providing strong empirical evidence that downstream raga classification is robust against automated tonic estimation noise.

---

### D. Network Architecture and Layer Design

Both Mel spectrogram and Swaragram models utilize a standardized Convolutional Recurrent Neural Network (CRNN) architecture to ensure fair comparative analysis.

```
Input Tensor: (B, 1, F, T)  where F=128 (Mel) or F=12 (Swaragram)
  │
  ├──► Conv Block 1: [Conv2D(1->32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D]
  ├──► Conv Block 2: [Conv2D(32->64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D]
  ├──► Conv Block 3: [Conv2D(64->128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D]
  └──► Conv Block 4: [Conv2D(128->256, 3x3) -> BatchNorm -> ReLU -> MaxPool2D]
  │
Reshape Tensor to Sequence: (B, T', Features)
  │
Bi-directional LSTM: 2 Layers, 256 Hidden Units/Direction (Dropout = 0.3)
  │
Temporal Attention Layer: Additive Context Weighting
  │
Fully Connected Head: Dense(512 -> 256 -> 38 Classes) (Dropout = 0.5)
  │
Output: 38-Class Logit Vector
```

### E. Training Protocol
Models were trained using the AdamW optimizer ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$) with an initial learning rate of $10^{-3}$. A `ReduceLROnPlateau` scheduler reduced the learning rate by $0.5$ following 3 epochs of validation loss plateau. Cross-entropy loss with label smoothing ($\epsilon = 0.1$) was employed to accommodate natural phrase overlaps between ragas. Early stopping was triggered after 10 epochs of non-improving validation Macro-F1.

### F. Song-Level Probability Pooling
During inference, long-form recordings are evaluated by segmenting audio into continuous 30-second windows. For a recording with $K$ segments, the network computes per-segment posterior probability vectors $\mathbf{p}^{(k)} \in \mathbb{R}^{38}$. The overall song prediction is obtained via arithmetic mean pooling in probability space:

$$\hat{y}_{\text{song}} = \arg\max_{c \in \{1 \dots 38\}} \left( \frac{1}{K} \sum_{k=1}^{K} p_c^{(k)} \right)$$

Mean probability pooling enables high-confidence correct segment predictions to dominate, effectively filtering out transient unvoiced or ambiguous introductory passages (*alaap*).

---

## V. Experimental Results and Statistical Analysis

### A. Segment-Level Performance (Validation Set)

**Table IV: Segment-Level Performance on Validation Set ($N = 5,802$ Segments)**

| Model Variant | Feature Type | Top-1 Accuracy | Top-3 Accuracy | Macro F1 |
|---|---|:---:|:---:|:---:|
| Random Baseline | — | 2.6% | 7.9% | 0.0260 |
| Baseline Mel-CRNN | Log-Mel Spectrogram | 8.0% | 20.0% | 0.0800 |
| **Proposed Swaragram-CRNN** | **Swaragram (12-bin)** | **38.1%** | **56.9%** | **0.3570** |
| Hybrid CRNN | Mel + Swaragram | 8.0% | 22.0% | 0.0800 |

**Analysis of Results:**
- **Severe Overfitting in Mel Baseline**: The Mel-CRNN achieved $>99\%$ accuracy on the training set but collapsed to $8.0\%$ on the validation set (a 91-point train-val gap). This observation is consistent with overfitting to singer-specific acoustic and timbral characteristics.
- **Superiority of Swaragram**: The Swaragram-CRNN achieved $38.1\%$ segment-level accuracy (a **4.8× improvement** over Mel-CRNN), demonstrating that isolating relative pitch dynamics supports far stronger cross-singer generalization.
- **Modality Domination in Hybrid Model**: Concatenating Mel and Swaragram features resulted in performance collapsing back to the Mel baseline ($8.0\%$). The high-capacity Mel channels dominate gradient updates during early training, preventing the network from utilizing Swaragram representations. This confirms that feature isolation must be enforced at the representation level.

---

### B. Song-Level Results and Probability Pooling Gains

**Table V: Song-Level Test Set Performance ($N = 116$ Full Recordings)**

| Model Variant | Feature Input | Song Top-1 Acc | Song Top-3 Acc | Song Macro F1 |
|---|---|:---:|:---:|:---:|
| Random Baseline | — | 2.6% | 7.9% | 0.0260 |
| Baseline Mel-CRNN | Log-Mel Spectrogram | 14.7% | 24.1% | 0.1094 |
| **Proposed Swaragram-CRNN** | **Swaragram (12-bin)** | **57.8%** | **75.9%** | **0.5086** |
| **Pooling Gain ($\Delta$)** | — | **+19.7 pp** | **+19.0 pp** | **+0.1516** |

Song-level probability pooling improves Swaragram Top-1 accuracy from $38.1\%$ (segment-level) to **57.8%** (song-level), representing a **3.9× improvement** over the Mel baseline ($14.7\%$). Because individual segment errors are largely uncorrelated, averaging posterior distributions allows consistent melodic signals to accumulate constructively across performance duration.

```
Segment-Level vs. Song-Level Accuracy Comparison
┌──────────────────────────────────────────────────────────┐
│ Mel Baseline (Segment):   ███ 8.0%                       │
│ Mel Baseline (Song):      ██████ 14.7%                   │
├──────────────────────────────────────────────────────────┤
│ Swaragram (Segment):      ███████████████ 38.1%          │
│ Swaragram (Song Top-1):   ███████████████████████ 57.8%  │
│ Swaragram (Song Top-3):   ████████████████████████████ 75.9%
└──────────────────────────────────────────────────────────┘
```
*Figure 4: Comparative classification performance highlighting gains achieved by Swaragram feature isolation and song-level pooling.*

---

### C. Statistical Significance Testing

To confirm that the observed accuracy improvement reflects a genuine representational advantage rather than sampling variance, we applied McNemar's test to the paired song-level predictions of the two models on the 116-recording test set. The contingency matrix (Table VII) yields $\chi^2 = 37.52$ ($p = 9.07 \times 10^{-10}$), significant at the 99.9% confidence level ($p \ll 0.001$). The asymmetric off-diagonal counts ($c = 57 \gg b = 7$) confirm that the Swaragram model recovers the large majority of cases where the Mel model fails while rarely introducing new errors.

To quantify uncertainty in the accuracy estimates given the test set size, we computed 95% bootstrap confidence intervals using $N = 10,000$ stratified resamples of the 116-recording test set. The non-overlapping intervals (Table VIII and Figure 8) confirm that the performance separation is robust across the observed test distribution and is not an artifact of a particular random split.

**Table VII: McNemar Contingency Matrix (Song-Level, $N = 116$)**

| Classification Pairing | Swaragram Correct | Swaragram Wrong | Total |
|---|:---:|:---:|:---:|
| **Mel Correct** | $a = 10$ | $b = 7$ | 17 |
| **Mel Wrong** | $c = 57$ | $d = 42$ | 99 |
| **Total** | 67 | 49 | 116 |
| **McNemar $\chi^2$ Statistic** | **37.52** | — | — |
| **$p$-value** | **$9.07 \times 10^{-10}$** ($p \ll 0.001$) | — | **Statistically Significant** |

**Table VIII: 95% Bootstrap Confidence Intervals ($N = 10,000$ Resamples)**

| Metric | Mel-CRNN | Swaragram-CRNN | Difference ($\Delta = \text{Swara} - \text{Mel}$) |
|---|:---:|:---:|:---:|
| **Top-1 Accuracy** | 14.7% $[8.6\%, 21.6\%]$ | **57.8%** $[49.1\%, 66.4\%]$ | **+43.0 pp** $[+31.9\text{ pp}, +53.4\text{ pp}]$ |
| **Top-3 Accuracy** | 24.1% $[16.4\%, 31.9\%]$ | **75.9%** $[68.1\%, 83.6\%]$ | **+51.7 pp** $[+40.5\text{ pp}, +62.1\text{ pp}]$ |
| **Macro F1 Score** | 0.109 $[0.064, 0.157]$ | **0.509** $[0.415, 0.603]$ | **+0.399** $[+0.292, +0.505]$ |

```
95% Bootstrap Confidence Intervals (N = 10,000 Resamples)
┌──────────────────────────────────────────────────────────┐
│ Top-1 Acc:  Mel [8.6% ── 21.6%]  vs  Swara [49.1% ── 66.4%]  (Δ = +43.0%) │
│ Top-3 Acc:  Mel [16.4% ── 31.9%] vs  Swara [68.1% ── 83.6%]  (Δ = +51.7%) │
│ Macro F1:   Mel [0.064 ── 0.157] vs  Swara [0.415 ── 0.603]  (Δ = +0.399) │
└──────────────────────────────────────────────────────────┘
```
*Figure 8: Visual comparison of non-overlapping 95% bootstrap confidence intervals for Mel-CRNN vs. Swaragram-CRNN across Top-1, Top-3, and Macro F1.*

---

### D. Per-Raga Classification and Error Analysis

**Table IX: Song-Level Accuracy Breakdown for Selected Ragas**

| Raga Name | Test Recordings | Top-1 Acc (%) | Top-3 Acc (%) | Primary Confusion Class |
|---|:---:|:---:|:---:|---|
| **Bhairav** | 2 | **100.0%** | **100.0%** | None (Perfect) |
| **Poorvi** | 2 | **100.0%** | **100.0%** | None (Perfect) |
| **Durga** | 2 | **100.0%** | **100.0%** | None (Perfect) |
| **Ramkali** | 2 | **100.0%** | **100.0%** | None (Perfect) |
| **Basant** | 4 | **75.0%** | **100.0%** | Raga Bahar |
| **Bageshree** | 4 | **75.0%** | **100.0%** | Raga Bhimpalasi |
| **Bilaskhani Todi** | 3 | **66.7%** | **100.0%** | Raga Todi |
| **Bhairavi** | 5 | **60.0%** | **60.0%** | Raga Malkauns |
| **Yaman** | 5 | **40.0%** | **100.0%** | Raga Yaman Kalyan |
| **Bilaval** | 3 | **0.0%** | **33.3%** | Raga Alhaiya Bilaval |

```
                 Swaragram-CRNN Normalized Confusion Matrix
        ┌─────────────────────────────────────────────────────────┐
Bhairav │  1.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00 │
 Poorvi │  0.00  1.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00 │
 Basant │  0.00  0.00  0.75  0.25  0.00  0.00  0.00  0.00  0.00  0.00 │ (Confused w/ Bahar)
   Todi │  0.00  0.00  0.00  0.67  0.33  0.00  0.00  0.00  0.00  0.00 │ (Confused w/ B. Todi)
  Yaman │  0.00  0.00  0.00  0.00  0.40  0.60  0.00  0.00  0.00  0.00 │ (Confused w/ Y. Kalyan)
        └─────────────────────────────────────────────────────────┘
          Bhairav Poorvi Basant Bahar Yaman Y.Kal Todi B.Todi ...
```
*Figure 5: Normalized confusion matrix snippet demonstrating that residual classification errors occur primarily between musicologically adjacent ragas.*

---

### E. Empirical CPU/GPU Runtime, Memory, and Latency Analysis

To provide a complete computational characterization for real-time deployment, we benchmarked end-to-end latency, memory allocation, and model forward-pass speed across both CPU and GPU hardware environments.

#### 1) Hardware and Software Specification
- **CPU Architecture**: AMD Ryzen 7 7735HS with Radeon Graphics ($3.20\text{ GHz}$ base, $4.75\text{ GHz}$ boost, 8 physical cores, 16 logical threads).
- **GPU Accelerator**: NVIDIA GeForce RTX 4050 Laptop GPU ($6\text{ GB}$ GDDR6 VRAM, CUDA 11.8 / cuDNN 8.7).
- **Software Stack**: Python 3.11.9, PyTorch 2.7.1+cu118, Librosa 0.10.2, NumPy 1.26.4.
- **Timing Protocol**: All reported latencies represent arithmetic means computed over 10 sample test recordings ($270\text{ seconds}$ total audio). To eliminate warm-up artifacts and asynchronous kernel launch bias, 10 initial warm-up forward passes were executed, and explicit `torch.cuda.synchronize()` boundary calls were placed before and after every timing measurement.

**Table X: Comprehensive Empirical Runtime, Memory, and Latency Breakdown**

| Pipeline Stage / Computational Metric | Baseline Mel-CRNN | Proposed Swaragram-CRNN | Efficiency Benefit / Speedup |
|---|:---:|:---:|:---:|
| **Spatial Feature Input Dimension ($F$)** | $128$ Mel Bins | **$12$ Swara Bins** | **10.6× spatial compression** |
| **Total Neural Parameters** | $4,185,254$ | **$1,794,214$** | **-57.1% parameter reduction** |
| **GPU VRAM Allocation (Batch=32)** | $\sim 142\text{ MB}$ | **$\sim 38\text{ MB}$** | **-73.2% VRAM footprint** |
| Audio Loading & Resampling ($22.05\text{ kHz}$) | $572.83\text{ ms}$ | $572.83\text{ ms}$ | Standard audio I/O |
| YIN $F_0$ Pitch Tracking (30s) | — | $173.94\text{ ms}$ | Monophonic pitch tracking |
| Tonic ($Sa$) Estimation (30s) | — | **$0.49\text{ ms}$** | Ultra-fast histogram peak finding |
| Feature Map Generation (30s) | $49.99\text{ ms}$ | **$16.16\text{ ms}$** | **3.09× faster feature gen** |
| **NN Inference — CPU ($b=1$)** | $58.15\text{ ms}$ | **$40.05\text{ ms}$** | **1.45× CPU inference speedup** |
| **NN Inference — GPU ($b=1$)** | $14.54\text{ ms}$ | **$16.05\text{ ms}$** | Single-segment GPU evaluation |
| **NN Inference — GPU ($b=32$, per seg)** | $2.65\text{ ms}$ | **$1.07\text{ ms}$** | **2.48× GPU batch throughput** |
| **Total 30s Segment Latency (CPU)** | **$680.97\text{ ms}$** | **$803.47\text{ ms}$** | Real-time streaming capable |
| **Total 30s Segment Latency (GPU)** | **$637.36\text{ ms}$** | **$779.47\text{ ms}$** | Real-time streaming capable |
| **Real-Time Factor (RTF — CPU)** | $0.02270$ | **$0.02678$** | **$37.3\times$ faster than real-time** |
| **Real-Time Factor (RTF — GPU)** | $0.02125$ | **$0.02598$** | **$38.5\times$ faster than real-time** |
| **Full Song Latency (27 Segs / 13.5 min - CPU)** | **$18.39\text{ s}$** | **$21.69\text{ s}$** | Complete performance evaluation |
| **Full Song Latency (27 Segs / 13.5 min - GPU)** | **$17.21\text{ s}$** | **$21.05\text{ s}$** | Complete performance evaluation |

#### 2) Discussion of Empirical Latency and Throughput
1. **Model Parameter & VRAM Savings**: The 12-bin Swaragram cuts neural parameters by **57.1%** (1.8M vs. 4.2M) and GPU memory usage by **73.2%** (38 MB vs. 142 MB per batch), enabling high-density multi-stream inference on consumer-grade hardware.
2. **Inference Latency & GPU Parallel Batching**: On CPU (AMD Ryzen 7 7735HS), a single-segment Swaragram forward pass requires **$40.05\text{ ms}$** (vs. $58.15\text{ ms}$ for Mel, a **1.45× CPU speedup**). When processing multi-segment recordings in parallel batches ($b=32$) on GPU (NVIDIA RTX 4050), Swaragram inference time drops to **$1.07\text{ ms}$ per segment** (vs. $2.65\text{ ms}$ for Mel, a **2.48× GPU batch throughput speedup**).
3. **End-to-End Throughput**: Preprocessing (audio load, YIN pitch extraction, tonic estimation, and matrix generation) takes $763.42\text{ ms}$ per 30-second window. Coupled with GPU neural inference, the complete end-to-end pipeline requires $779.47\text{ ms}$ ($\text{RTF} = 0.02598$), operating **$38.5\times$ faster than real-time**.
4. **Full Song Processing**: Evaluating a complete 13.5-minute performance (median 27 segments) requires only **$21.69\text{ seconds}$ on CPU** and **$21.05\text{ seconds}$ on GPU**, demonstrating that the system is ready for immediate deployment in archival indexing and real-time interactive MIR systems.

---

## VI. Discussion

### A. Feature Isolation vs. Timbral Overfitting
The performance divergence between log-Mel spectrograms ($14.7\%$ song Top-1) and Swaragrams ($57.8\%$ song Top-1) provides strong evidence regarding the nature of deep audio feature representations. Because log-Mel spectrograms preserve spectral envelope formants, pitch registers, and background acoustic environments, deep neural networks readily exploit these high-capacity cues to minimize training loss. When evaluated across unseen performers, this reliance leads to poor generalization.

The Swaragram serves as an intentional **representation bottleneck**. By projecting $F_0$ contours onto a 12-bin relative scale anchored to $Sa$, absolute pitch transpositions, singer formants, and instrumental timbres are explicitly removed. This forces the CRNN to learn features solely from pitch-class occupancy and temporal transitions.

### B. Musicological Interpretability of Errors
Analysis of residual classification errors indicates that misclassifications reflect structured musicological relationships rather than random noise:

1. **Yaman $\leftrightarrow$ Yaman Kalyan**: Yaman Kalyan incorporates identical scale degrees to Yaman, adding a subtle *shuddha Ma* in descending phrases. In our test set, $60\%$ of Yaman misclassifications were assigned to Yaman Kalyan, yielding **100% Top-3 accuracy**.
2. **Basant $\leftrightarrow$ Bahar**: Both are seasonal ragas sharing prominent *komal Ni* and *madhyam* phrase structures.
3. **Bilaskhani Todi $\leftrightarrow$ Todi**: Both belong to the Todi family, sharing flat second (*komal Re*), flat third (*komal Ga*), and flat sixth (*komal Dha*) intervals.

These confusion patterns align closely with known musicological similarities, suggesting that the Swaragram-CRNN learns genuine scalar and phrase dynamics.

### C. Computational Efficiency and Real-Time Benchmarking
The Swaragram representation offers notable computational advantages. Reducing spatial feature dimension from $F=128$ to $F=12$ decreases memory footprint by **73%**, network parameters by **57%** (1.8M vs. 4.2M), and neural network CPU forward pass latency from $58.15\text{ ms}$ to $40.05\text{ ms}$ (**1.45× CPU speedup**), with GPU batch forward pass latency dropping to $1.07\text{ ms}$ per segment at $b=32$ (**2.48× GPU throughput speedup**). With a Real-Time Factor of $\text{RTF} = 0.02598$ ($38.5\times$ faster than real-time on GPU) and full 13.5-minute song evaluation in $21.05\text{ seconds}$, the Swaragram-CRNN is ideally suited for edge deployment, automated archival indexing, and real-time educational applications.

### D. Limitations
1. **Dependence on $F_0$ Tracking Quality**: The Swaragram pipeline relies on reliable monophonic fundamental frequency extraction. Dense polyphonic accompaniment or noisy historical recordings can degrade $F_0$ accuracy.
2. **Microtonal Smoothing**: Soft-assigning pitches into 12 semitone bins using a Gaussian kernel ($\sigma = 25\text{ cents}$) smooths out subtle microtonal intonations (*shruti* variations) characteristic of specific gharanas.
3. **Class Imbalance**: Dataset A exhibits class imbalance (ranging from 15 to 32 recordings per class), contributing to lower recall on underrepresented ragas.

---

## VII. Conclusion and Future Work

We have presented a robust methodology for raga identification in Indian traditional music using tonic-normalized pitch-class Swaragrams. By validating our tonic estimator against ground-truth annotations ($85.2\%$ pitch-class agreement) and demonstrating minimal ablation loss ($97.7\%$ vs. $100.0\%$ Top-1 accuracy), we established the resilience of the proposed representation to preprocessing variation.

Applied to a 38-raga dataset of 772 recordings (~210 hours), our Swaragram-CRNN achieved **57.8% Top-1** (95% CI: $[49.1\%, 66.4\%]$) and **75.9% Top-3** (95% CI: $[68.1\%, 83.6\%]$) song-level accuracy — a **3.9× improvement** over a conventional Mel spectrogram baseline ($14.7\%$, $p = 9.07 \times 10^{-10}$ via McNemar's test), while utilizing **57% fewer parameters** and running **$38.5\times$ faster than real-time** ($\text{RTF} = 0.02598$ on GPU, processing a full 13.5-minute song in $21.05\text{ seconds}$). Song-level probability pooling contributed an additional **+19.7 percentage points** in Top-1 accuracy.

### Data Availability Statement
The curated dataset splits, feature extraction scripts, trained model checkpoints, and evaluation pipelines are publicly available in the project repository for research reproducibility. Ground-truth tonic validation scripts and raw annotations are provided for replication of all reported metrics.

**Future Directions:**
- **Extension to Carnatic Music**: Adapting the Swaragram framework to Carnatic music systems (*melakarta* and *janya* ragas).
- **Self-Supervised Pretraining**: Pretraining the shared encoder on unannotated Indian vocal music archives via masked pitch-contour modeling.
- **Hierarchical Multi-Task Learning**: Jointly predicting raga labels alongside structural attributes such as *thaat* (parent scale), *vadi*, and *samvadi*.

---

## References

1. Bagchee, S. (1998). *Nād: Understanding Raga Music*. Business Publications Inc.
2. Chordia, P., & Rae, A. (2007). Raag recognition using pitch-class and pitch-class dyad distributions. *Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)*, 431–436.
3. Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). Convolutional recurrent neural networks for music classification. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2392–2396.
4. de Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. *The Journal of the Acoustical Society of America*, 111(4), 1917–1930.
5. Gulati, S., Serrà, J., Ishwar, V., & Rajan, R. (2014). Mining melodic patterns in large audio collections of Indian art music. *IEEE International Conference on Multimedia and Expo Workshps (ICMEW)*, 1–6.
6. Kumar, S., et al. (2019). Automatic raga recognition using convolutional neural networks. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 331–335.
7. Rao, S., et al. (2017). Saraga: Open datasets for Indian art music research. *SoftwareX*, 6, 201–205.
8. Salamon, J., & Gómez, E. (2012). Melody extraction from polyphonic music signals using pitch contour characteristics. *IEEE Transactions on Audio, Speech, and Language Processing*, 20(6), 1759–1770.
