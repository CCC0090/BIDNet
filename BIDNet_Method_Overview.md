# BIDNet: Brain-Inspired Detection Strategy (BIDS)

> **Code Release Notice**  
> The full implementation will be released after the paper is accepted for publication.  
> This repository currently provides a high-level algorithmic description of the proposed method.

## Overview

This repository presents the core ideas of **BIDNet**, a brain-inspired frequency-aware network for small object detection and understanding in remote sensing imagery.  
The method is built upon a **Brain-Inspired Detection Strategy (BIDS)**, which couples:

- **Spatial Filtering and Focusing (SFF)** for frequency-aware perceptual refinement
- **Cognitive Loss Computation (CLC)** for ambiguity-aware and scale-aware supervisory optimization

The full training and inference code will be made publicly available after publication.  
At this stage, this document provides an implementation-oriented summary and pseudocode for the main components.

---

## 1. Brain-Inspired Detection Strategy (BIDS)

### Key idea

BIDS is designed from a brain-inspired perspective that combines:

1. **Selective perception**: preserve informative structures and suppress redundant responses
2. **Adaptive supervision**: assign stronger learning emphasis to ambiguous and scale-sensitive samples

Instead of improving only the backbone or only the loss function, BIDS jointly refines feature representation and training optimization.

### High-level pipeline

```text
Input Image
    ↓
Backbone Feature Extraction
    ↓
SFF: Frequency Subband Parsing → Selective Focusing → Frequency-guided Reconstruction
    ↓
Detection Head
    ↓
CLC: Ambiguity-aware Reweighting + Scale-aware Localization Modulation
    ↓
Final Optimization
```

---

## 2. Spatial Filtering and Focusing (SFF)

### Motivation

Repeated feature resampling in small object detection often causes:

- aliasing
- boundary blurring
- loss of fine structural details
- weakened responses for tiny targets

To address this, SFF performs frequency-aware refinement during feature resampling.

### Core design

SFF consists of three stages:

1. **Frequency subband parsing**
2. **Selective focusing**
3. **Frequency-guided reconstruction**

### Pseudocode

```python
def SFF(feature):
    # Step 1: Frequency subband parsing
    LL, LH, HL, HH = frequency_decompose(feature)

    # Step 2: Selective focusing
    # suppress redundant low-frequency responses
    # enhance target-relevant high-frequency details
    LH_refined = adaptive_focus(LH)
    HL_refined = adaptive_focus(HL)
    HH_refined = adaptive_focus(HH)

    # Step 3: Frequency-guided reconstruction
    refined_feature = frequency_reconstruct(LL, LH_refined, HL_refined, HH_refined)

    return refined_feature
```

### Functional interpretation

- **LL** preserves coarse semantic structure
- **LH / HL / HH** describe directional high-frequency details
- **adaptive_focus(·)** selectively enhances informative spectral responses
- **frequency_reconstruct(·)** maps the refined components back into a structurally consistent representation

### Expected effect

SFF is intended to:

- preserve fine object boundaries
- enhance texture-sensitive responses
- reduce aliasing introduced by downsampling
- improve the perceptual quality of small-object features

---

## 3. Cognitive Loss Computation (CLC)

### Motivation

Standard detection losses often treat all samples too uniformly or lack sufficient scale sensitivity.  
This is especially problematic in remote-sensing small object detection, where targets are:

- small and ambiguous
- heavily imbalanced
- sensitive to scale mismatch

CLC is introduced to provide cognition-inspired supervisory optimization.

### Core design

CLC contains two complementary parts:

1. **GRL**: ambiguity-aware sample reweighting
2. **LSAP**: scale-aware localization modulation

### Pseudocode

```python
def CLC(prediction, target):
    # Classification / confidence branch
    grl_weight = gaussian_reweight(prediction.confidence, target)
    classification_loss = grl_weight * base_classification_loss(prediction, target)

    # Localization branch
    scale_deviation = compute_scale_deviation(prediction.box, target.box)
    scale_weight = scale_aware_modulation(scale_deviation)
    localization_loss = scale_weight * base_localization_loss(prediction.box, target.box)

    total_loss = classification_loss + localization_loss
    return total_loss
```

---

## 4. Gaussian Reassignment Loss (GRL)

### Goal

GRL emphasizes informative ambiguous samples instead of treating all positives and negatives uniformly.

### Intuition

- very easy samples contribute less
- extremely unreliable samples are suppressed
- ambiguous yet informative samples receive stronger supervision

### Pseudocode

```python
def gaussian_reweight(confidence, target, mu, sigma):
    # confidence-aware Gaussian weighting
    weight = exp(-((confidence - mu) ** 2) / (2 * sigma ** 2))
    return weight
```

### Effect

GRL improves:

- ambiguity-aware learning
- sample discrimination
- optimization stability under hard examples

---

## 5. LSAP: Scale-Aware Localization Modulation

### Goal

LSAP improves localization robustness when object scale mismatch is severe.

### Intuition

Instead of using IoU alone, LSAP explicitly models scale discrepancy and adjusts localization supervision accordingly.

### Pseudocode

```python
def compute_scale_deviation(pred_box, gt_box):
    dw = width_difference(pred_box, gt_box)
    dh = height_difference(pred_box, gt_box)
    nw = normalized_width_gap(pred_box, gt_box)
    nh = normalized_height_gap(pred_box, gt_box)
    S = alpha * (dw + dh + nw + nh)
    return S


def scale_aware_modulation(S, lam):
    x = lam * exp(-S)
    weight = 3 * x * exp(-(x ** 2))
    return weight
```

### Effect

LSAP is intended to:

- strengthen localization learning for scale-sensitive targets
- suppress unstable regression under large mismatch
- improve robustness for small and overlapping objects

---

## 6. Why BIDS matters

The proposed strategy is designed to couple:

- **frequency-aware perceptual refinement** at the representation level
- **cognition-inspired supervisory optimization** at the learning level

This collaborative design is expected to improve both:

- feature quality
- optimization behavior

rather than relying on isolated module improvements.

---

## 7. Current Repository Status

At present, this repository provides:

- method overview
- design motivation
- pseudocode for SFF and CLC

The complete source code, training configuration, and experimental scripts will be released after the associated paper is accepted for publication.

---

## 8. Contact

If you are interested in the method or need additional information before the public code release, please contact the authors through the corresponding paper information.

