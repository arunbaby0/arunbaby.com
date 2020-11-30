---
title: Significance of spectral cues in automatic speech segmentation for Indian language speech synthesizers
categories:
- publications
tags:
- publications
---
\[Journal\] [**Speech Communication**](https://www.sciencedirect.com/science/article/abs/pii/S0167639320302375): Volume 123, October 2020, Pages 10-25;

## Authors: 
**Arun Baby**, Jeena J.Prakash, Aswin Shanmugam Subramanian, Hema A.Murthy

## Abstract: 
<em>Building speech synthesis systems for Indian languages is challenging owing to the fact that digital resources for these languages are hardly available. Vocabulary independent speech synthesis requires that a given text is split at the level of the smallest sound unit, namely, phone. The waveforms or models of phones are concatenated to produce speech. The waveforms corresponding to that of the phones are obtained manual (listening and marking) when digital resources are scarce. But the manual labeling of speech data (also known as speech segmentation) can lead to inconsistencies as the duration of phones can be as short as 10ms.</em>

<em>The most common approach to automatic segmentation of speech is to perform forced alignment using monophone hidden Markov models (HMMs) that have been obtained using embedded re-estimation after flat start initialization. These results are then used in neural network frameworks to build better acoustic models for speech synthesis/recognition. Segmentation using this approach requires large amounts of data and does not work very well for low resource languages. To address the issue of paucity of data, signal processing cues like short-term energy (STE) and sub-band spectral flux (SBSF) are used in tandem with HMM based forced alignment for automatic speech segmentation.</em>

<em>STE and SBSF are computed on the speech waveforms. STE yields syllable boundaries, while SBSF provides locations of significant change in spectral flux that are indicative of fricatives, affricates, and nasals. STE and SBSF cannot be used directly to segment an utterance. Minimum phase group delay based smoothing is performed to preserve these landmarks, while at the same time reducing the local fluctuations. The boundaries obtained with HMMs are corrected at the syllable level, wherever it is known that the syllable boundaries are correct. Embedded re-estimation of monophone HMM models is again performed using the corrected alignment. Thus, using signal processing cues and HMM re-estimation in tandem, robust monophone HMM models are built. These models are then used in Gaussian mixture model (GMM), deep neural network (DNN) and convolutional neural network (CNN) frameworks to obtain state-level frame posteriors. The boundaries are again iteratively corrected and re-estimated.</em>

<em>Text-to-speech (TTS) systems are built for different Indian languages using phone alignments obtained with and without the use of signal processing based boundary corrections. Unit selection based and statistical parametric based TTS systems are built. The result of the listening tests showed a significant improvement in the quality of synthesis with the use of signal processing based boundary correction.</em>

## Cite:
```
@article{BABY202010,
title = "Significance of spectral cues in automatic speech segmentation for Indian language speech synthesizers",
journal = "Speech Communication",
volume = "123",
pages = "10 - 25",
year = "2020",
issn = "0167-6393",
doi = "https://doi.org/10.1016/j.specom.2020.06.002",
url = "http://www.sciencedirect.com/science/article/pii/S0167639320302375",
author = "Arun Baby and Jeena J. Prakash and Aswin Shanmugam Subramanian and Hema A. Murthy",
keywords = "Speech segmentation, Signal processing cues, Short-term energy, Sub-band spectral flux, Hidden markov model, Gaussian mixture model, Deep neural network, Convolutional neural network",
}
```

## Links:
[Speech communication](https://www.sciencedirect.com/science/article/abs/pii/S0167639320302375)

[IndicTTS](https://www.iitm.ac.in/donlab/tts/publications.php)

## Code:

[Segmentation code](https://www.iitm.ac.in/donlab/tts/hybridSeg.php)
