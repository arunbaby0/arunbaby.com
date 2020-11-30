---
title: A Unified Approach to Speech Synthesis in Indian Languages
categories:
- publications
tags:
- publications
---
\[MS Thesis\] IIT Madras: February 2019;
Supervised by [Prof. Hema A Murthy](https://www.cse.iitm.ac.in/~hema/)

## Authors: 
Arun Baby

## Abstract: 
<em>India is a country with 22 official languages (written in 13 different scripts), 122 major languages and 1599 other languages.  These languages come from 5-6 different language families of the world.  It is only about 65% of this population that is literate, that too primarily in the vernacular.  Speech interfaces, especially in the vernacular, are enablers in such an environment.  Building text-to-speech (TTS) systems for such a diverse country necessitates a unified approach.  This research work aims to build Indian language TTS systems in a unified manner by exploiting the similarities that exist among these languages. Specifically, the focus is on two components of the TTS system, namely, text parsing and speech segmentation.</em>

<em>Parsing is the process of mapping graphemes to phonemes.  Indian languages are more or less phonetic and have about 35-38 consonants and 15-18 vowels. In spite of the number of different families which leads to divergence, there is a convergence owing to borrowings across language families. A Common Label Set (CLS) is defined to represent the various phones in Indian languages. In this work, a uniform parser is designed across all the languages capitalising on the syllable structure of these languages.</em>

<em>Segmentation is the process of finding phoneme boundaries in a speech utterance. The main drawback of the Gaussian mixture model - hidden Markov model (GMM-HMM) based forced-alignment is that the phoneme boundaries are not explicitly modeled. State-of-the-art  speech segmentation approach for speech segmentation in Indian languages is hybrid segmentation which uses signal processing cues along with GMM-HMM framework. Deep neural networks (DNN) and convolutional neural networks (CNN) are known for robust acoustic modelling. In this work, signal processing cues, that are agnostic to speaker and language, are used in tandem with deep learning techniques to improve the phonetic segmentation.</em>


## Cite:
```
@booklet{arunThesis, 
    author = {Baby, Arun},
    title = "{A Unified Approach to Speech Synthesis in Indian Languages}",
    address = "{M.} {S.} {T}hesis, Department of Computer Science Engineering, IIT Madras, India",
    booktitle = {msiitm},
    year = {2018}
}
```

## Links:

[PDF](/assets/docs/MSthesis_2019.pdf)

[IndicTTS](https://www.iitm.ac.in/donlab/tts/publications.php)

## Code:

[Unified Parser](https://www.iitm.ac.in/donlab/tts/unified.php)


[Segmentation code](https://www.iitm.ac.in/donlab/tts/hybridSeg.php)
