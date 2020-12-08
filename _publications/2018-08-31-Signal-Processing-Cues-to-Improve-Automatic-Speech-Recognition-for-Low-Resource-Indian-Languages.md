---
title: Signal Processing Cues to Improve Automatic Speech Recognition for Low Resource Indian Languages
categories:
- publications
tags:
- publications
---
\[Conference\] The 6th Intl Workshop on Spoken Language Technologies for Under Resourced Languages, Gurugram, India, August 2018


## Authors:
**Arun Baby**, Karthik Pandia D S, Hema A Murthy


## Abstract:
<em>Building accurate acoustic models for low resource languages is the focus of this paper. Acoustic models are likely to be accurate provided the phone boundaries are determined accurately. Conventional flat-start based Viterbi phone alignment (where only utterance level transcriptions are available) results in poor phone boundaries as the boundaries are not explicitly modeled in any statistical machine learning system. The focus of the effort in this paper is to explicitly model phrase boundaries using acoustic cues obtained using signal processing. A phrase is made up of a sequence of words, where each word is made up of a sequence of syllables. Syllable boundaries are detected using signal processing. The waveform corresponding to an utterance is spliced at phrase boundaries when it matches a syllable boundary. Gaussian mixture model - hidden Markov model (GMM-HMM) training is performed phrase by phrase, rather than utterance by utterance. Training using these short phrases yields better acoustic models. This alignment is then fed to a DNN to enable better discrimination between phones. During the training process, the syllable boundaries (obtained using signal processing) are restored in every iteration. A relative improvement is observed in WER over the baseline Indian languages, namely, Gujarati, Tamil, and Telugu.</em>


## Cite:
```
@inproceedings{Baby2018,
  author={Arun Baby and Karthik {Pandia D S} and Hema {A Murthy}},
  title={Signal Processing Cues to Improve Automatic Speech Recognition for Low Resource Indian Languages},
  year=2018,
  booktitle={Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages},
  pages={25--29},
  doi={10.21437/SLTU.2018-6},
  url={http://dx.doi.org/10.21437/SLTU.2018-6}
}
```

## Links:
[Proceedings](https://www.isca-speech.org/archive/SLTU_2018/pdfs/Arun.pdf)


## Code:
[Link](https://www.iitm.ac.in/donlab/tts/hybridSeg.php)

