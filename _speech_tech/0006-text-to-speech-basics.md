---
title: "Text-to-Speech (TTS) System Fundamentals"
day: 6
collection: speech_tech
categories:
  - speech-tech
tags:
  - tts
  - synthesis
  - prosody
  - neural-tts
subdomain: Speech Synthesis
tech_stack: [Python, PyTorch, TensorFlow, Tacotron, WaveNet, FastSpeech]
scale: "Real-time streaming"
companies: [Google, Amazon, Microsoft, Apple, Meta]
related_dsa_day: 6
related_ml_day: 6
---

**From text to natural speech: understanding modern neural TTS architectures that power Alexa, Google Assistant, and Siri.**

## Introduction

**Text-to-Speech (TTS)** converts written text into spoken audio. Modern TTS systems produce human-like speech quality using deep learning.

**Why TTS matters:**
- **Virtual assistants:** Alexa, Google Assistant, Siri
- **Accessibility:** Screen readers for visually impaired
- **Content creation:** Audiobooks, podcasts, voiceovers
- **Conversational AI:** Voice bots, IVR systems
- **Education:** Language learning apps

**Evolution:**
1. **Concatenative synthesis** (1990s-2000s): Stitch pre-recorded audio units
2. **Parametric synthesis** (2000s-2010s): Statistical models (HMM)
3. **Neural TTS** (2016+): End-to-end deep learning (Tacotron, WaveNet)
4. **Modern TTS** (2020+): Fast, controllable, expressive (FastSpeech, VITS)

---

## TTS Pipeline Architecture

### Traditional Two-Stage Pipeline

Most TTS systems use a two-stage approach:

```
Text → [Acoustic Model] → Mel Spectrogram → [Vocoder] → Audio Waveform
```

**Stage 1: Acoustic Model (Text → Mel Spectrogram)**
- Input: Text (characters/phonemes)
- Output: Mel spectrogram (acoustic features)
- Examples: Tacotron 2, FastSpeech 2

**Stage 2: Vocoder (Mel Spectrogram → Waveform)**
- Input: Mel spectrogram
- Output: Audio waveform
- Examples: WaveNet, WaveGlow, HiFi-GAN

### Why Two Stages?

**Advantages:**
- **Modularity:** Train acoustic model and vocoder separately
- **Efficiency:** Mel spectrogram is compressed representation
- **Controllability:** Can modify prosody at mel spectrogram level

**Alternative: End-to-End Models**
- VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
- Directly generates waveform from text
- Faster inference, fewer components

---

## Key Components

### 1. Text Processing (Frontend)

Transform raw text into model-ready input.

```python
class TextProcessor:
    """
    Text normalization and phoneme conversion
    """
    
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.g2p = Grapheme2Phoneme()  # Grapheme-to-Phoneme
    
    def process(self, text: str) -> list[str]:
        """
        Convert text to phoneme sequence
        
        Args:
            text: Raw input text
        
        Returns:
            List of phonemes
        """
        # 1. Normalize text
        normalized = self.normalizer.normalize(text)
        # "Dr. Smith has $100" → "Doctor Smith has one hundred dollars"
        
        # 2. Convert to phonemes
        phonemes = self.g2p.convert(normalized)
        # "hello" → ['HH', 'AH', 'L', 'OW']
        
        return phonemes

class TextNormalizer:
    """
    Normalize text (expand abbreviations, numbers, etc.)
    """
    
    def normalize(self, text: str) -> str:
        text = self._expand_abbreviations(text)
        text = self._expand_numbers(text)
        text = self._expand_symbols(text)
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Dr. → Doctor, Mr. → Mister, etc."""
        expansions = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Misses',
            'Ms.': 'Miss',
            'St.': 'Street',
        }
        for abbr, expansion in expansions.items():
            text = text.replace(abbr, expansion)
        return text
    
    def _expand_numbers(self, text: str) -> str:
        """$100 → one hundred dollars"""
        import re
        
        # Currency
        text = re.sub(r'\$(\d+)', r'\1 dollars', text)
        
        # Years
        text = re.sub(r'(\d{4})', self._year_to_words, text)
        
        return text
    
    def _year_to_words(self, match) -> str:
        """Convert year to words: 2024 → twenty twenty-four"""
        # Simplified implementation
        return match.group(0)  # Placeholder
    
    def _expand_symbols(self, text: str) -> str:
        """@ → at, % → percent, etc."""
        symbols = {
            '@': 'at',
            '%': 'percent',
            '#': 'number',
            '&': 'and',
        }
        for symbol, expansion in symbols.items():
            text = text.replace(symbol, expansion)
        return text
```

### 2. Acoustic Model

Generates mel spectrogram from text/phonemes.

**Tacotron 2 Architecture (simplified):**

```
Input: Phoneme sequence
   ↓
[Character Embeddings]
   ↓
[Encoder] (Bidirectional LSTM)
   ↓
[Attention] (Location-sensitive)
   ↓
[Decoder] (Autoregressive LSTM)
   ↓
[Mel Predictor]
   ↓
Output: Mel Spectrogram
```

```python
import torch
import torch.nn as nn

class SimplifiedTacotron(nn.Module):
    """
    Simplified Tacotron-style acoustic model
    
    Real Tacotron 2 is much more complex!
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        encoder_hidden: int = 256,
        decoder_hidden: int = 1024,
        n_mels: int = 80
    ):
        super().__init__()
        
        # Character/phoneme embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder
        self.encoder = nn.LSTM(
            embedding_dim,
            encoder_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention
        self.attention = LocationSensitiveAttention(
            encoder_hidden * 2,  # Bidirectional
            decoder_hidden
        )
        
        # Decoder
        self.decoder = nn.LSTMCell(
            encoder_hidden * 2 + n_mels,  # Context + previous mel frame
            decoder_hidden
        )
        
        # Mel predictor
        self.mel_predictor = nn.Linear(decoder_hidden, n_mels)
        
        # Stop token predictor
        self.stop_predictor = nn.Linear(decoder_hidden, 1)
    
    def forward(self, text, mel_targets=None):
        """
        Forward pass
        
        Args:
            text: [batch, seq_len] phoneme indices
            mel_targets: [batch, mel_len, n_mels] target mels (training only)
        
        Returns:
            mels: [batch, mel_len, n_mels]
            stop_tokens: [batch, mel_len]
        """
        # Encode text
        embedded = self.embedding(text)  # [batch, seq_len, embed_dim]
        encoder_outputs, _ = self.encoder(embedded)  # [batch, seq_len, hidden*2]
        
        # Decode (autoregressive)
        if mel_targets is not None:
            # Teacher forcing during training
            return self._decode_teacher_forcing(encoder_outputs, mel_targets)
        else:
            # Autoregressive during inference
            return self._decode_autoregressive(encoder_outputs, max_len=1000)
    
    def _decode_autoregressive(self, encoder_outputs, max_len):
        """Autoregressive decoding (inference)"""
        batch_size = encoder_outputs.size(0)
        n_mels = self.mel_predictor.out_features
        
        # Initialize
        decoder_hidden = torch.zeros(batch_size, self.decoder.hidden_size)
        decoder_cell = torch.zeros(batch_size, self.decoder.hidden_size)
        attention_context = torch.zeros(batch_size, encoder_outputs.size(2))
        mel_frame = torch.zeros(batch_size, n_mels)
        
        mels = []
        stop_tokens = []
        
        for _ in range(max_len):
            # Compute attention
            attention_context, _ = self.attention(
                decoder_hidden,
                encoder_outputs
            )
            
            # Decoder step
            decoder_input = torch.cat([attention_context, mel_frame], dim=1)
            decoder_hidden, decoder_cell = self.decoder(
                decoder_input,
                (decoder_hidden, decoder_cell)
            )
            
            # Predict mel frame and stop token
            mel_frame = self.mel_predictor(decoder_hidden)
            stop_token = torch.sigmoid(self.stop_predictor(decoder_hidden))
            
            mels.append(mel_frame)
            stop_tokens.append(stop_token)
            
            # Check if should stop
            if (stop_token > 0.5).all():
                break
        
        mels = torch.stack(mels, dim=1)  # [batch, mel_len, n_mels]
        stop_tokens = torch.stack(stop_tokens, dim=1)  # [batch, mel_len, 1]
        
        return mels, stop_tokens

class LocationSensitiveAttention(nn.Module):
    """
    Location-sensitive attention (simplified)
    
    Note: Real Tacotron uses cumulative attention features; this
    minimal version omits location convolution for brevity.
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim=128):
        super().__init__()
        
        self.query_layer = nn.Linear(decoder_dim, attention_dim)
        self.key_layer = nn.Linear(encoder_dim, attention_dim)
        self.value_layer = nn.Linear(attention_dim, 1)
        
        # For brevity, location features are omitted in this simplified demo
    
    def forward(self, query, keys):
        """
        Compute attention context
        
        Args:
            query: [batch, decoder_dim] - current decoder state
            keys: [batch, seq_len, encoder_dim] - encoder outputs
        
        Returns:
            context: [batch, encoder_dim]
            attention_weights: [batch, seq_len]
        """
        # Compute attention scores
        query_proj = self.query_layer(query).unsqueeze(1)  # [batch, 1, attn_dim]
        keys_proj = self.key_layer(keys)  # [batch, seq_len, attn_dim]
        
        scores = self.value_layer(torch.tanh(query_proj + keys_proj)).squeeze(2)
        attention_weights = torch.softmax(scores, dim=1)  # [batch, seq_len]
        
        # Compute context
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            keys
        ).squeeze(1)  # [batch, encoder_dim]
        
        return context, attention_weights
```

### 3. Vocoder

Converts mel spectrogram to waveform.

**Popular Vocoders:**

| Model | Type | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| WaveNet | Autoregressive | Excellent | Slow | Original neural vocoder |
| WaveGlow | Flow-based | Excellent | Fast | Parallel generation |
| HiFi-GAN | GAN-based | Excellent | Very Fast | Current SOTA |
| MelGAN | GAN-based | Good | Very Fast | Lightweight |

**HiFi-GAN Architecture:**

```python
import torch
import torch.nn as nn

class HiFiGANGenerator(nn.Module):
    """
    Simplified HiFi-GAN generator
    
    Upsamples mel spectrogram to waveform
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    ):
        super().__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Initial conv
        self.conv_pre = nn.Conv1d(n_mels, 512, kernel_size=7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    512 // (2 ** i),
                    512 // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2
                )
            )
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 512 // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # Final conv
        self.conv_post = nn.Conv1d(ch, 1, kernel_size=7, padding=3)
    
    def forward(self, mel):
        """
        Generate waveform from mel spectrogram
        
        Args:
            mel: [batch, n_mels, mel_len]
        
        Returns:
            waveform: [batch, 1, audio_len]
        """
        x = self.conv_pre(mel)
        
        for i, up in enumerate(self.ups):
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = up(x)
            
            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x

class ResBlock(nn.Module):
    """Residual block with dilated convolutions"""
    
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2
                )
            )
    
    def forward(self, x):
        for conv in self.convs:
            xt = torch.nn.functional.leaky_relu(x, 0.1)
            xt = conv(xt)
            x = x + xt
        return x
```

---

## Modern TTS: FastSpeech 2

**Problem with Tacotron:** Autoregressive decoding is slow and can have attention errors.

**FastSpeech 2 advantages:**
- **Non-autoregressive:** Generates all mel frames in parallel (much faster)
- **Duration prediction:** Predicts phoneme durations explicitly
- **Controllability:** Can control pitch, energy, duration

**Architecture:**

```
Input: Phoneme sequence
   ↓
[Phoneme Embeddings]
   ↓
[Feed-Forward Transformer]
   ↓
[Duration Predictor] → phoneme durations
[Pitch Predictor] → pitch contour
[Energy Predictor] → energy contour
   ↓
[Length Regulator] (expand phonemes by duration)
   ↓
[Feed-Forward Transformer]
   ↓
Output: Mel Spectrogram
```

**Key innovation: Length Regulator**

```python
def length_regulator(phoneme_features, durations):
    """
    Expand phoneme features based on predicted durations
    
    Args:
        phoneme_features: [batch, phoneme_len, hidden]
        durations: [batch, phoneme_len] - frames per phoneme
    
    Returns:
        expanded: [batch, mel_len, hidden]
    """
    expanded = []
    
    for batch_idx in range(phoneme_features.size(0)):
        batch_expanded = []
        
        for phoneme_idx in range(phoneme_features.size(1)):
            feature = phoneme_features[batch_idx, phoneme_idx]
            duration = durations[batch_idx, phoneme_idx].int()
            
            # Repeat feature 'duration' times
            batch_expanded.append(feature.repeat(duration, 1))
        
        batch_expanded = torch.cat(batch_expanded, dim=0)
        expanded.append(batch_expanded)
    
    # Pad to max length
    max_len = max(e.size(0) for e in expanded)
    expanded = [torch.nn.functional.pad(e, (0, 0, 0, max_len - e.size(0))) for e in expanded]
    
    return torch.stack(expanded)

# Example
phoneme_features = torch.randn(1, 10, 256)  # 10 phonemes
durations = torch.tensor([[5, 3, 4, 6, 2, 5, 7, 3, 4, 5]])  # Frames per phoneme

expanded = length_regulator(phoneme_features, durations)
print(f"Input shape: {phoneme_features.shape}")
print(f"Output shape: {expanded.shape}")  # [1, 44, 256] (sum of durations)
```

---

## Prosody Control

**Prosody:** Rhythm, stress, and intonation of speech

**Control dimensions:**
- **Pitch:** Fundamental frequency (F0)
- **Duration:** Phoneme/word length
- **Energy:** Loudness

```python
class ProsodyController:
    """
    Control prosody in TTS generation
    """
    
    def __init__(self, model):
        self.model = model
    
    def synthesize_with_prosody(
        self,
        text: str,
        pitch_scale: float = 1.0,
        duration_scale: float = 1.0,
        energy_scale: float = 1.0
    ):
        """
        Generate speech with prosody control
        
        Args:
            text: Input text
            pitch_scale: Multiply pitch by this factor (>1 = higher, <1 = lower)
            duration_scale: Multiply duration by this factor (>1 = slower, <1 = faster)
            energy_scale: Multiply energy by this factor (>1 = louder, <1 = softer)
        
        Returns:
            audio: Generated waveform
        """
        # Get model predictions
        phonemes = self.model.text_to_phonemes(text)
        mel_spec, pitch, duration, energy = self.model.predict(phonemes)
        
        # Apply prosody modifications
        pitch_modified = pitch * pitch_scale
        duration_modified = duration * duration_scale
        energy_modified = energy * energy_scale
        
        # Regenerate mel spectrogram with modified prosody
        mel_spec_modified = self.model.synthesize_mel(
            phonemes,
            pitch=pitch_modified,
            duration=duration_modified,
            energy=energy_modified
        )
        
        # Vocoder: mel → waveform
        audio = self.model.vocoder(mel_spec_modified)
        
        return audio

# Usage example
controller = ProsodyController(tts_model)

# Happy speech: higher pitch, faster
happy_audio = controller.synthesize_with_prosody(
    "Hello, how are you?",
    pitch_scale=1.2,
    duration_scale=0.9,
    energy_scale=1.1
)

# Sad speech: lower pitch, slower
sad_audio = controller.synthesize_with_prosody(
    "Hello, how are you?",
    pitch_scale=0.8,
    duration_scale=1.2,
    energy_scale=0.9
)
```

---

## Connection to Evaluation Metrics

Like ML model evaluation, TTS systems need multiple metrics:

### Objective Metrics

**Mel Cepstral Distortion (MCD):**
- Measures distance between generated and ground truth mels
- Lower is better
- Correlates with quality but not perfectly

```python
import librosa
import numpy as np

def mel_cepstral_distortion(generated_mel, target_mel):
    """
    Compute MCD between generated and target mel spectrograms
    
    Args:
        generated_mel: [n_mels, time]
        target_mel: [n_mels, time]
    
    Returns:
        MCD score (lower is better)
    """
    # Align lengths
    min_len = min(generated_mel.shape[1], target_mel.shape[1])
    generated_mel = generated_mel[:, :min_len]
    target_mel = target_mel[:, :min_len]
    
    # Compute simple L2 distance over mel frames (proxy for MCD)
    diff = generated_mel - target_mel
    mcd = float(np.linalg.norm(diff, axis=0).mean())
    
    return mcd
```

**F0 RMSE:** Root mean squared error of pitch

**Duration Accuracy:** How well predicted durations match ground truth

### Subjective Metrics

**Mean Opinion Score (MOS):**
- Human raters score quality 1-5
- Gold standard for TTS evaluation
- Expensive and time-consuming

**MUSHRA Test:** Compare multiple systems side-by-side

---

## Production Considerations

### Latency

**Components of TTS latency:**
1. **Text processing:** 5-10ms
2. **Acoustic model:** 50-200ms (depends on text length)
3. **Vocoder:** 20-100ms
4. **Total:** 75-310ms

**Optimization strategies:**
- **Streaming TTS:** Generate audio incrementally
- **Model distillation:** Smaller, faster models
- **Quantization:** INT8 inference
- **Caching:** Pre-generate common phrases

### Multi-Speaker TTS

```python
class MultiSpeakerTTS:
    """
    TTS supporting multiple voices
    
    Approach 1: Speaker embedding
    Approach 2: Separate models per speaker
    """
    
    def __init__(self, model, speaker_embeddings):
        self.model = model
        self.speaker_embeddings = speaker_embeddings
    
    def synthesize(self, text: str, speaker_id: int):
        """
        Generate speech in specific speaker's voice
        
        Args:
            text: Input text
            speaker_id: Speaker identifier
        
        Returns:
            audio: Waveform in target speaker's voice
        """
        # Get speaker embedding
        speaker_emb = self.speaker_embeddings[speaker_id]
        
        # Generate with speaker conditioning
        mel = self.model.generate_mel(text, speaker_embedding=speaker_emb)
        audio = self.model.vocoder(mel)
        
        return audio
```

---

## Training Data Requirements

### Dataset Characteristics

**Typical single-speaker TTS training:**
- **Audio hours:** 10-24 hours of clean speech
- **Utterances:** 5,000-15,000 sentences
- **Recording quality:** Studio quality, 22 kHz+ sample rate
- **Text diversity:** Cover phonetic diversity of language

**Multi-speaker TTS:**
- **Speakers:** 100-10,000 speakers
- **Hours per speaker:** 1-5 hours
- **Total hours:** 100-50,000 hours (e.g., LibriTTS: 585 hours, 2,456 speakers)

### Data Preparation Pipeline

```python
class TTSDataPreparation:
    """
    Prepare data for TTS training
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def prepare_dataset(
        self,
        audio_files: list[str],
        text_files: list[str]
    ) -> list[dict]:
        """
        Prepare audio-text pairs
        
        Steps:
        1. Text normalization
        2. Audio preprocessing
        3. Alignment (forced alignment)
        4. Feature extraction
        """
        dataset = []
        
        for audio_file, text_file in zip(audio_files, text_files):
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Load text
            with open(text_file) as f:
                text = f.read().strip()
            
            # Normalize text
            normalized_text = self.normalize_text(text)
            
            # Convert to phonemes
            phonemes = self.text_to_phonemes(normalized_text)
            
            # Extract mel spectrogram
            mel = self.extract_mel(audio)
            
            # Extract prosody features
            pitch = self.extract_pitch(audio)
            energy = self.extract_energy(audio)
            
            # Compute duration (requires forced alignment)
            duration = self.compute_durations(audio, phonemes)
            
            dataset.append({
                'audio_path': audio_file,
                'text': text,
                'normalized_text': normalized_text,
                'phonemes': phonemes,
                'mel': mel,
                'pitch': pitch,
                'energy': energy,
                'duration': duration
            })
        
        return dataset
    
    def extract_mel(self, audio):
        """Extract mel spectrogram"""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    
    def extract_pitch(self, audio):
        """Extract pitch (F0) contour"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        return f0
    
    def extract_energy(self, audio):
        """Extract energy (RMS)"""
        energy = librosa.feature.rms(
            y=audio,
            frame_length=1024,
            hop_length=256
        )[0]
        return energy
```

### Data Quality Challenges

**1. Noisy Audio:**
- Background noise degrades quality
- Solution: Use noise reduction, or data augmentation

**2. Alignment Errors:**
- Text-audio misalignment breaks training
- Solution: Forced alignment with Montreal Forced Aligner (MFA)

**3. Prosody Variation:**
- Inconsistent prosody confuses models
- Solution: Filter outliers, normalize prosody

**4. Out-of-Domain Text:**
- Model struggles with unseen words/names
- Solution: Diverse training text, robust G2P

---

## Voice Cloning & Few-Shot Learning

**Voice cloning:** Generate speech in a target voice with minimal data.

### Approaches

**1. Speaker Embedding (Zero-Shot/Few-Shot)**

```python
class SpeakerEncoder(nn.Module):
    """
    Encode speaker characteristics from reference audio
    
    Architecture: Similar to speaker recognition
    """
    
    def __init__(self, mel_dim=80, embedding_dim=256):
        super().__init__()
        
        # LSTM encoder
        self.encoder = nn.LSTM(
            mel_dim,
            embedding_dim,
            num_layers=3,
            batch_first=True
        )
        
        # Projection to speaker embedding
        self.projection = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, mel):
        """
        Extract speaker embedding from mel spectrogram
        
        Args:
            mel: [batch, mel_len, mel_dim]
        
        Returns:
            speaker_embedding: [batch, embedding_dim]
        """
        _, (hidden, _) = self.encoder(mel)
        
        # Use last hidden state
        speaker_emb = self.projection(hidden[-1])
        
        # L2 normalize
        speaker_emb = speaker_emb / (speaker_emb.norm(dim=1, keepdim=True) + 1e-8)
        
        return speaker_emb

class VoiceCloningTTS:
    """
    TTS with voice cloning capability
    """
    
    def __init__(self, acoustic_model, vocoder, speaker_encoder):
        self.acoustic_model = acoustic_model
        self.vocoder = vocoder
        self.speaker_encoder = speaker_encoder
    
    def clone_voice(
        self,
        text: str,
        reference_audio: torch.Tensor
    ):
        """
        Generate speech in reference voice
        
        Args:
            text: Text to synthesize
            reference_audio: Audio sample of target voice (3-10 seconds)
        
        Returns:
            synthesized_audio: Speech in target voice
        """
        # Extract speaker embedding from reference
        reference_mel = self.extract_mel(reference_audio)
        speaker_embedding = self.speaker_encoder(reference_mel)
        
        # Generate mel spectrogram conditioned on speaker
        mel = self.acoustic_model.generate(
            text,
            speaker_embedding=speaker_embedding
        )
        
        # Vocoder: mel → waveform
        audio = self.vocoder(mel)
        
        return audio

# Usage
tts = VoiceCloningTTS(acoustic_model, vocoder, speaker_encoder)

# Clone voice from 5-second reference
reference_audio = load_audio("reference_voice.wav")
cloned_speech = tts.clone_voice(
    "Hello, this is a cloned voice!",
    reference_audio
)
```

**2. Fine-Tuning (10-60 minutes of data)**

```python
class VoiceCloner:
    """
    Fine-tune pre-trained TTS on target voice
    """
    
    def __init__(self, pretrained_model):
        self.model = pretrained_model
    
    def fine_tune(
        self,
        target_voice_data: list[tuple],  # [(audio, text), ...]
        num_steps: int = 1000,
        learning_rate: float = 1e-4
    ):
        """
        Fine-tune model on target voice
        
        Args:
            target_voice_data: Audio-text pairs for target speaker
            num_steps: Training steps
            learning_rate: Learning rate
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for step in range(num_steps):
            # Sample batch
            batch = random.sample(target_voice_data, min(8, len(target_voice_data)))
            
            # Forward pass
            loss = self.model.compute_loss(batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
        
        return self.model

# Usage
cloner = VoiceCloner(pretrained_tts_model)

# Collect 30 minutes of target voice
target_data = [...]  # List of (audio, text) pairs

# Fine-tune
cloned_model = cloner.fine_tune(target_data, num_steps=1000)
```

---

## Production Deployment Patterns

### Pattern 1: Caching + Dynamic Generation

```python
class HybridTTSSystem:
    """
    Hybrid system: Cache common phrases, generate on-the-fly for novel text
    """
    
    def __init__(self, tts_model, cache_backend):
        self.tts = tts_model
        self.cache = cache_backend  # e.g., Redis
        self.common_phrases = [
            "Welcome", "Thank you", "Goodbye",
            "Please hold", "One moment please"
        ]
        self._warm_cache()
    
    def _warm_cache(self):
        """Pre-generate and cache common phrases"""
        for phrase in self.common_phrases:
            if not self.cache.exists(phrase):
                audio = self.tts.synthesize(phrase)
                self.cache.set(phrase, audio, ttl=86400)  # 24-hour TTL
    
    def synthesize(self, text: str):
        """
        Synthesize with caching
        
        Cache hit: <5ms
        Cache miss: 100-300ms (generate)
        """
        # Check cache
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        
        # Generate
        audio = self.tts.synthesize(text)
        
        # Cache if frequently requested
        request_count = self.cache.increment(f"count:{text}")
        if request_count > 10:
            self.cache.set(text, audio, ttl=3600)
        
        return audio
```

### Pattern 2: Streaming TTS

```python
class StreamingTTS:
    """
    Stream audio as it's generated (reduce latency)
    """
    
    def __init__(self, acoustic_model, vocoder):
        self.acoustic_model = acoustic_model
        self.vocoder = vocoder
    
    def stream_synthesize(self, text: str):
        """
        Generate audio in chunks
        
        Yields audio chunks as they're ready
        """
        # Generate mel spectrogram
        mel_frames = self.acoustic_model.generate_streaming(text)
        
        # Stream vocoder output
        mel_buffer = []
        chunk_size = 50  # mel frames per chunk
        
        for mel_frame in mel_frames:
            mel_buffer.append(mel_frame)
            
            if len(mel_buffer) >= chunk_size:
                # Vocoder: mel chunk → audio chunk
                mel_chunk = torch.stack(mel_buffer)
                audio_chunk = self.vocoder(mel_chunk)
                
                yield audio_chunk
                
                # Keep overlap for smoothness
                mel_buffer = mel_buffer[-10:]
        
        # Final chunk
        if mel_buffer:
            mel_chunk = torch.stack(mel_buffer)
            audio_chunk = self.vocoder(mel_chunk)
            yield audio_chunk

# Usage
streaming_tts = StreamingTTS(acoustic_model, vocoder)

# Stream audio
for audio_chunk in streaming_tts.stream_synthesize("Long text to synthesize..."):
    # Play audio_chunk immediately
    play_audio(audio_chunk)
    # User starts hearing speech before full generation completes!
```

### Pattern 3: Edge Deployment

```python
class EdgeTTS:
    """
    TTS optimized for edge devices (phones, IoT)
    """
    
    def __init__(self):
        self.model = self.load_optimized_model()
    
    def load_optimized_model(self):
        """
        Load quantized, pruned model
        
        Techniques:
        - INT8 quantization (4x smaller, 2-4x faster)
        - Knowledge distillation (smaller student model)
        - Pruning (remove 30-50% of weights)
        """
        import torch.quantization
        
        # Load full precision model
        model = load_full_model()
        
        # Quantize to INT8
        # Use dynamic quantization for linear-heavy modules as a safe default
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        return model_quantized
    
    def synthesize_on_device(self, text: str):
        """
        Synthesize on edge device
        
        Latency: 50-150ms
        Memory: <100MB
        """
        audio = self.model.generate(text)
        return audio
```

---

## Quality Assessment in Production

### Automated Quality Monitoring

```python
class TTSQualityMonitor:
    """
    Monitor TTS quality in production
    """
    
    def __init__(self):
        self.baseline_mcd = 2.5  # Expected MCD
        self.alert_threshold = 3.5  # Alert if MCD > this
    
    def monitor_synthesis(self, text: str, generated_audio: np.ndarray):
        """
        Check quality of generated audio
        
        Red flags:
        - Abnormal duration
        - Clipping / distortion
        - Silent segments
        - MCD drift
        """
        issues = []
        
        # Check duration
        expected_duration = len(text.split()) * 0.5  # ~0.5s per word
        actual_duration = len(generated_audio) / 22050
        if abs(actual_duration - expected_duration) / expected_duration > 0.5:
            issues.append(f"Abnormal duration: {actual_duration:.2f}s vs {expected_duration:.2f}s expected")
        
        # Check for clipping
        if np.max(np.abs(generated_audio)) > 0.99:
            issues.append("Clipping detected")
        
        # Check for silent segments
        rms = librosa.feature.rms(y=generated_audio)[0]
        silent_ratio = (rms < 0.01).sum() / len(rms)
        if silent_ratio > 0.3:
            issues.append(f"Too much silence: {silent_ratio:.1%}")
        
        # Log for drift detection
        self.log_quality_metrics({
            'text_length': len(text),
            'audio_duration': actual_duration,
            'max_amplitude': np.max(np.abs(generated_audio)),
            'silent_ratio': silent_ratio
        })
        
        return issues
    
    def log_quality_metrics(self, metrics: dict):
        """Log metrics for drift detection"""
        # Send to monitoring system (Datadog, Prometheus, etc.)
        pass
```

---

## Comparative Analysis

### Tacotron 2 vs FastSpeech 2

| Aspect | Tacotron 2 | FastSpeech 2 |
|--------|-----------|--------------|
| **Speed** | Slow (autoregressive) | Fast (parallel) |
| **Quality** | Excellent | Excellent |
| **Robustness** | Can skip/repeat words | More robust |
| **Controllability** | Limited | Explicit control (pitch, duration) |
| **Training** | Simpler (no duration model) | Needs duration labels |
| **Latency** | 200-500ms | 50-150ms |

### When to Use Each

**Use Tacotron 2 when:**
- Maximum naturalness is critical
- Training data is limited (easier to train)
- Latency is acceptable

**Use FastSpeech 2 when:**
- Low latency required
- Need prosody control
- Robustness is critical (production systems)

---

## Recent Advances (2023-2024)

### 1. VALL-E (Zero-Shot Voice Cloning)

Microsoft's VALL-E can clone a voice from a 3-second sample using language model approach.

**Key idea:** Treat TTS as conditional language modeling over discrete audio codes.

### 2. VITS (End-to-End TTS)

Combines acoustic model and vocoder into single model.

**Advantages:**
- Faster training and inference
- Better audio quality
- Simplified pipeline

### 3. YourTTS (Multi-lingual Voice Cloning)

Zero-shot multi-lingual TTS supporting 16+ languages.

### 4. Bark (Generative Audio Model)

Text-to-audio model that can generate music, sound effects, and speech with emotions.

---

## Key Takeaways

✅ **Two-stage pipeline** - Acoustic model + vocoder is standard  
✅ **Text processing critical** - Normalization and G2P affect quality  
✅ **Autoregressive vs non-autoregressive** - Tacotron vs FastSpeech trade-offs  
✅ **Prosody control** - Pitch, duration, energy for expressiveness  
✅ **Multiple metrics** - Objective (MCD) and subjective (MOS) both needed  
✅ **Production optimization** - Latency, caching, streaming for real-time use  
✅ **Like climbing stairs** - Build incrementally (phoneme → mel → waveform)  

---

**Originally published at:** [arunbaby.com/speech-tech/0006-text-to-speech-basics](https://www.arunbaby.com/speech-tech/0006-text-to-speech-basics/)

*If you found this helpful, consider sharing it with others who might benefit.*

