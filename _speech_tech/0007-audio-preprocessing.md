---
title: "Audio Preprocessing & Signal Processing"
day: 7
collection: speech_tech
categories:
  - speech-tech
tags:
  - audio-preprocessing
  - signal-processing
  - noise-reduction
  - data-augmentation
subdomain: Audio Processing
tech_stack: [Python, Librosa, PyDub, SciPy, NumPy, sox]
scale: "Real-time streaming"
companies: [Google, Amazon, Microsoft, Apple, Meta, Spotify]
related_dsa_day: 7
related_ml_day: 7
related_agents_day: 7
---

**Clean audio is the foundation of robust speech systems, master preprocessing pipelines that handle real-world noise and variability.**

## Introduction

**Audio preprocessing** transforms raw audio into clean, standardized representations suitable for ML models.

**Why it matters:**
- **Garbage in, garbage out:** Poor audio quality destroys model performance
- **Real-world audio is messy:** Background noise, varying volumes, different devices
- **Standardization:** Models expect consistent input formats
- **Data augmentation:** Increase training data diversity

**Pipeline overview:**

```
Raw Audio (microphone)
   ↓
[Loading & Format Conversion]
   ↓
[Resampling]
   ↓
[Normalization]
   ↓
[Noise Reduction]
   ↓
[Voice Activity Detection]
   ↓
[Segmentation]
   ↓
[Feature Extraction]
   ↓
Clean Features → Model
```

---

## Audio Fundamentals

### Digital Audio Representation

```
Analog Sound Wave:
  ∿∿∿∿∿∿∿∿∿∿∿

Sampling (digitization):
  ●─●─●─●─●─●─●─●  (sample points)

Key parameters:
- Sample Rate: Samples per second (Hz)
  - CD quality: 44,100 Hz
  - Speech: 16,000 Hz or 22,050 Hz
  - Telephone: 8,000 Hz

- Bit Depth: Bits per sample
  - 16-bit: 65,536 possible values
  - 24-bit: 16,777,216 values
  - 32-bit float: Highest precision

- Channels:
  - Mono: 1 channel
  - Stereo: 2 channels (left, right)
```

### Nyquist-Shannon Sampling Theorem

**Rule:** To capture frequency `f`, sample rate must be ≥ 2f

```
Human hearing: 20 Hz - 20 kHz
→ Need ≥40 kHz sample rate
→ CD uses 44.1 kHz (margin above 40 kHz)

Speech frequencies: ~300 Hz - 8 kHz
→ 16 kHz sample rate is sufficient
```

---

## Loading & Format Conversion

### Loading Audio

```python
import librosa
import soundfile as sf
import numpy as np

def load_audio(file_path, sr=None):
    """
    Load audio file
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (None = keep original)
    
    Returns:
        audio: np.array of samples
        sr: Sample rate
    """
    # Librosa (resamples automatically)
    audio, sample_rate = librosa.load(file_path, sr=sr)
    
    return audio, sample_rate

# Example
audio, sr = load_audio('speech.wav', sr=16000)
print(f"Shape: {audio.shape}, Sample Rate: {sr} Hz")
print(f"Duration: {len(audio) / sr:.2f} seconds")
```

### Format Conversion

```python
from pydub import AudioSegment

def convert_audio_format(input_path, output_path, output_format='wav'):
    """
    Convert between audio formats
    
    Supports: mp3, wav, ogg, flac, m4a, etc.
    """
    audio = AudioSegment.from_file(input_path)
    
    # Export in new format
    audio.export(output_path, format=output_format)

# Convert MP3 to WAV
convert_audio_format('input.mp3', 'output.wav', 'wav')
```

### Mono/Stereo Conversion

```python
def stereo_to_mono(audio_stereo):
    """
    Convert stereo to mono by averaging channels
    
    Args:
        audio_stereo: Shape (2, n_samples) or (n_samples, 2)
    
    Returns:
        audio_mono: Shape (n_samples,)
    """
    if audio_stereo.ndim == 1:
        # Already mono
        return audio_stereo
    
    # Average across channels
    if audio_stereo.shape[0] == 2:
        # Shape: (2, n_samples)
        return np.mean(audio_stereo, axis=0)
    else:
        # Shape: (n_samples, 2)
        return np.mean(audio_stereo, axis=1)

# Example
audio_stereo, sr = librosa.load('stereo.wav', sr=None, mono=False)
audio_mono = stereo_to_mono(audio_stereo)
```

---

## Resampling

**Purpose:** Convert sample rate to match model requirements

### High-Quality Resampling

```python
import librosa

def resample_audio(audio, orig_sr, target_sr):
    """
    Resample audio using high-quality algorithm
    
    Args:
        audio: Audio samples
        orig_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        resampled_audio
    """
    if orig_sr == target_sr:
        return audio
    
    # Librosa uses high-quality resampling (Kaiser window)
    resampled = librosa.resample(
        audio,
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type='kaiser_best'  # Highest quality
    )
    
    return resampled

# Example: Downsample 44.1 kHz to 16 kHz
audio_44k, _ = librosa.load('audio.wav', sr=44100)
audio_16k = resample_audio(audio_44k, orig_sr=44100, target_sr=16000)

print(f"Original length: {len(audio_44k)}")
print(f"Resampled length: {len(audio_16k)}")
print(f"Ratio: {len(audio_44k) / len(audio_16k):.2f}")  # ~2.76
```

**Resampling visualization:**

```
Original (44.1 kHz):
●●●●●●●●●●●●●●●●●●●●●●●●●●●●  (44,100 samples/second)

Downsampled (16 kHz):
●───●───●───●───●───●───●───●  (16,000 samples/second)

Algorithm interpolates to avoid aliasing
```

---

## Normalization

### Amplitude Normalization

```python
def normalize_audio(audio, target_level=-20.0):
    """
    Normalize audio to target level (dB)
    
    Args:
        audio: Audio samples
        target_level: Target RMS level in dB
    
    Returns:
        normalized_audio
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Convert target level from dB to linear
    target_rms = 10 ** (target_level / 20.0)
    
    # Scale audio
    scaling_factor = target_rms / rms
    normalized = audio * scaling_factor
    
    # Clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized

# Example
audio, sr = librosa.load('speech.wav', sr=16000)
normalized_audio = normalize_audio(audio, target_level=-20.0)
```

### Peak Normalization

```python
def peak_normalize(audio):
    """
    Normalize to peak amplitude = 1.0
    
    Simple but can be problematic if audio has spikes
    """
    peak = np.max(np.abs(audio))
    
    if peak == 0:
        return audio
    
    return audio / peak
```

### DC Offset Removal

```python
def remove_dc_offset(audio):
    """
    Remove DC bias (mean offset)
    
    DC offset can cause clicking sounds
    """
    return audio - np.mean(audio)

# Example
audio_clean = remove_dc_offset(audio)
```

---

## Noise Reduction

### 1. Spectral Subtraction

```python
import scipy.signal as signal

def spectral_subtraction(audio, sr, noise_duration=0.5):
    """
    Reduce noise using spectral subtraction
    
    Assumes first noise_duration seconds are noise only
    
    Args:
        audio: Audio signal
        sr: Sample rate
        noise_duration: Duration of noise-only segment (seconds)
    
    Returns:
        denoised_audio
    """
    # Extract noise profile from beginning
    noise_samples = int(noise_duration * sr)
    noise_segment = audio[:noise_samples]
    
    # Compute noise spectrum
    noise_fft = np.fft.rfft(noise_segment)
    noise_power = np.abs(noise_fft) ** 2
    noise_power_avg = np.mean(noise_power)
    
    # STFT of full audio
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=1024)
    
    # Subtract noise spectrum
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Spectral subtraction
    noise_estimate = np.sqrt(noise_power_avg)
    magnitude_denoised = np.maximum(magnitude - noise_estimate, 0.0)
    
    # Reconstruct
    Zxx_denoised = magnitude_denoised * np.exp(1j * phase)
    _, audio_denoised = signal.istft(Zxx_denoised, fs=sr)
    
    return audio_denoised[:len(audio)]

# Example
audio, sr = librosa.load('noisy_speech.wav', sr=16000)
denoised = spectral_subtraction(audio, sr, noise_duration=0.5)
```

### 2. Wiener Filtering

```python
def wiener_filter(audio, sr, noise_reduction_factor=0.5):
    """
    Apply Wiener filter for noise reduction
    
    More sophisticated than spectral subtraction
    """
    from scipy.signal import wiener
    
    # Apply Wiener filter
    filtered = wiener(audio, mysize=5, noise=noise_reduction_factor)
    
    return filtered
```

### 3. High-Pass Filter (Remove Low-Frequency Noise)

```python
def high_pass_filter(audio, sr, cutoff_freq=80):
    """
    Remove low-frequency noise (e.g., rumble, hum)
    
    Args:
        audio: Audio signal
        sr: Sample rate
        cutoff_freq: Cutoff frequency in Hz
    
    Returns:
        filtered_audio
    """
    from scipy.signal import butter, filtfilt
    
    # Design high-pass filter
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(N=5, Wn=normalized_cutoff, btype='high')
    
    # Apply filter (zero-phase filtering)
    filtered = filtfilt(b, a, audio)
    
    return filtered

# Example: Remove rumble below 80 Hz
audio_filtered = high_pass_filter(audio, sr=16000, cutoff_freq=80)
```

---

## Voice Activity Detection (VAD)

**Purpose:** Identify speech segments, remove silence

```python
import librosa

def voice_activity_detection(audio, sr, frame_length=2048, hop_length=512, energy_threshold=0.02):
    """
    Simple energy-based VAD
    
    Args:
        audio: Audio signal
        sr: Sample rate
        energy_threshold: Threshold for voice activity
    
    Returns:
        speech_segments: List of (start_sample, end_sample) tuples
    """
    # Compute frame energy
    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Normalize energy
    energy_normalized = energy / (np.max(energy) + 1e-8)
    
    # Threshold to get voice activity
    voice_activity = energy_normalized > energy_threshold
    
    # Convert to sample indices
    def frame_to_sample(frame_idx):
        start = frame_idx * hop_length
        end = min(start + frame_length, len(audio))
        return start, end
    
    # Find continuous speech segments
    segments = []
    in_speech = False
    start_frame = 0
    
    for i, is_voice in enumerate(voice_activity):
        if is_voice and not in_speech:
            # Start of speech
            start_frame = i
            in_speech = True
        elif not is_voice and in_speech:
            # End of speech
            end_frame = i
            start_sample, _ = frame_to_sample(start_frame)
            end_sample, _ = frame_to_sample(end_frame)
            segments.append((start_sample, end_sample))
            in_speech = False
    
    # Handle case where speech goes to end
    if in_speech:
        start_sample, _ = frame_to_sample(start_frame)
        end_sample = len(audio)
        segments.append((start_sample, end_sample))
    
    return segments

# Example
audio, sr = librosa.load('speech_with_pauses.wav', sr=16000)
segments = voice_activity_detection(audio, sr)

print(f"Found {len(segments)} speech segments:")
for i, (start, end) in enumerate(segments):
    duration = (end - start) / sr
    print(f"  Segment {i+1}: {start/sr:.2f}s - {end/sr:.2f}s ({duration:.2f}s)")
```

**VAD visualization:**

```
Audio waveform:
     ___           ___       ___
    /   \         /   \     /   \
___/     \______/     \___/     \___

Energy:
    ████          ████      ████
    ████          ████      ████
────████──────────████──────████────  ← threshold

VAD output:
    SSSS          SSSS      SSSS
    (S = Speech,  spaces = Silence)
```

---

## Segmentation

### Fixed-Length Segmentation

```python
def segment_audio_fixed_length(audio, sr, segment_duration=3.0, hop_duration=1.0):
    """
    Segment audio into fixed-length chunks with overlap
    
    Args:
        audio: Audio signal
        sr: Sample rate
        segment_duration: Segment length in seconds
        hop_duration: Hop between segments in seconds
    
    Returns:
        segments: List of audio segments
    """
    segment_samples = int(segment_duration * sr)
    hop_samples = int(hop_duration * sr)
    
    segments = []
    start = 0
    
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop_samples
    
    return segments

# Example: 3-second segments with 1-second hop (2-second overlap)
segments = segment_audio_fixed_length(audio, sr=16000, segment_duration=3.0, hop_duration=1.0)
print(f"Created {len(segments)} segments")
```

### Adaptive Segmentation (Based on Pauses)

```python
def segment_by_pauses(audio, sr, min_silence_duration=0.3, silence_threshold=0.02):
    """
    Segment audio at silence/pause points
    
    Better than fixed-length for natural speech
    """
    # Detect voice activity
    speech_segments = voice_activity_detection(
        audio, sr,
        energy_threshold=silence_threshold
    )
    
    # Filter out very short segments
    min_segment_samples = int(min_silence_duration * sr)
    filtered_segments = [
        (start, end) for start, end in speech_segments
        if end - start >= min_segment_samples
    ]
    
    # Extract audio segments
    audio_segments = []
    for start, end in filtered_segments:
        segment = audio[start:end]
        audio_segments.append(segment)
    
    return audio_segments, filtered_segments

# Example
audio_segments, timestamps = segment_by_pauses(audio, sr=16000)
```

---

## Data Augmentation

**Purpose:** Increase training data diversity, improve model robustness

### 1. Time Stretching

```python
def time_stretch(audio, rate=1.0):
    """
    Speed up or slow down audio without changing pitch
    
    Args:
        audio: Audio signal
        rate: Stretch factor
              > 1.0: speed up
              < 1.0: slow down
    
    Returns:
        stretched_audio
    """
    return librosa.effects.time_stretch(audio, rate=rate)

# Example: Speed up by 20%
audio_fast = time_stretch(audio, rate=1.2)

# Slow down by 20%
audio_slow = time_stretch(audio, rate=0.8)
```

### 2. Pitch Shifting

```python
def pitch_shift(audio, sr, n_steps=2):
    """
    Shift pitch without changing speed
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_steps: Semitones to shift (positive = higher, negative = lower)
    
    Returns:
        pitch_shifted_audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

# Example: Shift up 2 semitones
audio_high = pitch_shift(audio, sr=16000, n_steps=2)

# Shift down 2 semitones
audio_low = pitch_shift(audio, sr=16000, n_steps=-2)
```

### 3. Adding Noise

```python
def add_noise(audio, noise_factor=0.005):
    """
    Add random Gaussian noise
    
    Args:
        audio: Audio signal
        noise_factor: Standard deviation of noise
    
    Returns:
        noisy_audio
    """
    noise = np.random.randn(len(audio)) * noise_factor
    return audio + noise

# Example
audio_noisy = add_noise(audio, noise_factor=0.01)
```

### 4. Background Noise Mixing

```python
def mix_background_noise(speech_audio, noise_audio, snr_db=10):
    """
    Mix speech with background noise at specified SNR
    
    Args:
        speech_audio: Clean speech
        noise_audio: Background noise
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        mixed_audio
    """
    # Match lengths
    if len(noise_audio) < len(speech_audio):
        # Repeat noise to match speech length
        repeats = int(np.ceil(len(speech_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)[:len(speech_audio)]
    else:
        # Trim noise
        noise_audio = noise_audio[:len(speech_audio)]
    
    # Calculate signal and noise power
    speech_power = np.mean(speech_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    
    # Calculate scaling factor for noise
    snr_linear = 10 ** (snr_db / 10)
    noise_scaling = np.sqrt(speech_power / (snr_linear * noise_power))
    
    # Mix
    mixed = speech_audio + noise_scaling * noise_audio
    
    # Normalize to prevent clipping
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-8)
    
    return mixed

# Example: Mix with café noise at SNR=15dB
cafe_noise, _ = librosa.load('cafe_background.wav', sr=16000)
noisy_speech = mix_background_noise(audio, cafe_noise, snr_db=15)
```

### 5. SpecAugment (For Spectrograms)

```python
def spec_augment(mel_spectrogram, num_mask=2, freq_mask_param=20, time_mask_param=30):
    """
    SpecAugment: mask random time-frequency patches
    
    Popular augmentation for speech recognition
    
    Args:
        mel_spectrogram: Shape (n_mels, time)
        num_mask: Number of masks to apply
        freq_mask_param: Max width of frequency mask
        time_mask_param: Max width of time mask
    
    Returns:
        augmented_spectrogram
    """
    aug_spec = mel_spectrogram.copy()
    n_mels, n_frames = aug_spec.shape
    
    # Frequency masking
    for _ in range(num_mask):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, n_mels - f)
        aug_spec[f0:f0+f, :] = 0
    
    # Time masking
    for _ in range(num_mask):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, n_frames - t)
        aug_spec[:, t0:t0+t] = 0
    
    return aug_spec

# Example
mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000)
aug_mel_spec = spec_augment(mel_spec, num_mask=2)
```

---

## Connection to Feature Engineering (Day 7)

Audio preprocessing is feature engineering for speech:

```python
class AudioFeatureEngineeringPipeline:
    """
    Complete pipeline: raw audio → features
    
    Similar to general ML feature engineering
    """
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def process(self, audio_path):
        """
        Full preprocessing pipeline
        
        Analogous to feature engineering pipeline in ML
        """
        # 1. Load (like data loading)
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        # 2. Normalize (like feature scaling)
        audio = normalize_audio(audio)
        
        # 3. Noise reduction (like outlier removal)
        audio = high_pass_filter(audio, sr)
        
        # 4. VAD (like removing null values)
        segments = voice_activity_detection(audio, sr)
        
        # 5. Feature extraction (like creating derived features)
        features = self.extract_features(audio, sr)
        
        return features
    
    def extract_features(self, audio, sr):
        """
        Extract multiple feature types
        
        Like creating feature crosses and aggregations
        """
        features = {}
        
        # Spectral features (numerical features)
        features['mfcc'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
        
        # Temporal features (time-based features)
        features['rms_energy'] = librosa.feature.rms(y=audio)
        
        # Aggregations (like SQL GROUP BY)
        features['mfcc_mean'] = np.mean(features['mfcc'], axis=1)
        features['mfcc_std'] = np.std(features['mfcc'], axis=1)
        
        return features
```

---

## Production Pipeline

```python
class ProductionAudioPreprocessor:
    """
    Production-ready audio preprocessing
    
    Handles errors, logging, monitoring
    """
    
    def __init__(self, config):
        self.sr = config.get('sample_rate', 16000)
        self.normalize_level = config.get('normalize_level', -20.0)
        self.enable_vad = config.get('enable_vad', True)
    
    def preprocess(self, audio_bytes):
        """
        Preprocess audio from bytes
        
        Returns: (processed_audio, metadata, success)
        """
        metadata = {}
        
        try:
            # Load from bytes
            audio = self._load_from_bytes(audio_bytes)
            metadata['original_length'] = len(audio)
            
            # Resample
            if self.sr != 16000:  # Assuming input is 16kHz
                audio = resample_audio(audio, 16000, self.sr)
            
            # Normalize
            audio = normalize_audio(audio, self.normalize_level)
            metadata['normalized'] = True
            
            # VAD
            if self.enable_vad:
                segments = voice_activity_detection(audio, self.sr)
                if segments:
                    # Keep only speech
                    speech_audio = np.concatenate([
                        audio[start:end] for start, end in segments
                    ])
                    audio = speech_audio
                    metadata['vad_segments'] = len(segments)
            
            metadata['final_length'] = len(audio)
            metadata['duration_seconds'] = len(audio) / self.sr
            
            return audio, metadata, True
        
        except Exception as e:
            return None, {'error': str(e)}, False
    
    def _load_from_bytes(self, audio_bytes):
        """Load audio from bytes"""
        import io
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sr)
        return audio
```

---

## Real-World Challenges & Solutions

### Challenge 1: Codec Artifacts

**Problem:** Different audio codecs introduce artifacts

```python
def detect_codec_artifacts(audio, sr):
    """
    Detect codec artifacts (e.g., from MP3 compression)
    
    Returns: artifact_score (higher = more artifacts)
    """
    import scipy.signal as signal
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(audio, fs=sr)
    
    # MP3 artifacts often appear as:
    # 1. High-frequency cutoff (lossy codecs)
    cutoff_freq = 16000  # Hz
    high_freq_mask = f > cutoff_freq
    high_freq_energy = np.mean(Sxx[high_freq_mask, :])
    
    # 2. Pre-echo artifacts
    # Sudden changes in energy
    energy = np.sum(Sxx, axis=0)
    energy_diff = np.diff(energy)
    pre_echo_score = np.std(energy_diff)
    
    artifact_score = {
        'high_freq_loss': high_freq_energy,
        'pre_echo': pre_echo_score,
        'overall': 1.0 - high_freq_energy + pre_echo_score
    }
    
    return artifact_score

# Example
audio_mp3, sr = librosa.load('compressed.mp3', sr=16000)
audio_wav, sr = librosa.load('lossless.wav', sr=16000)

artifacts_mp3 = detect_codec_artifacts(audio_mp3, sr)
artifacts_wav = detect_codec_artifacts(audio_wav, sr)

print(f"MP3 artifacts: {artifacts_mp3['overall']:.3f}")
print(f"WAV artifacts: {artifacts_wav['overall']:.3f}")
```

### Challenge 2: Variable Sample Rates

```python
class AdaptiveResampler:
    """
    Handle audio from various sources with different sample rates
    
    Production systems receive audio from:
    - Phone calls: 8 kHz
    - Bluetooth: 16 kHz  
    - Studio mics: 44.1 kHz / 48 kHz
    """
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.cache = {}  # Cache resampling filters
    
    def resample(self, audio, orig_sr):
        """
        Efficiently resample with caching
        """
        if orig_sr == self.target_sr:
            return audio
        
        # Check cache
        cache_key = (orig_sr, self.target_sr)
        if cache_key not in self.cache:
            # Compute resampling filter once
            self.cache[cache_key] = self._compute_filter(orig_sr, self.target_sr)
        
        # Apply cached filter
        return librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=self.target_sr,
            res_type='kaiser_fast'  # Good balance of quality/speed
        )
    
    def _compute_filter(self, orig_sr, target_sr):
        """Compute and cache resampling filter"""
        # In real implementation, would compute filter coefficients
        return None

# Usage
resampler = AdaptiveResampler(target_sr=16000)

# Handle various sources
phone_audio = resampler.resample(phone_audio, orig_sr=8000)
bluetooth_audio = resampler.resample(bluetooth_audio, orig_sr=16000)
studio_audio = resampler.resample(studio_audio, orig_sr=48000)
```

### Challenge 3: Clipping & Distortion

```python
def detect_and_fix_clipping(audio, threshold=0.99):
    """
    Detect clipped samples and attempt interpolation
    
    Args:
        audio: Audio signal
        threshold: Clipping threshold (absolute value)
    
    Returns:
        fixed_audio, was_clipped
    """
    # Detect clipping
    clipped_mask = np.abs(audio) >= threshold
    num_clipped = np.sum(clipped_mask)
    
    if num_clipped == 0:
        return audio, False
    
    print(f"⚠️ Detected {num_clipped} clipped samples ({100*num_clipped/len(audio):.2f}%)")
    
    # Simple interpolation for clipped regions
    fixed_audio = audio.copy()
    
    # Find clipped regions
    clipped_indices = np.where(clipped_mask)[0]
    
    for idx in clipped_indices:
        # Skip edges
        if idx == 0 or idx == len(audio) - 1:
            continue
        
        # Interpolate from neighbors
        if not clipped_mask[idx-1] and not clipped_mask[idx+1]:
            fixed_audio[idx] = (audio[idx-1] + audio[idx+1]) / 2
    
    return fixed_audio, True

# Example
audio_with_clipping, sr = librosa.load('clipped_audio.wav', sr=16000)
fixed_audio, was_clipped = detect_and_fix_clipping(audio_with_clipping)

if was_clipped:
    print("Applied clipping repair")
```

### Challenge 4: Background Babble Noise

```python
def reduce_babble_noise(audio, sr, noise_profile_duration=1.0):
    """
    Reduce background babble (multiple speakers)
    
    More challenging than stationary noise
    """
    import noisereduce as nr
    
    # Estimate noise profile from segments with lowest energy
    frame_length = int(0.1 * sr)  # 100ms frames
    hop_length = frame_length // 2
    
    # Compute frame energy
    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Select low-energy frames as noise
    noise_threshold = np.percentile(energy, 20)
    noise_frames = np.where(energy < noise_threshold)[0]
    
    # Extract noise samples
    noise_samples = []
    for frame_idx in noise_frames:
        start = frame_idx * hop_length
        end = start + frame_length
        if end <= len(audio):
            noise_samples.extend(audio[start:end])
    
    noise_profile = np.array(noise_samples)
    
    # Apply noise reduction
    if len(noise_profile) > sr * noise_profile_duration:
        reduced_noise = nr.reduce_noise(
            y=audio,
            y_noise=noise_profile[:int(sr * noise_profile_duration)],
            sr=sr,
            stationary=False,  # Non-stationary noise
            prop_decrease=0.8
        )
        return reduced_noise
    else:
        print("⚠️ Insufficient noise profile, returning original")
        return audio

# Example
audio_with_babble, sr = librosa.load('meeting_audio.wav', sr=16000)
clean_audio = reduce_babble_noise(audio_with_babble, sr)
```

---

## Audio Quality Metrics

### Signal-to-Noise Ratio (SNR)

```python
def calculate_snr(clean_signal, noisy_signal):
    """
    Calculate SNR in dB
    
    Args:
        clean_signal: Ground truth clean signal
        noisy_signal: Signal with noise
    
    Returns:
        SNR in dB
    """
    # Ensure same length
    min_len = min(len(clean_signal), len(noisy_signal))
    clean = clean_signal[:min_len]
    noisy = noisy_signal[:min_len]
    
    # Compute noise
    noise = noisy - clean
    
    # Power
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    # SNR in dB
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

# Example
clean, sr = librosa.load('clean_speech.wav', sr=16000)
noisy, sr = librosa.load('noisy_speech.wav', sr=16000)

snr = calculate_snr(clean, noisy)
print(f"SNR: {snr:.2f} dB")

# Typical SNRs:
# > 40 dB: Excellent
# 25-40 dB: Good
# 10-25 dB: Fair
# < 10 dB: Poor
```

### Perceptual Evaluation of Speech Quality (PESQ)

```python
# PESQ is a standard metric for speech quality
# Requires pesq library: pip install pesq

from pesq import pesq

def evaluate_speech_quality(reference_audio, degraded_audio, sr=16000):
    """
    Evaluate speech quality using PESQ
    
    Args:
        reference_audio: Clean reference
        degraded_audio: Processed/degraded audio
        sr: Sample rate (8000 or 16000)
    
    Returns:
        PESQ score (1.0 to 4.5, higher is better)
    """
    # PESQ requires 8kHz or 16kHz
    if sr not in [8000, 16000]:
        raise ValueError("PESQ requires sr=8000 or sr=16000")
    
    # Ensure same length
    min_len = min(len(reference_audio), degraded_audio)
    ref = reference_audio[:min_len]
    deg = degraded_audio[:min_len]
    
    # Compute PESQ
    if sr == 8000:
        mode = 'nb'  # Narrowband
    else:
        mode = 'wb'  # Wideband
    
    score = pesq(sr, ref, deg, mode)
    
    return score

# Example
reference, sr = librosa.load('clean.wav', sr=16000)
processed, sr = librosa.load('processed.wav', sr=16000)

quality_score = evaluate_speech_quality(reference, processed, sr)
print(f"PESQ Score: {quality_score:.2f}")

# PESQ interpretation:
# 4.0+: Excellent
# 3.0-4.0: Good
# 2.0-3.0: Fair
# < 2.0: Poor
```

---

## Advanced Augmentation Strategies

### Room Impulse Response (RIR) Convolution

```python
def apply_room_impulse_response(speech, rir):
    """
    Simulate room acoustics using RIR
    
    Makes model robust to reverberation
    
    Args:
        speech: Clean speech signal
        rir: Room impulse response
    
    Returns:
        Reverberant speech
    """
    from scipy.signal import fftconvolve
    
    # Convolve speech with RIR
    reverb_speech = fftconvolve(speech, rir, mode='same')
    
    # Normalize
    reverb_speech = reverb_speech / (np.max(np.abs(reverb_speech)) + 1e-8)
    
    return reverb_speech

# Example: Generate synthetic RIR
def generate_synthetic_rir(sr=16000, room_size='medium', rt60=0.5):
    """
    Generate synthetic room impulse response
    
    Args:
        sr: Sample rate
        room_size: 'small', 'medium', 'large'
        rt60: Reverberation time (seconds)
    
    Returns:
        RIR signal
    """
    # Duration based on RT60
    duration = int(rt60 * sr)
    
    # Exponential decay
    t = np.arange(duration) / sr
    decay = np.exp(-6.91 * t / rt60)  # -60 dB decay
    
    # Add random reflections
    rir = decay * np.random.randn(duration)
    
    # Initial spike (direct path)
    rir[0] = 1.0
    
    # Normalize
    rir = rir / np.max(np.abs(rir))
    
    return rir

# Usage
clean_speech, sr = librosa.load('speech.wav', sr=16000)

# Simulate different rooms
small_room_rir = generate_synthetic_rir(sr, 'small', rt60=0.3)
large_room_rir = generate_synthetic_rir(sr, 'large', rt60=1.2)

speech_small_room = apply_room_impulse_response(clean_speech, small_room_rir)
speech_large_room = apply_room_impulse_response(clean_speech, large_room_rir)
```

### Codec Simulation

```python
def simulate_codec(audio, sr, codec='mp3', bitrate=32):
    """
    Simulate lossy codec compression
    
    Makes model robust to codec artifacts
    
    Args:
        audio: Clean audio
        sr: Sample rate
        codec: 'mp3', 'aac', 'opus'
        bitrate: Bitrate in kbps
    
    Returns:
        Codec-compressed audio
    """
    import subprocess
    import tempfile
    import os
    import soundfile as sf
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
        sf.write(tmp_in.name, audio, sr)
        input_path = tmp_in.name
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
        output_path = tmp_out.name
    
    try:
        # Compress with ffmpeg
        if codec == 'mp3':
            subprocess.run([
                'ffmpeg', '-i', input_path,
                '-codec:a', 'libmp3lame',
                '-b:a', f'{bitrate}k',
                '-y', output_path
            ], capture_output=True, check=True)
        elif codec == 'opus':
            subprocess.run([
                'ffmpeg', '-i', input_path,
                '-codec:a', 'libopus',
                '-b:a', f'{bitrate}k',
                '-y', output_path
            ], capture_output=True, check=True)
        
        # Load compressed audio
        compressed_audio, _ = librosa.load(output_path, sr=sr)
        
        return compressed_audio
    
    finally:
        # Cleanup
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

# Usage
audio, sr = librosa.load('clean.wav', sr=16000)

# Simulate low-bitrate compression
audio_32kbps = simulate_codec(audio, sr, codec='mp3', bitrate=32)
audio_64kbps = simulate_codec(audio, sr, codec='mp3', bitrate=64)
```

### Dynamic Range Compression

```python
def dynamic_range_compression(audio, threshold=-20, ratio=4, attack=0.005, release=0.1, sr=16000):
    """
    Apply dynamic range compression (like audio compressors)
    
    Reduces loudness variation, simulating broadcast audio
    
    Args:
        audio: Input audio
        threshold: Threshold in dB
        ratio: Compression ratio (4:1 means 4dB input → 1dB output above threshold)
        attack: Attack time in seconds
        release: Release time in seconds
        sr: Sample rate
    
    Returns:
        Compressed audio
    """
    # Convert to dB
    audio_db = 20 * np.log10(np.abs(audio) + 1e-8)
    
    # Compute gain reduction
    gain_db = np.zeros_like(audio_db)
    
    for i in range(len(audio_db)):
        if audio_db[i] > threshold:
            # Above threshold: apply compression
            excess_db = audio_db[i] - threshold
            gain_db[i] = -excess_db * (1 - 1/ratio)
        else:
            gain_db[i] = 0
    
    # Smooth gain reduction (attack/release)
    attack_samples = int(attack * sr)
    release_samples = int(release * sr)
    
    smoothed_gain = np.zeros_like(gain_db)
    for i in range(1, len(gain_db)):
        if gain_db[i] < smoothed_gain[i-1]:
            # Attack
            alpha = 1 - np.exp(-1 / attack_samples)
        else:
            # Release
            alpha = 1 - np.exp(-1 / release_samples)
        
        smoothed_gain[i] = alpha * gain_db[i] + (1 - alpha) * smoothed_gain[i-1]
    
    # Apply gain
    gain_linear = 10 ** (smoothed_gain / 20)
    compressed = audio * gain_linear
    
    return compressed

# Example
audio, sr = librosa.load('speech.wav', sr=16000)
compressed = dynamic_range_compression(audio, threshold=-20, ratio=4, sr=sr)
```

---

## End-to-End Preprocessing Pipeline

```python
class ProductionAudioPipeline:
    """
    Complete production-ready preprocessing pipeline
    
    Handles all edge cases and monitors quality
    """
    
    def __init__(self, config):
        self.target_sr = config.get('sample_rate', 16000)
        self.target_duration = config.get('target_duration', None)
        self.enable_noise_reduction = config.get('noise_reduction', True)
        self.enable_vad = config.get('vad', True)
        self.augmentation_enabled = config.get('augmentation', False)
        
        self.stats = {
            'processed': 0,
            'failed': 0,
            'clipped': 0,
            'too_short': 0,
            'avg_snr': []
        }
    
    def process(self, audio_path):
        """
        Process single audio file
        
        Returns: (processed_audio, metadata, success)
        """
        metadata = {'original_path': audio_path}
        
        try:
            # 1. Load
            audio, orig_sr = librosa.load(audio_path, sr=None)
            metadata['original_sr'] = orig_sr
            metadata['original_duration'] = len(audio) / orig_sr
            
            # 2. Detect issues
            clipped = np.max(np.abs(audio)) >= 0.99
            if clipped:
                audio, _ = detect_and_fix_clipping(audio)
                self.stats['clipped'] += 1
                metadata['had_clipping'] = True
            
            # 3. Resample
            if orig_sr != self.target_sr:
                audio = resample_audio(audio, orig_sr, self.target_sr)
                metadata['resampled'] = True
            
            # 4. Normalize
            audio = normalize_audio(audio, target_level=-20.0)
            metadata['normalized'] = True
            
            # 5. Noise reduction
            if self.enable_noise_reduction:
                audio = high_pass_filter(audio, self.target_sr, cutoff_freq=80)
                metadata['noise_reduction'] = True
            
            # 6. Voice Activity Detection
            if self.enable_vad:
                segments = voice_activity_detection(audio, self.target_sr)
                if segments:
                    speech_audio = np.concatenate([
                        audio[start:end] for start, end in segments
                    ])
                    audio = speech_audio
                    metadata['vad_segments'] = len(segments)
                else:
                    # No speech detected
                    return None, {'error': 'No speech detected'}, False
            
            # 7. Duration handling
            current_duration = len(audio) / self.target_sr
            
            if self.target_duration:
                target_samples = int(self.target_duration * self.target_sr)
                
                if len(audio) < target_samples:
                    # Pad
                    audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
                    metadata['padded'] = True
                elif len(audio) > target_samples:
                    # Trim
                    audio = audio[:target_samples]
                    metadata['trimmed'] = True
            
            # 8. Quality checks
            if len(audio) < 0.5 * self.target_sr:  # Less than 0.5 seconds
                self.stats['too_short'] += 1
                return None, {'error': 'Too short after VAD'}, False
            
            # 9. Augmentation (training only)
            if self.augmentation_enabled:
                audio = self._augment(audio)
                metadata['augmented'] = True
            
            # 10. Final normalization
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
            
            metadata['final_duration'] = len(audio) / self.target_sr
            metadata['final_samples'] = len(audio)
            
            self.stats['processed'] += 1
            
            return audio, metadata, True
        
        except Exception as e:
            self.stats['failed'] += 1
            return None, {'error': str(e)}, False
    
    def _augment(self, audio):
        """Apply random augmentation"""
        import random
        
        aug_type = random.choice(['noise', 'pitch', 'speed', 'none'])
        
        if aug_type == 'noise':
            audio = add_noise(audio, noise_factor=random.uniform(0.001, 0.01))
        elif aug_type == 'pitch':
            steps = random.choice([-2, -1, 1, 2])
            audio = pitch_shift(audio, self.target_sr, n_steps=steps)
        elif aug_type == 'speed':
            rate = random.uniform(0.9, 1.1)
            audio = time_stretch(audio, rate=rate)
        
        return audio
    
    def get_stats(self):
        """Get processing statistics"""
        return self.stats

# Usage
config = {
    'sample_rate': 16000,
    'target_duration': 3.0,
    'noise_reduction': True,
    'vad': True,
    'augmentation': False  # True for training
}

pipeline = ProductionAudioPipeline(config)

# Process single file
audio, metadata, success = pipeline.process('input.wav')

if success:
    print(f"✓ Processed successfully")
    print(f"Duration: {metadata['final_duration']:.2f}s")
    # Save
    sf.write('output.wav', audio, pipeline.target_sr)
else:
    print(f"✗ Failed: {metadata.get('error')}")

# Process batch
for audio_file in audio_files:
    audio, metadata, success = pipeline.process(audio_file)
    if success:
        save_processed(audio, metadata)

# Get statistics
stats = pipeline.get_stats()
print(f"Processed: {stats['processed']}")
print(f"Failed: {stats['failed']}")
print(f"Clipped: {stats['clipped']}")
```

---

## Key Takeaways

✅ **Clean audio is critical** - Preprocessing can make/break model performance  
✅ **Standardize formats** - Consistent sample rate, bit depth, mono/stereo  
✅ **Remove noise** - Spectral subtraction, filtering reduce artifacts  
✅ **VAD improves efficiency** - Remove silence saves compute  
✅ **Augmentation boosts robustness** - Time stretch, pitch shift, noise mixing  
✅ **Like feature engineering** - Transform raw data into useful representations  
✅ **Pipeline thinking** - Chain transformations like tree traversal  

---

**Originally published at:** [arunbaby.com/speech-tech/0007-audio-preprocessing](https://www.arunbaby.com/speech-tech/0007-audio-preprocessing/)

*If you found this helpful, consider sharing it with others who might benefit.*

