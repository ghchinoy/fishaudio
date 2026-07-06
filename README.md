# Fish Audio S2 Pro CLI - MLX Text-to-Speech Generator

Generate high-quality speech synthesis using the Fish Audio S2 Pro MLX model optimized for Apple Silicon.

## Features

- **MLX-Powered**: Native Apple Silicon text-to-speech using the Fish Audio S2 Pro model
- **Voice Cloning**: Create personalized voices using reference audio and text
- **Emotion Tags**: Support for 15,000+ inline prosody and emotion tags
- **Voice Management**: Easy voice organization in a local `voices/` directory with `--voice` and `--list-voices` flags
- **Autoplay Output**: Automatic playback support using macOS `afplay` with `--play` flag
- **YAML Configuration**: Load persistent settings from `fishaudio.yml`
- **Flexible Overrides**: Command-line flags override config values
- **Professional Output**: High-quality WAV audio generation
- **Cross-Platform**: Works on Apple Silicon with MLX framework

## Prerequisites

### 1. Download the MLX Model

First, download the [Fish Audio S2 Pro MLX model from HuggingFace](https://huggingface.co/mlx-community/fishaudio-s2-pro-8bit-mlx):

```bash
hf download --local-dir ./models/fishaudio-s2-pro-8bit-mlx mlx-community/fishaudio-s2-pro-8bit-mlx
```

**Note:** The model is ~6.72 GB and contains the int8 quantized weights with a bundled codec for waveform decoding.

### 2. Install Dependencies

Install the required packages using **uv** (recommended):

```bash
uv pip install click pyyaml
```

*(Alternatively, if using standard python virtual environments: `pip install click pyyaml`)*

### 3. Configure Output Directory

Create an output directory for generated audio files:

```bash
mkdir -p ./outputs
```

## Installation

### Using uv (Recommended)

**Simple installation:**

```bash
# Install the CLI tool globally
uv pip install fishaudio-cli

# Run the CLI tool directly
fishaudio
```

**Installation from source:**

```bash
# Clone the repository
# Assuming you're already in the repository root

# Install the package in editable mode
uv pip install -e .

# Run the CLI tool
fishaudio
```

### Using python -m fishaudio

```bash
# When installed in editable mode, you can run the CLI via:
python -m fishaudio
```

### Manual Installation

```bash
# Clone the repository
# Assuming you're already in the repository root

# Make the script executable (Unix-like systems)
chmod +x fishaudio.py

# Install Python dependencies
uv pip install click pyyaml  # Or: pip install click pyyaml

# Run the CLI using the script directly
python fishaudio.py
```

## Usage & Examples

### 1. Basic Generation
Generate a standard speech synthesis file:
```bash
uv run fishaudio --text "Hello from Fish S2 Pro!"
```

### 2. Autoplay After Generation
Synthesize and automatically play the output audio immediately using macOS `afplay`:
```bash
uv run fishaudio --text "Awesome! Let's listen to this." --play
```

### 3. Custom Voice Management (Zero-Shot Cloning)
To make voice cloning easy, you can manage a directory of voices. Create a `voices/` directory:
```bash
mkdir -p voices
```
For each custom voice (e.g. `sarah`), add two files:
1. `voices/sarah.wav` - A 5-10 second clear reference audio sample.
2. `voices/sarah.txt` - The exact text transcript of the `sarah.wav` reference sample.

Then, list all available custom voices:
```bash
uv run fishaudio --list-voices
```

Use a custom voice by passing its name:
```bash
uv run fishaudio --text "Hi, this is Sarah speaking." --voice sarah --play
```

### 4. Zero-Shot Cloning (Ad-hoc)
Clone any arbitrary audio file directly from the command line:
```bash
uv run fishaudio --text "I am cloning this voice." --reference-audio /path/to/ref.wav --reference-text "Reference transcript."
```

## Testing

To install development dependencies and run the tests, run:

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```
