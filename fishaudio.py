#!/usr/bin/env python3
"""
Fish Audio S2 Pro CLI Tool - MLX-based Text-to-Speech Generation with Config Support

This tool generates high-quality speech synthesis using the Fish Audio S2 Pro MLX model
optimized for Apple Silicon. It supports voice cloning, inline emotion/prosody tags,
and can load configuration from YAML files.

The tool reads a configuration file (default: fishaudio.yml) and uses command-line
flags to override specific settings. This allows for easy environment-specific
configuration while maintaining flexibility through CLI overrides.
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml


# Define types for YAML parsing and validation
def dir_path(value):
    """Convert string to Path object with expansion and validation."""
    if value:
        return Path(value).expanduser().resolve()
    return None


def list_available_voices():
    """List all custom voices found in the voices/ directory."""
    voices_dir = Path("voices")
    if not voices_dir.exists() or not voices_dir.is_dir():
        click.echo("ℹ️ No 'voices/' directory found in the current folder.")
        click.echo("💡 To create custom voices, make a 'voices/' folder and add:")
        click.echo("   - <voice_name>.wav (the reference audio clip)")
        click.echo("   - <voice_name>.txt (the text transcript of the audio clip)")
        return

    wav_files = list(voices_dir.glob("*.wav"))
    if not wav_files:
        click.echo("ℹ️ No voice files (*.wav) found in the voices/ directory.")
        return

    click.echo("🎙️ Available custom voices in voices/:")
    for wav_file in sorted(wav_files):
        voice_name = wav_file.stem
        txt_file = wav_file.with_suffix(".txt")
        transcript = ""
        if txt_file.exists():
            try:
                transcript = txt_file.read_text(encoding="utf-8").strip()
                transcript = f' - "{transcript}"'
            except Exception:
                pass
        click.echo(f"  • {voice_name}{transcript}")


class DefaultGroup(click.Group):
    """Custom Click Group to automatically run 'generate' if no subcommand is provided."""

    def parse_args(self, ctx, args):
        if not args:
            return super().parse_args(ctx, args)

        # Check if the first argument is a known command or help option
        cmd_name = args[0]
        if cmd_name in self.commands or cmd_name in ctx.help_option_names:
            return super().parse_args(ctx, args)

        # Otherwise, insert default command name
        args.insert(0, "generate")
        return super().parse_args(ctx, args)


def parse_config_from_yaml(ctx: click.Context, param: str, value: str) -> dict:
    """Load and validate YAML configuration file."""
    if not value:
        default_config = Path("fishaudio.yml")
        if default_config.exists():
            config_path = default_config
        else:
            return {}
    else:
        config_path = Path(value)
        if not config_path.exists():
            raise click.BadParameter(f"Config file not found: {value}")

    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # The config should have a 'fishaudio' top-level key
        if not config_data or "fishaudio" not in config_data:
            found_keys = list(config_data.keys()) if config_data else []
            raise click.BadParameter(
                f"Config file must have a 'fishaudio' top-level key. Found keys: {found_keys}"
            )

        return config_data["fishaudio"]

    except yaml.YAMLError as e:
        raise click.BadParameter(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise click.BadParameter(f"Error reading config file: {e}")


@click.command(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "allow_extra_args": False,
        "allow_interspersed_args": False,
        "show_default": True,
    }
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    callback=parse_config_from_yaml,
    help="Load configuration from YAML file (flags override config values). "
    "Expected file has a 'fishaudio' section. "
    "Default: fishaudio.yml in current directory if file exists.",
)
@click.option(
    "--text",
    required=False,
    help="Text to synthesize. "
    "Supports Fish S2 Pro inline tags like [whisper], [chuckle], [laugh], "
    "[clearing throat], [excited], [sad], [pause].",
)
@click.option(
    "--output-dir",
    type=dir_path,
    default=None,
    help="Output directory for generated audio files. Overrides config value.",
)
@click.option(
    "--output-name",
    default="output.wav",
    show_default=True,
    help="Output filename (including extension). Overrides config value.",
)
@click.option(
    "--model-dir",
    type=dir_path,
    default=None,
    help="Local MLX model directory. Overrides config value.",
)
@click.option(
    "--reference-audio",
    type=dir_path,
    default=None,
    help="Reference audio WAV path for voice cloning. Must be used with --reference-text.",
)
@click.option(
    "--reference-text",
    default=None,
    help="Transcript of the reference audio for voice cloning. "
    "Must be used with --reference-audio. Overrides config value.",
)
@click.option(
    "--max-new-tokens",
    type=click.IntRange(min=1, max=4096),
    default=1024,
    show_default=True,
    help="Maximum tokens to generate. Cli overrides always.",
)
@click.option(
    "--trim-leading-silence/--no-trim-silence",
    default=False,
    show_default=True,
    help="Trim leading low-energy audio before writing the WAV. Cli overrides always.",
)
@click.option(
    "--normalize-peak",
    type=click.FloatRange(min=-1.0, max=1.0, clamp=True),
    default=-1.0,
    show_default=True,
    help="Target peak amplitude for output normalization. Set <= 0 to disable. Cli overrides always.",
)
@click.option(
    "--verbose/--quiet", default=False, is_flag=True, help="Enable or disable verbose output."
)
@click.option(
    "--play",
    is_flag=True,
    default=False,
    help="Autoplay the generated WAV file using macOS afplay if available.",
)
@click.option(
    "--voice",
    default=None,
    help="Use a pre-configured voice from the voices/ directory (e.g. '--voice alex' looks for voices/alex.wav and voices/alex.txt). Overrides config value.",
)
@click.option(
    "--list-voices",
    is_flag=True,
    help="List all custom voices found in the voices/ directory and exit.",
)
@click.pass_context
def generate(
    ctx: click.Context,
    config: dict,
    text: str | None,
    output_dir: Path | None,
    output_name: str,
    model_dir: Path | None,
    reference_audio: Path | None,
    reference_text: str | None,
    max_new_tokens: int,
    trim_leading_silence: bool,
    normalize_peak: float,
    verbose: bool,
    play: bool,
    voice: str | None,
    list_voices: bool,
):
    """
    Generate speech using Fish Audio S2 Pro MLX model with YAML configuration support.

    This tool provides a complete CLI interface for the Fish Audio S2 Pro MLX model,
    supporting voice cloning, emotion tags, and flexible configuration management.

    The configuration file (default: fishaudio.yml) can store persistent settings
    like model directory, output paths, and voice cloning references. Command-line
    flags take precedence over configuration file values when both are provided.

    Configuration file format:
    ```yaml
    fishaudio:
      model_dir: ./models/fish_s2_pro/original
      output_dir: ./outputs
      reference_audio: ./references/voice_sample.wav
      reference_text: "Hello, how are you today?"
      max_new_tokens: 512
      trim_leading_silence: true
      normalize_peak: 0.95
      output_name: my_speech
    ```

    Examples:

    Basic generation with default config:
    ```
    fishaudio --text "Hello from Fish S2 Pro."
    ```

    Voice cloning from config:
    ```
    fishaudio --text "This is a cloned voice."
              --reference-audio ./ref.wav
              --reference-text "This is a cloned voice."
    ```

    Override specific settings:
    ```
    fishaudio --text "Hello!" --model-dir ./my-models --output-dir ./my-outputs
    fishaudio --text "Hello!" --max-new-tokens 2048
    ```

    Use custom config file:
    ```
    fishaudio --text "Hello!" --config myconfig.yml
    ```
    """
    if list_voices:
        list_available_voices()
        return

    if not text:
        raise click.BadParameter("Error: Missing option '--text'.")

    # Load config with CLI flags taking precedence
    config_overrides = {
        "output_dir": output_dir,
        "output_name": output_name,
        "model_dir": model_dir,
        "reference_audio": reference_audio,
        "reference_text": reference_text,
        "max_new_tokens": max_new_tokens,
        "trim_leading_silence": trim_leading_silence,
        "normalize_peak": normalize_peak,
        "play": play,
        "voice": voice,
    }

    # Identify which config overrides were explicitly passed by the user
    # versus which ones are just using Click defaults.
    explicit_overrides = {}
    for key, val in config_overrides.items():
        if val is not None:
            source = ctx.get_parameter_source(key)
            if source != click.core.ParameterSource.DEFAULT:
                explicit_overrides[key] = val
            elif key not in config:
                explicit_overrides[key] = val

    # Merge config: default values from config file, then apply explicit overrides
    final_config = {
        **config,  # Base config from file
        **explicit_overrides,  # Explicit CLI overrides
    }

    # Normalize path fields to Path objects with expanduser and resolve
    for path_key in ["model_dir", "output_dir", "reference_audio"]:
        if final_config.get(path_key):
            final_config[path_key] = Path(final_config[path_key]).expanduser().resolve()

    # If model_dir is not specified, default to standard path
    if not final_config.get("model_dir"):
        final_config["model_dir"] = (
            Path("./models/fishaudio-s2-pro-8bit-mlx").expanduser().resolve()
        )

    # Resolve --voice flag if provided (from YAML or CLI)
    if final_config.get("voice"):
        voice_name = final_config["voice"]
        voices_dir = Path("voices")
        voice_audio = voices_dir / f"{voice_name}.wav"
        voice_text_file = voices_dir / f"{voice_name}.txt"

        if not voice_audio.exists():
            raise click.BadParameter(
                f"Error: Voice file '{voice_audio}' not found. "
                f"Ensure you have a wav file at that path."
            )

        if not voice_text_file.exists():
            raise click.BadParameter(
                f"Error: Voice transcript file '{voice_text_file}' not found. "
                f"Zero-shot cloning requires a transcript of the reference audio."
            )

        try:
            transcript = voice_text_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            raise click.BadParameter(
                f"Error reading transcript file '{voice_text_file}': {e}"
            )

        final_config["reference_audio"] = voice_audio.expanduser().resolve()
        final_config["reference_text"] = transcript

    # Validate references
    if (final_config.get("reference_audio") and not final_config.get("reference_text")) or (
        final_config.get("reference_text") and not final_config.get("reference_audio")
    ):
        raise click.BadParameter(
            "Error: Both reference-audio and reference-text must be provided together for voice cloning"
        )

    # Create output path from output_dir and output_name
    output_dir_val = final_config.get("output_dir")
    output_name_val = final_config.get("output_name", "output.wav")

    if output_dir_val:
        output_path = Path(output_dir_val) / Path(output_name_val).name
    else:
        output_path = Path(output_name_val)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo("🐟 Fish Audio S2 Pro - Configuration Summary:")
        click.echo(f"  Text: {text}")
        click.echo(f"  Output: {output_path}")

        # Check if any values came from CLI vs config
        model_source = "CLI flag" if model_dir else "config file"
        click.echo(f"  Model: {final_config['model_dir']} (from {model_source})")

        if final_config.get("reference_audio"):
            ref_source = "CLI flag" if reference_audio else "config file"
            click.echo("  Voice Cloning: YES")
            click.echo(f"    Reference: {final_config['reference_audio']} (from {ref_source})")
            click.echo(f"    Transcript: {final_config['reference_text']}")

        click.echo("  Generation Settings:")
        click.echo(f"    Max tokens: {final_config['max_new_tokens']}")
        click.echo(f"    Trim silence: {'✓' if final_config['trim_leading_silence'] else '✗'}")
        click.echo(f"    Normalize peak: {final_config['normalize_peak']}")

        click.echo("\n🎯 To actually generate audio:")
        click.echo("   1. Install the MLX model:")
        click.echo("      pip install huggingface_hub[hf_xet]")
        click.echo(
            "      hf download --local-dir fishaudio-s2-pro-8bit-mlx mlx-community/fishaudio-s2-pro-8bit-mlx"
        )
        click.echo(f"   2. Place the downloaded model at: {final_config['model_dir']}")
        click.echo("   3. Run the generation script:")
        click.echo("      python scripts/generate/fish_s2_pro.py \\")
        click.echo(f'        --text "{text}" \\')
        click.echo(f"        --model-dir {final_config['model_dir']} \\")
        click.echo(f"        --output {output_path} \\")
        if final_config.get("reference_audio"):
            click.echo(f"        --reference-audio {final_config['reference_audio']} \\")
            click.echo(f'        --reference-text "{final_config["reference_text"]}" \\')
        click.echo(f"        --max-new-tokens {final_config['max_new_tokens']}")

    click.echo("✅ Generation plan prepared:")
    click.echo(f"   Text: {text}")
    click.echo(f"   Output: {output_path}")
    click.echo(
        f"   With emotion tags: {'✓' if any(tag in text for tag in ['[whisper]', '[chuckle]', '[laugh]', '[clearing throat]', '[excited]', '[sad]', '[pause]']) else '✗'}"
    )
    click.echo(f"   With voice cloning: {'✓' if final_config.get('reference_audio') else '✗'}")

    # Show config used
    if verbose and config:
        click.echo(f"\n📄 Used configuration file: {ctx.params.get('config')}")

    # --- Actual MLX-Speech Generation ---
    click.echo("\n🐟 Fish Audio MLX-Speech Generation active...")
    try:
        import mlx.core as mx
        import numpy as np
        from mlx_speech.audio import write_wav
        from mlx_speech.tts import load as load_tts

        click.echo(f"🔄 Loading model from: {final_config['model_dir']} ...")
        # Load model using the local directory
        model = load_tts(str(final_config["model_dir"]))

        click.echo("🔮 Generating wave codes and synthesis...")
        generate_kwargs = {}
        if final_config.get("reference_audio"):
            generate_kwargs["reference_audio"] = str(final_config["reference_audio"])
            generate_kwargs["reference_text"] = final_config["reference_text"]
        if final_config.get("max_new_tokens"):
            generate_kwargs["max_new_tokens"] = final_config["max_new_tokens"]

        result = model.generate(text, **generate_kwargs)

        audio = np.array(result.waveform)

        # Trim leading silence if enabled
        if final_config.get("trim_leading_silence"):
            chunk_size = 512
            threshold = 0.01  # Energy threshold
            start_idx = 0
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                if len(chunk) > 0 and np.sqrt(np.mean(chunk**2)) > threshold:
                    start_idx = i
                    break
            audio = audio[start_idx:]

        # Normalize peak if enabled (> 0)
        normalize_peak_val = final_config.get("normalize_peak", -1.0)
        if normalize_peak_val > 0.0:
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio * (normalize_peak_val / peak)

        click.echo(f"💾 Saving generated audio to: {output_path} ...")
        write_wav(output_path, mx.array(audio), sample_rate=result.sample_rate)
        click.echo(f"🎉 Successfully generated speech! File written to: {output_path}")

        # Autoplay if --play option is selected and afplay is available
        if final_config.get("play"):
            import shutil
            import subprocess

            if shutil.which("afplay"):
                click.echo(f"▶️ Autoplay starting: {output_path} ...")
                subprocess.run(["afplay", str(output_path)], check=True)
                click.echo("⏹️ Playback finished.")
            else:
                click.echo(
                    "⚠️ Warning: --play was specified but 'afplay' utility was not found on this system."
                )

    except ImportError as e:
        click.echo(f"❌ Error: Could not import mlx_speech or its components. ({e})")
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error during generation: {e}")
        raise click.Abort()


@click.group(
    cls=DefaultGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli():
    """
    Fish Audio S2 Pro CLI - MLX-based Text-to-Speech for Apple Silicon

    Generate high-quality speech synthesis with voice cloning and emotion support.
    This tool leverages the Fish Audio S2 Pro MLX model optimized for Apple Silicon
    and supports flexible configuration through YAML files.

    The tool reads configuration from fishaudio.yml (by default) or a custom file
    specified with --config. Command-line flags override configuration file values,
    allowing for environment-specific customization while maintaining persistent
    settings across deployments.
    """
    pass


cli.add_command(generate)


def main():
    """Entry point for the CLI tool when installed."""
    cli()


if __name__ == "__main__":
    main()
