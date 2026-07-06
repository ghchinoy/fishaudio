import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from fishaudio import cli


@patch("mlx_speech.tts.load")
@patch("mlx_speech.audio.write_wav")
def test_basic_generation(mock_write_wav, mock_load):
    # Setup mock return values
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.waveform = [0.1, 0.2]
    mock_result.sample_rate = 44100
    mock_model.generate.return_value = mock_result
    mock_load.return_value = mock_model

    runner = CliRunner()
    result = runner.invoke(cli, ["--text", "Hello from unit test"])
    assert result.exit_code == 0
    assert "✅ Generation plan prepared:" in result.output
    assert "Text: Hello from unit test" in result.output
    mock_load.assert_called_once()


@patch("mlx_speech.tts.load")
@patch("mlx_speech.audio.write_wav")
def test_config_merging(mock_write_wav, mock_load, tmp_path):
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.waveform = [0.1, 0.2]
    mock_result.sample_rate = 44100
    mock_model.generate.return_value = mock_result
    mock_load.return_value = mock_model

    config_file = tmp_path / "fishaudio_test.yml"
    config_data = {
        "fishaudio": {
            "model_dir": "./models/test_model",
            "output_dir": "./test_outputs",
            "max_new_tokens": 128,
            "trim_leading_silence": True,
            "normalize_peak": 0.8,
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["--config", str(config_file), "--text", "Testing config", "--verbose"]
    )

    assert result.exit_code == 0
    assert "test_model" in result.output
    assert "Max tokens: 128" in result.output
    assert "Trim silence: ✓" in result.output
    assert "Normalize peak: 0.8" in result.output


@patch("mlx_speech.tts.load")
@patch("mlx_speech.audio.write_wav")
def test_config_merging_with_cli_override(mock_write_wav, mock_load, tmp_path):
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.waveform = [0.1, 0.2]
    mock_result.sample_rate = 44100
    mock_model.generate.return_value = mock_result
    mock_load.return_value = mock_model

    config_file = tmp_path / "fishaudio_test.yml"
    config_data = {"fishaudio": {"model_dir": "./models/test_model", "max_new_tokens": 128}}
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(config_file),
            "--text",
            "Testing config override",
            "--max-new-tokens",
            "256",
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "Max tokens: 256" in result.output


@patch("mlx_speech.tts.load")
@patch("mlx_speech.audio.write_wav")
def test_voice_cloning_validation(mock_write_wav, mock_load):
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.waveform = [0.1, 0.2]
    mock_result.sample_rate = 44100
    mock_model.generate.return_value = mock_result
    mock_load.return_value = mock_model

    runner = CliRunner()

    # Missing reference text
    result = runner.invoke(cli, ["--text", "Test cloning", "--reference-audio", "ref.wav"])
    assert result.exit_code != 0
    assert "Both reference-audio and reference-text must be provided together" in result.output

    # Missing reference audio
    result = runner.invoke(
        cli, ["--text", "Test cloning", "--reference-text", "Some reference transcript"]
    )
    assert result.exit_code != 0
    assert "Both reference-audio and reference-text must be provided together" in result.output

    # Both provided
    result = runner.invoke(
        cli,
        [
            "--text",
            "Test cloning",
            "--reference-audio",
            "ref.wav",
            "--reference-text",
            "Some reference transcript",
        ],
    )
    assert result.exit_code == 0
    assert "With voice cloning: ✓" in result.output


@patch("mlx_speech.tts.load")
@patch("mlx_speech.audio.write_wav")
@patch("shutil.which")
@patch("subprocess.run")
def test_autoplay_flag(mock_sub_run, mock_which, mock_write_wav, mock_load):
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.waveform = [0.1, 0.2]
    mock_result.sample_rate = 44100
    mock_model.generate.return_value = mock_result
    mock_load.return_value = mock_model

    mock_which.return_value = "/usr/bin/afplay"

    runner = CliRunner()
    result = runner.invoke(cli, ["--text", "Hello", "--play"])
    assert result.exit_code == 0
    assert "▶️ Autoplay starting:" in result.output
    assert "⏹️ Playback finished." in result.output
    mock_which.assert_called_once_with("afplay")
    mock_sub_run.assert_called_once_with(["afplay", "output.wav"], check=True)


def test_list_voices_no_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--list-voices"])
    assert result.exit_code == 0
    assert "No 'voices/' directory found" in result.output


def test_list_voices_with_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()

    # Create a voice reference
    (voices_dir / "sarah.wav").touch()
    (voices_dir / "sarah.txt").write_text("Hello, this is Sarah.", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["--list-voices"])
    assert result.exit_code == 0
    assert "Available custom voices" in result.output
    assert "sarah" in result.output
    assert "Hello, this is Sarah" in result.output


@patch("mlx_speech.tts.load")
@patch("mlx_speech.audio.write_wav")
def test_voice_option_resolution(mock_write_wav, mock_load, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create voice reference
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "sarah.wav").touch()
    (voices_dir / "sarah.txt").write_text("Hello, this is Sarah.", encoding="utf-8")

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.waveform = [0.1, 0.2]
    mock_result.sample_rate = 44100
    mock_model.generate.return_value = mock_result
    mock_load.return_value = mock_model

    runner = CliRunner()
    result = runner.invoke(cli, ["--voice", "sarah", "--text", "Test speech"])

    assert result.exit_code == 0
    assert "With voice cloning: ✓" in result.output
