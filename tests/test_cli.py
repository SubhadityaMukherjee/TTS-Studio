import shutil
import subprocess

import click
import pytest
from click.testing import CliRunner

from tts_studio.cli import (
    _audio_duration,
    _trim_voice_sample,
    load_chapters,
    main,
    process_chapter,
)


class TestCli:
    def test_help(self):
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "convert" in result.output

    def test_convert_help_lists_engines(self):
        result = CliRunner().invoke(main, ["convert", "--help"])
        assert result.exit_code == 0
        for engine in ("kokoro", "edge", "breeze"):
            assert engine in result.output

    def test_missing_pdf_is_skipped_gracefully(self, tmp_path):
        result = CliRunner().invoke(
            main, ["convert", str(tmp_path / "missing.pdf")], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "File not found" in result.output


class TestLoadChapters:
    def test_text_file(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("Some content here.")
        chapters = load_chapters(str(path))
        assert chapters == [
            {"title": "notes", "content": "Some content here.", "order": 1}
        ]

    def test_missing_path_treated_as_raw_text(self):
        chapters = load_chapters("just some words")
        assert chapters == [
            {"title": "Content", "content": "just some words", "order": 1}
        ]


class TestProcessChapter:
    def test_skips_existing_output(self, tmp_path):
        (tmp_path / "01_My_Chapter.wav").write_bytes(b"")
        result = process_chapter(
            {"title": "My Chapter", "content": "text", "order": 1},
            voice="af_heart",
            speed=1.0,
            lang="a",
            split_output=str(tmp_path),
            abstract_only=False,
            engine="kokoro",
        )
        assert result[0].endswith("01_My_Chapter.wav")
        assert result[1] == "skipped"

    def test_edge_uses_mp3_extension(self, tmp_path):
        (tmp_path / "01_My_Chapter.mp3").write_bytes(b"")
        result = process_chapter(
            {"title": "My Chapter", "content": "text", "order": 1},
            voice="af_heart",
            speed=1.0,
            lang="a",
            split_output=str(tmp_path),
            abstract_only=False,
            engine="edge",
        )
        assert result[0].endswith("01_My_Chapter.mp3")
        assert result[1] == "skipped"

    def test_abstract_only_skips_non_abstract_chapters(self, tmp_path):
        result = process_chapter(
            {"title": "Introduction", "content": "text", "order": 1},
            voice="af_heart",
            speed=1.0,
            lang="a",
            split_output=str(tmp_path),
            abstract_only=True,
            engine="kokoro",
        )
        assert result[1] == "skipped"

    def test_filename_sanitization(self, tmp_path):
        expected = tmp_path / "02_Chapter_1_The_Start.wav"
        expected.write_bytes(b"")
        result = process_chapter(
            {"title": "Chapter 1: The *Start*!", "content": "text", "order": 2},
            voice="af_heart",
            speed=1.0,
            lang="a",
            split_output=str(tmp_path),
            abstract_only=False,
            engine="kokoro",
        )
        assert result[0] == str(expected)
        assert result[1] == "skipped"


class TestAudioDuration:
    def test_wav_via_soundfile(self, tmp_path):
        import numpy as np
        import soundfile as sf

        path = tmp_path / "sample.wav"
        sf.write(path, np.zeros(24000, dtype=np.float32), 24000)
        assert _audio_duration(str(path)) == 1.0

    @pytest.mark.skipif(
        not shutil.which("ffmpeg"), reason="ffmpeg not installed"
    )
    def test_m4a_via_ffprobe_fallback(self, tmp_path):
        # libsndfile cannot read m4a; the ffprobe fallback must handle it.
        path = tmp_path / "sample.m4a"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=24000:cl=mono",
                "-t",
                "1",
                "-c:a",
                "aac",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        assert _audio_duration(str(path)) == pytest.approx(1.0, abs=0.1)

    def test_unreadable_file_raises_click_exception(self, tmp_path):
        path = tmp_path / "garbage.wav"
        path.write_bytes(b"not audio at all")
        # soundfile rejects it, so this ends in ffprobe - which either is
        # missing (ClickException) or also rejects it (ClickException).
        with pytest.raises(click.ClickException):
            _audio_duration(str(path))


def _write_wav(path, seconds, rate=24000):
    import numpy as np
    import soundfile as sf

    sf.write(path, np.zeros(int(seconds * rate), dtype=np.float32), rate)


class TestTrimVoiceSample:
    def test_short_sample_returned_untouched(self, tmp_path):
        path = tmp_path / "short.wav"
        _write_wav(path, 1.0)
        assert _trim_voice_sample(str(path), 15.0) == str(path)

    def test_disabled_when_max_seconds_zero(self, tmp_path):
        path = tmp_path / "long.wav"
        _write_wav(path, 2.0)
        assert _trim_voice_sample(str(path), 0) == str(path)

    def test_long_wav_trimmed_to_max(self, tmp_path):
        path = tmp_path / "long.wav"
        _write_wav(path, 30.0)
        trimmed = _trim_voice_sample(str(path), 15.0)
        assert trimmed != str(path)
        assert _audio_duration(trimmed) == pytest.approx(15.0, abs=0.1)

    @pytest.mark.skipif(
        not shutil.which("ffmpeg"), reason="ffmpeg not installed"
    )
    def test_long_m4a_trimmed_via_ffmpeg(self, tmp_path):
        # m4a is not readable by soundfile; trimming must go through ffmpeg.
        path = tmp_path / "long.m4a"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=24000:cl=mono",
                "-t",
                "30",
                "-c:a",
                "aac",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        trimmed = _trim_voice_sample(str(path), 15.0)
        assert trimmed != str(path)
        assert _audio_duration(trimmed) == pytest.approx(15.0, abs=0.1)
