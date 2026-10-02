from click.testing import CliRunner

from tts_studio.cli import load_chapters, main, process_chapter


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
