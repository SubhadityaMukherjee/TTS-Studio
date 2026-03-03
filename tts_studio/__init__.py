"""
TTS-Studio: A high-quality text-to-speech CLI.

A modular text-to-speech system using the Kokoro API that supports:
- Converting text, EPUB, and PDF files to audio
- Multiple voice options (af_heart, etc.)
- Adjustable speech speed
- Multi-language support (English variants)
- Real-time audio streaming
- Parallel processing with progress tracking
- Chapter-based output for EPUB/PDF files

Modules:
    cli: Command-line interface with convert command
    processor: Core TTS processing with streaming support
    parsers: Document parsers for EPUB and PDF formats
    utils: Utility functions
"""

__version__ = "1.0.0"
