"""WhisperBlocks desktop GUI (PySide6).

The GUI is a thin shell: it edits config/config.toml and runs
src/transcript.py as a separate process, reading its progress events.
Modules without a ``Qt`` import (paths, config_store, prompts, events,
i18n) are covered by the offline test suite.
"""
