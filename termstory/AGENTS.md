# termstory/ — Source Package

This directory contains the core implementation of **TermStory**, including the command-line interface, data storage, parsing pipeline, AI integrations, terminal user interface, and supporting utilities.

---

## Entry Points

* `cli.py` — Main Typer-based command-line interface coordinating application features.
* `__main__.py` — Executes the package as `python -m termstory` by invoking the CLI entry point.
* `__init__.py` — Package initialization and version information.

---

## Core Infrastructure

* `config.py` — Loads, saves, and manages application configuration and data directories.
* `database.py` — SQLite database layer responsible for persistence and connection management.
* `models.py` — Shared data models used across the application.
* `date_utils.py` — Common date and time helper utilities.
* `sanitizer.py` — Utilities for sanitizing terminal history and sensitive information.

---

## History Processing

* `parser.py` — Parses terminal history into application data structures.
* `session.py` — Session grouping and session-related logic.
* `timestamp_detective.py` — Timestamp detection and recovery utilities.
* `archive.py` — Archive management and archive-related database operations.

---

## AI & Analysis

* `ai.py` — AI provider integration and language model communication.
* `ask.py` — AI-powered querying of stored terminal history.
* `rag.py` — Retrieval-augmented generation: hybrid BM25 + cosine-similarity search over terminal sessions. Supports zero-keyword queries via:
  * (a) topic-anchor query expansion (`_expand_query`)
  * (b) command-shorthand preprocessing (`_preprocess_command_text`) so the embedding model receives natural-language signal
  * (c) an enlarged bounded fallback candidate pool when FTS returns no matches.
  * Includes a semantic-only mode (`alpha >= 0.999`) that bypasses FTS entirely.
* `predict.py` — Prediction-related functionality.
* `insights.py` — Insight generation utilities.

---

## User Interface

* `tui.py` — Textual terminal user interface.
* `formatter.py` — Formatting and rendering helpers.
* `timeline.py` — Timeline-related functionality.
* `replay.py` — Replay-related functionality.

---

## Search & Organization

* `search.py` — Search-related functionality (FTS5 + standard `LIKE`-based fallback).
* `project.py` — Project-related utilities.
* `tags.py` — Tag management utilities.
* `stats.py` — Statistics and analytics utilities.
* `notebook.py` — Notebook-related functionality.
* `reminder.py` — Reminder-related functionality.

---

## Integrations

* `git_integration.py` — Git integration utilities.
* `web.py` — Web-related functionality.
* `exporter.py` — Export functionality.
* `backup.py` — Database backup utilities.
* `mcp_snapshot.py` — MCP snapshot utilities.
* `hermes_obs.py` — Hermes observability and integration utilities.

---

## Data Flow

A typical processing flow is:

1. CLI entry points initialize the application.
2. Configuration is loaded.
3. Terminal history is parsed.
4. Sessions are grouped.
5. Timestamps are recovered when necessary.
6. Data is stored in the SQLite database.
7. Search, analytics, AI, and visualization operate on stored data.
8. Results are presented through the CLI, TUI, or export features.

---

## Key Conventions

* The application uses a SQLite database for persistent storage.
* Configuration is managed through `config.py` and stored in the application's configuration directory.
* The Textual interface uses `call_after_refresh(...)` when UI updates must occur after screen refresh.
* Background tasks use Textual's `@work` decorator where appropriate.
* Shared models and utilities are designed to be reused across CLI, TUI, AI, and analysis components.
* The `rag.py` module is the single entry point for semantic search. It does **NOT** invoke any LLM directly — all embeddings are computed locally via `sentence-transformers`. Sanitization of LLM-facing inputs remains the responsibility of the caller (e.g., `ask.py`, `ai.py`).

---

## Notes for Contributors

* Keep module responsibilities focused and cohesive.
* Reuse shared helpers from `config.py`, `database.py`, `date_utils.py`, and `models.py` where appropriate.
* Prefer extending existing modules before introducing new abstractions.
* Follow existing project structure and naming conventions when adding features.
* **When extending `rag.py`:** The query-expansion (`_TOPIC_EXPANSIONS`) and command-preprocessing (`_COMMAND_VERB_MAP`) maps are intentionally small and conservative. Add new entries only when you have a concrete failing test case demonstrating the gap. Avoid aggressive expansion — it can dilute the embedding signal for normal keyword queries.
