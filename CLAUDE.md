# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
python main.py
```

No external dependencies — standard library only.

## Architecture

The project is a pipeline-based ABC music notation checker/auto-fixer. All code lives in `main.py`.

**Core pattern:** `CheckerModule` is a base class with a single `process(lines, auto_fix) -> (issues, lines)` method. Each module receives the line list from the previous stage and returns a (possibly modified) line list along with any issues found. `ABCProcessor` owns the ordered list of modules and chains them together in `run_pipeline`.

**Current modules:**

- `HeaderChecker` — verifies required header fields (X, T, M, L, etc.) are present; inserts defaults when `auto_fix=True`. Configured via a `{field: default}` dict at registration time.
- `ClefAutoSelector` — parses note pitches into MIDI values, compares against treble/bass comfort ranges, and either suggests or auto-inserts a better clef at the voice (`V:`) or `K:` level. Also detects sustained register shifts within a voice and inserts inline `[K:clef=xxx]` markers at the measure level.

**Reference:** `abc_notation_standard2.1.md` is the ABC standard v2.1 spec — consult it when implementing new rules about pitch (§4.1), clefs (§4.6), inline fields (§3.1–3.2), or any other notation feature.

**Adding a new checker:** subclass `CheckerModule`, implement `process`, and call `engine.register_module(...)` in the `__main__` block. Order matters — modules run sequentially and each sees the output of the previous.
