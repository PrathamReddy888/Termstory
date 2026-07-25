## 12. TUI Dashboard

Launch with `termstory ui`.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔥 12 day streak  •  43 active days  •  127h 14m total                     │
│  Activity (Last 90 Days):  ░░▒▒░▓▓█▓▒░░▒▒▓▓█▓▒░░▒▓▓█▓▒░░▒▒▓▓█▓▒░░▒░░      │
├───────────────────────────────┬─────────────────────────────────────────────┤
│  June 2026                    │  📂 termstory  •  Tue Jun 03                │
│  ├─ Jun 07 (Sat)              │  ─────────────────────────────────────────  │
│  │  └─ termstory              │  [💻 Dev Log]                               │
│  │     ✨ v0.2.9 Timestamp… │  ├─ 🔨 Built: Timestamp Detective with      │
│  ├─ Jun 06 (Fri)              │         5 forensic detectors & interpolation│
│  │  ├─ termstory              │  ├─ 🔧 Flow: pytest 155 passed, twine      │
│  │  └─ Other                  │         upload, git push origin main        │
│  │     📦 Legacy Archive      │  └─ 🚀 Result: v0.2.9 published to PyPI    │
│  └─ Jun 05 (Thu)              │                                             │
│     └─ 🔍 Recovered Archive   │  💻 Command Timeline:                       │
│                               │  • 00:47:57  git commit -m "feat: v0.2.9…" │
│                               │      [🔍 git log: termstory@bbe9dfc]        │
│                               │  • 00:48:12  python3 -m pytest tests/       │
│                               │  • 00:49:03  twine upload dist/*            │
└───────────────────────────────┴─────────────────────────────────────────────┘
  ?:help  /:search  o:ai-config  c:copy  q:quit
```

### TUI Features

- **Responsive Resizing:** The layout gracefully reflows if you resize your terminal window.
- **Copy Feedback:** Visual "Copied" flash animations when using the `c` clipboard shortcut.

### Layout

- **StatsHeader** (top, full-width): streak count, active days, total duration, and a GitHub-style command-volume heatmap synced to the timeline's `--days` limit.
- **HistoryTree** (left 30%): collapsible Year → Month → Day → Project → Session tree. The root node shows an All-Time Wrapped dashboard. Month nodes show Monthly Wrapped.
- **DetailsCanvas** (right 70%): renders whatever the selected tree node points to — date chronicle, session details, wrapped views, or search results.

### Session Tree Labels

```
✨ npm init, express installed  (14:00 - 14:45)   ← real timestamp
🔍 Recovered Archive (38 cmds • 14:00 - 14:45)    ← Detective recovered
📦 Legacy Archive (12 recovered cmds)              ← fully synthetic
```

# Keyboard Shortcuts

| Key / Shortcut | Action |
| :--- | :--- |
| `?` | Toggle help overlay |
| `/` | Open search box (real-time filter of 90-day window) |
| `Enter` (in search) | Escape to all-time deep search |
| `Esc` | Close modal / clear search / restore timeline |
| `o` | Open AI provider configuration screen |
| `c` | Copy selected canvas content to OS clipboard |
| `d` | Launch the legacy grid-based MatrixDefragScreen cyberpunk overlay |
| `m` | Launch the Matrix Defrag data-ingestion animation (issue #41) |
| `j` / `k` / `↑` / `↓` | Navigate tree |
| `Ctrl+↓` / `Ctrl+↑` | Scroll canvas |
| `q` | Quit |

### Search Engine

Typing in `/` search filters the pre-loaded 90-day timeline in real-time (no DB hit). Pressing `Enter` escapes to an all-time database search across commands, commit messages, project names, and AI summaries. The tree root label changes to `🔍 Search Results: "query"`. Pressing `Esc` or clearing the box instantly restores the original timeline.

### Clipboard

`c` pipes text directly to `pbcopy` (macOS), `xclip`/`xsel`/`wl-copy` (Linux), or `clip` (Windows), bypassing OSC 52 terminal restrictions. ANSI escape codes are stripped before copy.

### Onboarding

On first launch with no config, an interactive onboarding screen offers `Ctrl+G` (Groq), `Ctrl+A` (OpenAI), `Ctrl+L` (Ollama), `Ctrl+C` (Custom), or `Ctrl+D` (disable AI). If your history lacks timestamps, a consent prompt offers to append `setopt EXTENDED_HISTORY` to `~/.zshrc`.

### Wrapped Views

Selecting the root **Timeline** node or a **Month** node shows a high-density wrapped dashboard:

- **Code churn matrix:** git insertions vs. deletions, net LOC growth
- **Time distribution:** hourly activity punch-card across the period
- **Project breakdown:** percentage distribution across all projects
- **Top commits & tooling:** most-used commands, editors, languages
- **AI Behavioral Audit:** a witty, witty roast of your patterns and productivity archetypes (*"The Midnight Alchemist"*, *"The Expansionist Architect"*) — regeneratable with `r`

---

# Feature Specification: Matrix Defrag Data-Ingestion Animation (Issue #41)

When **TermStory** boots for the first time (empty DB) or ingests a massive batch of new shell history (any single history file > 1 MB), the standard progress bar is replaced with a cascading Matrix-style data stream that takes over the entire `DetailsCanvas`. 

---

## 🚀 Triggers

* **Automatic (First Boot):** Triggered when starting TermStory with an empty database.
* **Automatic (Large File):** Triggered during ingestion of any single history file larger than 1 MB.
* **Manual (CLI):** `termstory ui --matrix`
* **Manual (TUI Shortcut):** Pressing `m` inside the TUI.

---

## 🎨 Visual Contract

* **Canvas Swap:** The `DetailsCanvas` is cleared and replaced by a single `MatrixDefragCanvas` widget — no extra UI panels are added.
* **Data Stream:** Raw shell commands interlaced with `0x........` hex codes scroll down the canvas in dim green/cyan at **~16 FPS**.
* **Status Banner:** A top banner tracks the Parser Engine's current lifecycle stage:
  
  $$\text{INITIALIZING} \rightarrow \text{SCANNING} \rightarrow \text{PARSING} \rightarrow \text{CORRELATING} \rightarrow \text{DETECTING PROJECTS} \rightarrow \text{LOCKING} \rightarrow \text{DEFRAG COMPLETE}$$

* **Real-time Counters:**
  * `[INGESTED: N]` — Commands parsed by the engine.
  * `[LOCKED: M]` — Commands flushed to SQLite.
* **Commit "Snap" Effect:**
  * As `db.save_data()` commits commands to the SQLite DB, corresponding visible lines "snap" to **bright-white readable text** on a **faint green background**.
  * Duration: **~6 animation ticks** ($\approx 360\text{ ms}$) before fading back to dim text and scrolling away.
* **Completion Flow:**
  * Canvas automatically restores to its normal view (timeline, session details, or wrapped view).
  * Fires notification: `💚 Matrix Defrag complete`

---

## ⚡ Threading Model

The full ingestion pipeline runs strictly in a background worker:

```text
parse_all_histories ➔ create_sessions ➔ detect_projects ➔ db.save_data ➔ git commits ➔ auto-tag ➔ MCP snapshot ➔ sleep daemon

## 13. AI Narrative Design

TermStory's prompts are designed to produce **high-density, CLI-styled developer logs** — not marketing prose.

### Session Summary Format

```
[💻 Dev Log]
├─ 🔨 Built: <short punchy action — what was built or coded>
├─ 🔧 Flow: <tools used, tests run, configs edited>
└─ 🚀 Result: <milestone shipped, fixed, or pushed>
```

or:

```
[🤖 Codebase Pulse]
• Hacked: <what was designed, refactored, or debugged>
• Tooling: <commands run, docker setups, libraries configured>
• Outcome: <what was verified, resolved, or shipped>
```

**Rules enforced in the prompt:**
- No paragraphs. No filler. No "ultimately the hard work paid off".
- Every line starts with a past-tense engineering verb: *wired up, refactored, debugged, spun up, stabilized, shipped*.
- Second-person narrative (`"You"`) for the Daily Chronicle view.

### Daily Chronicle (Date node)

When a date is selected in the TUI, a full "Story of You" daily narrative is generated — including an inferred breaks timeline, hourly activity punch-card, and second-person storytelling of the whole day.

---
