# Tool usage rules

Cross-tool workflow rules. Follow them on every task.

## Find before you change
- Locate code with glob (names), grep (contents) or find_symbols
  (def/callers/deps/imports/symbols) BEFORE reading whole files.
- summarize_file any file over ~300 lines before editing it.

## Editing workflow — the standard loop
1. RISKY OR WIDE CHANGE? -> checkpoint take FIRST (rollback is the undo).
2. MULTIPLE EDITS? -> same file: batch several edit calls in one message;
   several files or big restructures: ONE apply_patch (atomic all-or-nothing,
   position-tolerant, undo reverts everything).
3. ALWAYS verify after editing; fix every ❌ and re-verify until green.
4. THEN run the project's tests (from open/: python -m pytest tests/ -q).
5. Drop the checkpoint once green. NEVER commit unless explicitly asked.

## Calling tools efficiently
- Batch independent calls (reads/searches/checks) in ONE message; chain only
  when a later call needs an earlier result.
- quick_calc instead of bash for math/json/base64/hash/regex/time.
- read/glob/grep instead of bash cat/ls/find.

## Long-running work
- Over ~90s -> background_task, never bash (120s hard cap kills its group).
  Poll read/status; wait caps at 600s; stop() what you no longer need.
- device wake_lock before multi-minute jobs; wake_unlock after.

## Per-tool quick rules (not covered by the base prompt)
- write: only genuinely NEW files; atomic; creates parent dirs.
- edit: exact oldString incl. indentation (never the `N:` prefix); fails if
  missing/ambiguous -> add context or replaceAll; prefer over write.
- apply_patch: `--- /dev/null` creates, `+++ /dev/null` deletes; paths are
  relative to open/; undo pops the last patch.
- verify: auto-tracks everything edited; check clears list only when fully
  green; reset forgets tracking.
- checkpoint: skips binaries/>512KB; rollback auto-saves a pre-rollback copy;
  drop old snapshots to avoid clutter.
- glob/grep: glob maxes at 100 results, `{a,b}` braces work; grep filters by
  include pattern.
- webfetch: one URL, markdown default; webfetch_many: up to 50 URLs parallel;
  LAN hosts blocked.
- task: sub-agent needs a SELF-CONTAINED prompt; use for long independent work.
- question: blocks until answered; ask when ambiguous, then go autonomously.
- remember: durable facts auto-load next sessions; delete stale notes.
- history_search: check how past sessions fixed things BEFORE redoing work.
- todo: multi-step tasks get a list; exactly ONE item in_progress, update live.
- device/screen_view: inspect the running TUI; vibrate/battery need termux-api.

## Cleanup discipline
- Remove every artifact created while testing: scratch files, checkpoints,
  memory notes, background tasks. Never touch pre-existing user changes.

## Communication
- Ambiguous or tradeoffs -> question first; once told go, finish completely.
