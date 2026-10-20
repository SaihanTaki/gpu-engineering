# CLAUDE.md

## Working style

- Use the `ponytail` skill for all coding tasks: smallest change that fully solves the task, no over-engineering.
- Use the `antislop` skill (plus `antislop-code` for code comments) to keep AI slop out of code.
- Use the `caveman` skill for replies: terse, answer first, keep every technical fact.

## Subagents

- Delegate independent subtasks to subagents and run them in parallel whenever possible (e.g. separate chapters, a search while you edit elsewhere).
- Keep work that depends on a previous step's result in the main thread.
- Give each parallel subagent its own git worktree so they never touch the same files:
  - Create with `git worktree add ../<repo>-<task> -b <task-branch>`, or use `isolation: worktree` on the subagent if supported.
  - Each subagent works and commits only inside its own worktree and branch.
  - Merge or cherry-pick the branches back into the main branch from the main thread, resolve conflicts there, then remove the worktree with `git worktree remove ../<repo>-<task>`.
- Skip worktrees for read-only subtasks (search, review) since nothing is written.
- When the same kind of subtask repeats (e.g. writing a chapter, reviewing content against the simulator), create a custom subagent for it in `.claude/agents/<name>.md` with a focused prompt and only the tools it needs. Reuse existing agents before adding new ones.

## Git commits

- Follow the `git-workflow-and-versioning` skill for branching, atomic commits, PRs, and releases.
- Write commit messages with the `caveman-commit` skill (Conventional Commits, intent only).
- Do NOT add a `Co-Authored-By` trailer or any other attribution line to commit messages.