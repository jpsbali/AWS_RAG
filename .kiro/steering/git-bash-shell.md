---
inclusion: always
---

# Shell Environment

When executing shell commands on this project, always use Git Bash (bash) as the shell environment, not PowerShell or CMD.

- Use Unix-style commands (e.g., `ls`, `rm`, `cp`, `mkdir -p`)
- Use forward slashes for paths (e.g., `src/main.py`)
- Use `&&` to chain commands
- Use `source` instead of `.` for activating virtual environments
- Assume standard Unix utilities are available via Git Bash
