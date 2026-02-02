# Security model

The runner executes pipeline and actions commands under a conservative policy.

## Hard deny (always blocked)

Some destructive patterns are always blocked (e.g. `rm -rf /`, fork bombs).

## `security.mode`

- `safe` (default): adds a default denylist (e.g. `sudo`, `docker system prune`, `mkfs`, `dd`, `reboot`)
- `system`: looser (still keeps hard deny)

You can also provide:

- `security.denylist`: additional regex patterns to block
- `security.allowlist`: if set, commands must match at least one allowlist pattern

## Unattended mode

- `--unattended strict` (default):
  - sets `CI=1`, `GIT_TERMINAL_PROMPT=0`, etc.
  - blocks commands that look interactive (e.g. `docker login` without non-interactive flags)
- `--unattended guided`:
  - allows `pipeline.auth.interactive: true` commands to run interactively

The runner still records artifacts and exit codes for auditability.

If you run without a `pipeline.yml`, the runner still applies the **safe** default deny patterns.
