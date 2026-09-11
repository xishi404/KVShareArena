"""Check the tracked release for accidental internal files and common secret formats.

Only file names and categories are printed on failure, never matched secret values.
This check supplements review; it is not a guarantee that arbitrary content is safe.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_ROOTS = {'pilot', 'slides', 'analysis', 'patches', '.claude', '_smoke'}
BANNED_NAMES = {'CLAUDE.md', 'AGENTS.md', 'RUN_LOG.md', 'TODO.md', 'RESEARCH.md',
                'PLAN_ARCC.md', 'EXP_SETTING.md', 'MIGRATION.md', '.DS_Store'}
PATTERNS = {
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'GitHub token': re.compile(rb'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
    'API key': re.compile(rb'sk-[A-Za-z0-9_-]{35,}'),
    'personal or cluster path': re.compile(rb'/(?:lustre/fs\d/home|blue/[^/]+|Users)/[^/\s]+/'),
}


def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    problems, count, size = [], 0, 0
    for name in filter(None, files):
        p = ROOT / name
        if p.parts[len(ROOT.parts)] in BANNED_ROOTS or p.name in BANNED_NAMES:
            problems.append((name, 'internal file'))
        if p.is_symlink():
            problems.append((name, 'symlink'))
            continue
        data = p.read_bytes()
        count += 1
        size += len(data)
        if len(data) > 50 * 1024 * 1024:
            problems.append((name, 'file larger than 50 MiB'))
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                problems.append((name, label))
    for name, reason in problems:
        print(f'REVIEW {name}: {reason}')
    if not count:
        raise SystemExit('No tracked files. Stage the release before auditing.')
    if problems:
        raise SystemExit(f'{len(problems)} release audit issue(s)')
    print(f'PASS: {count} tracked files, {size / 1024 / 1024:.1f} MiB; no banned files or matched credentials/paths.')


if __name__ == '__main__':
    main()
