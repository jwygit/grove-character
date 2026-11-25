"""Recreate the pinned experimental Grove checkout. Does not overwrite an existing directory."""
import json
import hashlib
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lock = json.loads((root/'sources.lock.json').read_text())['grove']
target = root/'vendor'/'Grove'
if target.exists():
    raise SystemExit('vendor/Grove already exists; inspect it rather than overwrite local changes')
target.parent.mkdir(exist_ok=True)
bundle = root/'upstream'/'Grove.bundle'
if not bundle.is_file() or hashlib.sha256(bundle.read_bytes()).hexdigest() != lock['bundle_sha256']:
    raise SystemExit('Missing or modified upstream/Grove.bundle; restore it from this repository')
patch = root/'patches/grove-character.patch'
if hashlib.sha256(patch.read_bytes()).hexdigest() != lock['patch_sha256']:
    raise SystemExit('Grove patch checksum mismatch')
subprocess.run(['git', 'clone', '--branch', 'main', str(bundle), str(target)], check=True)
subprocess.run(['git', 'remote', 'set-url', 'origin', lock['url']], cwd=target, check=True)
subprocess.run(['git', 'checkout', '-b', 'experiment/character-attribution', lock['commit']], cwd=target, check=True)
subprocess.run(['git', 'apply', '--check', str(root/'patches/grove-character.patch')], cwd=target, check=True)
subprocess.run(['git', 'apply', str(root/'patches/grove-character.patch')], cwd=target, check=True)
