"""Fetch a pinned demo asset, or verify a copy transferred to an offline server.

Uses only the Python standard library. Video metadata and SHA256 are pinned in
data/yuanzhuopai-source-manifest.json; no YouTube access or downloader is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/yuanzhuopai-source-manifest.json'


def verify(path, asset):
    path = Path(path)
    if path.stat().st_size != asset['bytes']:
        raise ValueError(f'{path}: size mismatch; expected {asset["bytes"]} bytes')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != asset['sha256']:
        raise ValueError(f'{path}: SHA256 mismatch')
    return path


def fetch(asset, directory, *, verify_only=False):
    filename = asset['filename']
    if Path(filename).name != filename or filename in ('', '.', '..'):
        raise ValueError('manifest filename must be a basename')
    directory = Path(directory)
    target = directory / filename
    # Existing files are never silently replaced, including corrupted copies.
    if target.exists() or verify_only:
        return verify(target, asset)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(asset['download_url'], headers={'User-Agent': 'grove-character-demo/1'})
        with urlopen(request, timeout=60) as source:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=filename+'.', suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                received = 0
                progress = 0
                while block := source.read(1024 * 1024):
                    received += len(block)
                    if received > asset['bytes']:
                        raise ValueError('download exceeds manifest size')
                    output.write(block)
                    if received - progress >= 32 * 1024**2:
                        print(f'{filename}: {received / 1024**2:.0f}/{asset["bytes"] / 1024**2:.0f} MiB', file=sys.stderr)
                        progress = received
        verify(temporary, asset)
        # A same-directory hard link publishes atomically without overwriting a
        # file another process may have created during this download.
        try:
            target.hardlink_to(temporary)
        except FileExistsError:
            verify(target, asset)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('short', 'full'), default='short')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/media')
    parser.add_argument('--verify-only', action='store_true', help='offline validation; never access the network')
    args = parser.parse_args()
    try:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        path = fetch(manifest['assets'][args.variant], args.output_dir, verify_only=args.verify_only)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    print(f'Verified: {path.resolve()}')


if __name__ == '__main__':
    main()
