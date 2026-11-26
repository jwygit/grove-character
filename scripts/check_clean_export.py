"""Recreate the distributable project, apply its pinned patch, test, compare files.

Includes untracked nonignored files so a prospective commit can be checked before
publishing. Never copies the existing vendor checkout, venv, runs, or model weights.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    logs = report_path.with_suffix('.logs')
    logs.mkdir(exist_ok=True)
    report = dict(status='running', steps=[], model_accuracy_verified=False)
    def run(command, cwd, name):
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
        (logs/(name+'.log')).write_text(result.stdout+result.stderr)
        report['steps'].append(dict(name=name, exit_code=result.returncode))
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {logs/(name+".log")}')
        return result.stdout
    try:
        paths = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
        changed = subprocess.check_output(['git','diff','--name-only'],cwd=root/'vendor/Grove').decode().splitlines()
        with tempfile.TemporaryDirectory(prefix='grove-clean-') as temp:
            export = Path(temp)
            for name in set(paths)-{''}:
                source = root/name
                if source.is_file():
                    target = export/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
            run([sys.executable,'scripts/setup_grove.py'],export,'setup')
            for name in changed:
                if (root/'vendor/Grove'/name).read_bytes() != (export/'vendor/Grove'/name).read_bytes():
                    raise AssertionError(f'patch does not reproduce {name}')
            report['patched_files_equal'] = len(changed)
            files = [p for directory in ('lab','scripts','tests') for p in (export/directory).rglob('*.py')]
            files += [export/'vendor/Grove'/p for p in changed if p.endswith('.py')]
            for path in files:
                ast.parse(path.read_text(), filename=str(path))
            report['python_ast_files'] = len(files)
            for section in ('modules','grove'):
                run([sys.executable,'-m','unittest','discover','-s','tests/'+section,'-t','.','-q'],export,'tests-'+section)
            for name in ('smoke','smoke_visual','smoke_characters','smoke_fact_clips','smoke_full_loop'):
                run([sys.executable,'scripts/'+name+'.py','runs/'+name],export,name)
            preflight = run([sys.executable,'scripts/preflight.py'],export,'preflight')
            report['preflight'] = json.loads(preflight)
            run([sys.executable,'-m','lab.full_loop','--help'],export,'cli-help')
        run(['git','diff','--check'],root,'diff-root')
        run(['git','diff','--check'],root/'vendor/Grove','diff-vendor')
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
