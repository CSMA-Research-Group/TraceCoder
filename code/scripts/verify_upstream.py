#!/usr/bin/env python3
"""Read-only preservation/source audit in the full research checkout (not required to run ZIP)."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = ROOT / 'news/audit/branch-inventory.json'
    if not audit.exists():
        raise SystemExit('This check needs the preserved research checkout news/; use verify_release.py for the standalone package.')
    inventory = json.loads(audit.read_text())
    errors, branches = [], []
    for branch in inventory['branches']:
        directory = ROOT / 'news' / branch['worktree']
        commit = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain'], text=True).strip()
        unchanged = all(sha(directory / name) == expected for name, expected in branch['tracked_file_hashes'].items())
        if not unchanged or commit != branch['commit'] or status:
            errors.append(branch['name'])
        branches.append({'branch': branch['name'], 'commit': commit, 'files_checked': len(branch['tracked_file_hashes']), 'unchanged': unchanged, 'clean': not status})
    originals = json.loads((ROOT / 'data/original_materials.sha256.json').read_text())
    for name, expected in originals.items():
        if sha(ROOT / name) != expected:
            errors.append(name)
    old = (ROOT / 'news/main/problem_processor.py').read_text()
    new = (ROOT / 'code/tracecoder/official_prompts.py').read_text()
    functions = lambda text: {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    a, b = functions(old), functions(new)
    prompts = {name: a[name] == b[name] for name in ('_build_instrumentation_prompt', '_build_analysis_planning_prompt', '_build_code_implementation_prompt')}
    errors.extend(name for name, same in prompts.items() if not same)
    html = (ROOT / 'news/TraceCoder-WebDemo/web_demo/templates/examples.html').read_text()
    raw = html.split('const allExamples = ', 1)[1].split('\n];', 1)[0] + '\n]'
    raw = re.sub(r'^(\s*)(id|name|category|difficulty|description|test_case|hints):', r'\1"\2":', raw, flags=re.M)
    examples = ast.literal_eval(raw)
    examples_same = examples == json.loads((ROOT / 'data/official/examples.json').read_text())
    if not examples_same:
        errors.append('official examples')
    backup = ROOT / 'data/backups/pre-official-integration-20261018T202359Z'
    info = json.loads((backup / 'backup.json').read_text())
    archive = backup / 'code-data-before-integration.zip'
    backup_ok = sha(archive) == info['sha256']
    with zipfile.ZipFile(archive) as z:
        zip_ok = z.testzip() is None
    if not backup_ok or not zip_ok:
        errors.append('preservation backup')
    output = {'checked_at': datetime.now(timezone.utc).isoformat(), 'passed': not errors, 'official_branches': branches,
              'original_files_checked': len(originals), 'official_prompt_bodies_equal': prompts,
              'official_examples_equal': examples_same, 'official_examples': len(examples),
              'backup_sha256': info['sha256'], 'backup_verified': backup_ok and zip_ok, 'errors': errors}
    destination = ROOT / 'data/verification/integration-v2.1/upstream-preservation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
