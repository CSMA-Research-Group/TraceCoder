#!/usr/bin/env python3
"""Build a standalone source/data ZIP from an allowlist; never package user state."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracecoder import __version__
from tracecoder.catalog import ROOT

BLOCKED = {".git", ".venv", "node_modules", "__pycache__", "releases", ".runtime", "workspace", "runs", "backups"}
ALLOWED_EXT = {".py", ".sh", ".md", ".txt", ".toml", ".html", ".css", ".js", ".svg", ".json", ".yaml", ".yml", ".example", ".dockerignore"}
CODE_NAMES = {"Dockerfile", "Dockerfile.sandbox", "Makefile", ".gitignore", ".env.example"}


def release_files():
    for path in sorted((ROOT / "code").rglob("*")):
        relative = path.relative_to(ROOT)
        if relative.as_posix() == "code/docs/VERIFICATION.md":
            continue  # Historical v1 acceptance remains in the research checkout.
        if path.is_symlink() or not path.is_file() or any(part in BLOCKED for part in relative.parts):
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        if path.suffix in ALLOWED_EXT or path.name in CODE_NAMES:
            yield relative.as_posix(), path.read_bytes()
    for directory in ("benchmarks", "examples", "templates", "official"):
        for path in sorted((ROOT / "data" / directory).glob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".jsonl"):
                yield path.relative_to(ROOT).as_posix(), path.read_bytes()
    for name in ("README.md", ".gitignore", "benchmark_index.json", "source_manifest.json", "original_materials.sha256.json"):
        path = ROOT / "data" / name
        if path.is_file():
            yield path.relative_to(ROOT).as_posix(), path.read_bytes()
    # Curated acceptance artifacts only; excludes all runtime/project histories.
    verification = ROOT / "data" / "verification" / "integration-v2.1"
    if verification.exists():
        for path in sorted(verification.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".txt", ".jpg"):
                yield path.relative_to(ROOT).as_posix(), path.read_bytes()


def main():
    destination = ROOT / "code" / "releases"
    destination.mkdir(exist_ok=True)
    name = "TraceCoder-%s" % __version__
    archive_path = destination / (name + ".zip")
    files = dict(release_files())
    files["START_HERE.md"] = ("# TraceCoder %s\n\n" % __version__ +
        "在解压目录打开终端，执行：\n\n```bash\nsh code/scripts/start.sh --port 8767\n```\n\n"
        "打开 http://127.0.0.1:8767 。需 macOS / Linux + Python 3.9 或以上；Windows 可用 WSL2 或 Docker。\n\n"
        "首次启动自动安装锁定的 Python 依赖，需要访问包源；安装后固定演示无需联网与模型密钥。代码与测试每次实际执行。\n\n"
        "- 完整说明：[code/README.md](code/README.md)\n"
        "- 部署与接入：[code/docs/DEPLOYMENT.md](code/docs/DEPLOYMENT.md)\n"
        "- 论文对应：[code/docs/PAPER_MAPPING.md](code/docs/PAPER_MAPPING.md)\n"
        "- 验收边界：[code/docs/SYSTEM_ACCEPTANCE.md](code/docs/SYSTEM_ACCEPTANCE.md)\n\n"
        "解压后可运行 `python3 code/scripts/verify_release.py .` 校验全部文件哈希。\n"
        "运行数据由首次启动创建，本包不含用户项目、历史运行或密钥。\n").encode()
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in files.items()}
    manifest = {"version": __version__, "format": 1, "files": hashes, "runtime_dependencies": "Pinned Flask / Waitress and transitive dependencies in code/requirements.txt",
                "excludes": ["credentials", "user_projects", "runtime_history", "original_reference_scripts", "virtualenvs"],
                "paper": "10.1145/3744916.3773187"}
    files["MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    files["SHA256SUMS"] = ("\n".join(hashlib.sha256(value).hexdigest() + "  " + key for key, value in files.items()) + "\n").encode()
    temporary = archive_path.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, content in sorted(files.items()):
            entry = zipfile.ZipInfo(name + "/" + path, (2026, 10, 19, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (0o100755 if path.endswith(".sh") else 0o100644) << 16
            archive.writestr(entry, content)
    temporary.replace(archive_path)
    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (destination / (archive_path.name + ".sha256")).write_text(sha + "  " + archive_path.name + "\n")
    info = {"available": True, "version": __version__, "filename": archive_path.name, "sha256": sha,
            "bytes": archive_path.stat().st_size, "files": len(files), "created_at": datetime.now(timezone.utc).isoformat()}
    (destination / "latest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
