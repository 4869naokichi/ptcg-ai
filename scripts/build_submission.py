from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PACKAGE = ROOT / "src" / "ptcg_ai"
SUBMISSION_DIR = ROOT / "submission"
GENERATED_PACKAGE = SUBMISSION_DIR / "ptcg_ai"


def copy_package() -> None:
    if GENERATED_PACKAGE.exists():
        shutil.rmtree(GENERATED_PACKAGE)
    shutil.copytree(
        SRC_PACKAGE,
        GENERATED_PACKAGE,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def build_archive(output: Path) -> None:
    members = [
        SUBMISSION_DIR / "main.py",
        SUBMISSION_DIR / "deck.csv",
        SUBMISSION_DIR / "cg",
        GENERATED_PACKAGE,
    ]
    with tarfile.open(output, "w:gz") as tar:
        for member in members:
            tar.add(
                member,
                arcname=member.relative_to(SUBMISSION_DIR),
                filter=_archive_filter,
            )


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if "__pycache__" in parts:
        return None
    if info.name.endswith((".pyc", ".pyo")):
        return None
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "submission.tar.gz",
        help="Path to write the Kaggle submission archive.",
    )
    args = parser.parse_args()

    copy_package()
    build_archive(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
