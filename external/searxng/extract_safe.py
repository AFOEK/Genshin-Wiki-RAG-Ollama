from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


INVALID_CHARS = set('<>:"|?*')
RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def invalid_component(component: str) -> bool:
    if any(char in INVALID_CHARS for char in component):
        return True

    if component.endswith((" ", ".")):
        return True

    base = component.split(".", 1)[0].upper()
    return base in RESERVED_NAMES


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: extract_safe.py ARCHIVE DESTINATION")
        return 2

    archive_path = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()

    if not archive_path.is_file():
        print(f"Archive not found: {archive_path}")
        return 1

    destination.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped: list[str] = []

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()

        if not members:
            print("Archive is empty.")
            return 1

        archive_root = members[0].filename.split("/", 1)[0]
        prefix = archive_root + "/"

        for member in members:
            name = member.filename.replace("\\", "/")

            if not name.startswith(prefix):
                continue

            relative_name = name[len(prefix):]

            if not relative_name:
                continue

            parts = PurePosixPath(relative_name).parts

            if any(invalid_component(part) for part in parts):
                skipped.append(relative_name)
                continue

            target = destination.joinpath(*parts)
            resolved_target = target.resolve()

            if (
                resolved_target != destination
                and destination not in resolved_target.parents
            ):
                skipped.append(relative_name)
                continue

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(member) as source:
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)

            extracted += 1

    print(f"Extracted files: {extracted}")
    print(f"Skipped paths: {len(skipped)}")

    for path in skipped:
        print(f"SKIPPED: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
