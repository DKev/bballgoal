import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name == "nt", reason="macOS packaging uses Bash")
@pytest.mark.parametrize(
    "mode,expected_status,expected_attempts",
    [("busy-once", 0, 2), ("busy-always", 16, 5), ("other-error", 7, 1), ("verify-error", 37, 1)],
)
def test_dmg_creation_retries_only_busy_and_requires_verification(
    tmp_path, mode, expected_status, expected_attempts,
):
    tools = tmp_path / "bin"
    tools.mkdir()
    hdiutil = tools / "hdiutil"
    hdiutil.write_text('''#!/bin/sh
if [ "$1" = verify ]; then
  [ "$TEST_MODE" != verify-error ] || exit 37
  exit 0
fi
count=0
[ ! -f "$TEST_COUNT" ] || count=$(cat "$TEST_COUNT")
count=$((count + 1))
printf '%s' "$count" > "$TEST_COUNT"
if [ "$TEST_MODE" = other-error ]; then
  echo 'hdiutil: create failed - Permission denied' >&2
  exit 7
fi
if [ "$TEST_MODE" = busy-always ] || { [ "$TEST_MODE" = busy-once ] && [ "$count" = 1 ]; }; then
  echo 'hdiutil: create failed - Resource busy' >&2
  exit 16
fi
for argument in "$@"; do output="$argument"; done
printf 'test-dmg' > "$output"
''', encoding="utf-8")
    hdiutil.chmod(0o755)
    sleep = tools / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sleep.chmod(0o755)
    source = tmp_path / "source"
    source.mkdir()
    count_file = tmp_path / "count"
    result = subprocess.run(
        ["bash", str(Path(__file__).resolve().parents[1] / "scripts/create_macos_dmg.sh"),
         str(source), str(tmp_path / "output.dmg")],
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
             "TEST_MODE": mode, "TEST_COUNT": str(count_file)},
        capture_output=True, text=True,
    )
    assert result.returncode == expected_status, result.stderr
    assert int(count_file.read_text()) == expected_attempts
