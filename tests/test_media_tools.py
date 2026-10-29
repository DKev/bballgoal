from engine.python.basketball_engine import media_tools
import create_proxy
import pytest


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
def test_windows_media_tools_use_bundled_exe_without_system_path(tmp_path, monkeypatch, name):
    executable = tmp_path / "bin" / f"{name}.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"binary")
    executable.chmod(0o755)
    monkeypatch.setenv("BHE_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(media_tools.sys, "platform", "win32")
    monkeypatch.setattr(create_proxy.platform, "system", lambda: "Windows")
    assert media_tools.resolve_media_tool(name) == str(executable.resolve())
    assert create_proxy.resolve_media_tool(name) == str(executable.resolve())
