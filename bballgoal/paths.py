"""每个视频的结果放在哪。

比赛文件夹（output/<比赛>/，里面有 game.json）里的视频：结果放在同一文件夹下以视频名命名的子文件夹，
例如 output/msk_game2/full.mp4 → output/msk_game2/full/。
其他视频（单独跑某个模块时）：output/track_<视频名>/。
"""

from __future__ import annotations

from pathlib import Path

OUTPUT = Path("output")


def out_dir(video: str | Path) -> Path:
    video = Path(video)
    if (video.parent / "game.json").exists():
        return video.parent / video.stem
    return OUTPUT / f"track_{video.stem}"
