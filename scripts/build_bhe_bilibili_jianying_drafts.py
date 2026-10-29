"""Create JianYing drafts for the recent BHE PC and Android demos.

The script intentionally creates drafts only. It never publishes to a platform.
"""

import os
import sys
from pathlib import Path


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_ROOT = os.getenv("JY_SKILL_ROOT", str(Path.home() / ".agents/skills/jianying-editor"))
SKILL_SCRIPTS = os.path.join(SKILL_ROOT, "scripts")
if SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)
SKILL_VENDOR = os.path.join(SKILL_SCRIPTS, "vendor")
if SKILL_VENDOR not in sys.path:
    sys.path.insert(0, SKILL_VENDOR)

import pyJianYingDraft as draft

from jy_wrapper import JyProject


PROJECT_ROOT = os.getenv(
    "JY_DRAFTS_ROOT",
    str(Path.home() / "Movies/JianyingPro/User Data/Projects/com.lveditor.draft"),
)
CAPTURE_DIR = os.getenv("BHE_CAPTURE_DIR", str(Path(__file__).resolve().parents[1] / "capture"))


def build_project(
    name: str,
    video_name: str,
    width: int,
    height: int,
    title: str,
    narration: str,
) -> None:
    video_path = os.path.join(CAPTURE_DIR, video_name)
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    project = JyProject(
        name,
        width=width,
        height=height,
        drafts_root=PROJECT_ROOT,
        overwrite=True,
    )

    video = project.add_media_safe(video_path, start_time="0s", track_name="VideoTrack")
    if video is None:
        raise RuntimeError(f"Failed to import video: {video_path}")

    # Keep the screen-recording sound as a quiet bed under the narration.
    video.volume = 0.28

    project.add_text_simple(
        title,
        start_time="0.3s",
        duration="4.5s",
        track_name="TitleTrack",
        clip_settings=draft.ClipSettings(transform_y=0.62),
        style=draft.TextStyle(size=8.5),
        border=draft.TextBorder(color=(0.0, 0.0, 0.0), alpha=1.0, width=45.0),
        anim_in="复古打字机",
    )

    project.add_narrated_subtitles(
        narration,
        speaker="zh_male_huoli",
        start_time="0.8s",
        track_name="Subtitles",
    )

    result = project.save()
    print(f"DRAFT {name}: {result['draft_path']}")


def main() -> None:
    build_project(
        name="BHE_9月8日_PC介绍版",
        video_name="pc录制.mp4",
        width=1436,
        height=896,
        title="BHE｜电脑端剪辑流程",
        narration=(
            "给兄弟们分享一下我自己开发的开源篮球剪辑工具 BHE。"
            "之前已经有 Mac 和 Windows 版本，现在 Android 版也在开发中。"
            "电脑端的流程很简单：选择一段比赛视频，开始分析，审核识别出来的片段，最后导出篮球集锦。"
        ),
    )
    build_project(
        name="BHE_9月9日_Android介绍版",
        video_name="app 录制.mp4",
        width=1176,
        height=2560,
        title="BHE｜Android 版操作演示",
        narration=(
            "这是 BHE 的 Android 版，手机上也可以完成同样的流程。"
            "选择视频，开始分析，审核高光片段，最后导出集锦，省去自己一段一段找视频的时间。"
        ),
    )


if __name__ == "__main__":
    main()
