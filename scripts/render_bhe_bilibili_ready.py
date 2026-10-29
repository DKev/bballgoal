"""Render Bilibili-ready MP4s with a title, narration, and aligned subtitles.

This is a fallback renderer for macOS JianYing 10.x: the installed
jianying-editor skill currently writes the older Windows draft schema, while
the local macOS app expects the newer Timelines/project.json schema.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = Path(os.getenv("BHE_CAPTURE_DIR", str(ROOT / "capture"))).expanduser()
OUTPUT = CAPTURE / "bilibili-ready"
DRAFTS_ROOT = Path(os.getenv(
    "JY_DRAFTS_ROOT",
    str(Path.home() / "Movies/JianyingPro/User Data/Projects/com.lveditor.draft"),
)).expanduser()


def ass_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ):
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size, index=0)
    raise FileNotFoundError("No macOS Chinese font was found")


def make_caption_png(path: Path, width: int, height: int, text: str, size: int, y: int) -> None:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = load_font(size)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=max(2, size // 14))
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    x = width // 2
    pad_x = max(24, size // 2)
    pad_y = max(12, size // 4)
    draw.rounded_rectangle(
        (x - text_width // 2 - pad_x, y - text_height // 2 - pad_y,
         x + text_width // 2 + pad_x, y + text_height // 2 + pad_y),
        radius=max(10, size // 4),
        fill=(0, 0, 0, 135),
    )
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        anchor="mm",
        stroke_width=max(2, size // 14),
        stroke_fill=(0, 0, 0, 240),
    )
    image.save(path)


def make_overlays(
    output: Path,
    width: int,
    height: int,
    title: str,
    lines: list[tuple[float, float, str]],
) -> list[tuple[Path, float, float]]:
    overlays: list[tuple[Path, float, float]] = []
    title_png = output.with_name(output.stem + "_title.png")
    make_caption_png(title_png, width, height, title, max(36, round(width / 30)), max(90, round(height / 10)))
    overlays.append((title_png, 0.3, 4.8))
    flow_png = output.with_name(output.stem + "_flow.png")
    make_caption_png(flow_png, width, height, "选择视频  →  分析  →  审核  →  导出", max(24, round(width / 48)), max(170, round(height / 5)))
    overlays.append((flow_png, 0.8, 4.8))
    for index, (start, end, text) in enumerate(lines, start=1):
        subtitle_png = output.with_name(output.stem + f"_subtitle_{index:02d}.png")
        make_caption_png(subtitle_png, width, height, text, max(28, round(width / 42)), height - max(90, round(height / 12)))
        overlays.append((subtitle_png, start, end))
    return overlays


def render(
    *,
    source: Path,
    output: Path,
    width: int,
    height: int,
    title: str,
    audio_dir: Path,
    audio_files: list[str],
    audio_starts: list[float],
    audio_durations: list[float],
    subtitle_text: list[str],
) -> None:
    duration = float(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(source)],
            text=True,
        ).strip()
    )
    lines = [
        (start, start + dur, text)
        for start, dur, text in zip(audio_starts, audio_durations, subtitle_text)
    ]
    overlays = make_overlays(output, width, height, title, lines)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    for overlay_path, _, _ in overlays:
        cmd += ["-loop", "1", "-i", str(overlay_path)]
    for filename in audio_files:
        cmd += ["-i", str(audio_dir / filename)]

    filters = []
    video_label = "[0:v]"
    for index, (_, start, end) in enumerate(overlays, start=1):
        next_label = f"v{index}"
        filters.append(
            f"{video_label}[{index}:v]overlay=0:0:eof_action=pass:enable='between(t,{start},{end})'[{next_label}]"
        )
        video_label = f"[{next_label}]"
    filters.append(f"{video_label}format=yuv420p[v]")
    audio_labels = []
    audio_input_offset = len(overlays) + 1
    for index, start in enumerate(audio_starts, start=audio_input_offset):
        label = f"vo{index}"
        delay_ms = round(start * 1000)
        filters.append(f"[{index}:a]adelay={delay_ms}:all=1,apad=whole_dur={duration}[{label}]")
        audio_labels.append(f"[{label}]")
    filters.append(
        f"{''.join(audio_labels)}amix=inputs={len(audio_files)}:duration=longest:dropout_transition=0:normalize=0[vo]"
    )
    filters.append("[0:a]volume=0.28[orig]")
    filters.append("[orig][vo]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    print(f"RENDERED {output}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    render(
        source=CAPTURE / "pc录制.mp4",
        output=OUTPUT / "BHE_9月8日_PC介绍版.mp4",
        width=1436,
        height=896,
        title="BHE｜电脑端剪辑流程",
        audio_dir=DRAFTS_ROOT / "BHE_9月8日_PC介绍版/temp_assets",
        audio_files=[
            "tts_4687e46f.ogg", "tts_1ca659ee.ogg", "tts_a332431e.ogg", "tts_52078d2e.ogg",
            "tts_6bbdb536.ogg", "tts_436d5e62.ogg", "tts_53b6927e.ogg",
        ],
        audio_starts=[0.8, 5.1, 7.8, 10.1, 13.6, 14.5, 16.2],
        audio_durations=[4.2, 2.6, 2.2, 3.4, 0.8, 1.6, 1.6],
        subtitle_text=[
            "给兄弟们分享一下我自己开发的开源篮球剪辑工具 BHE",
            "之前已经有 Mac 和 Windows 版本",
            "现在 Android 版也在开发中",
            "电脑端的流程很简单：选择一段比赛视频",
            "开始分析",
            "审核识别出来的片段",
            "最后导出篮球集锦",
        ],
    )

    render(
        source=CAPTURE / "app 录制.mp4",
        output=OUTPUT / "BHE_9月9日_Android介绍版.mp4",
        width=1176,
        height=2560,
        title="BHE｜Android 版操作演示",
        audio_dir=DRAFTS_ROOT / "BHE_9月9日_Android介绍版/temp_assets",
        audio_files=[
            "tts_8be38404.ogg", "tts_1e079451.ogg", "tts_a2de49c4.ogg", "tts_7ae9781e.ogg",
            "tts_3d78f7cc.ogg", "tts_a0728abc.ogg", "tts_6ca02147.ogg",
        ],
        audio_starts=[0.8, 2.7, 5.38, 6.46, 7.36, 8.66, 9.96],
        audio_durations=[1.8, 2.58, 0.98, 0.8, 1.2, 1.2, 2.8],
        subtitle_text=[
            "这是 BHE 的 Android 版",
            "手机上也可以完成同样的流程",
            "选择视频",
            "开始分析",
            "审核高光片段",
            "最后导出集锦",
            "省去自己一段一段找视频的时间",
        ],
    )


if __name__ == "__main__":
    main()
