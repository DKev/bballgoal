"""本地审核网页：在浏览器里确认每个人是谁（号码、名字、队），合并 / 拆开卡片。

只在本机运行（默认只有这台电脑能打开），所有画面和结果都留在本地，不上传。

用法（先跑 bballgoal.cards）：
    .venv/bin/python -m bballgoal.review videos/比赛.mp4
    .venv/bin/python -m bballgoal.review videos/比赛.mp4 --lan   # 同一 Wi-Fi 下的 iPad / 手机也能打开
结果随时自动保存到 output/track_<视频名>/review/people.json。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bballgoal.paths import out_dir as out_dir_for

WEB_DIR = Path(__file__).with_name("review_web")


def make_handler(review_dir: Path):
    review_dir = review_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:  # 安静一点
            pass

        def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = HTTPStatus.OK) -> None:
            self._send(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send((WEB_DIR / "people.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                people = review_dir / "people.json"
                self._json({
                    "cards": json.loads((review_dir / "cards.json").read_text()),
                    "people": json.loads(people.read_text()) if people.exists() else None,
                })
            elif path.startswith("/files/"):
                target = (review_dir / path[len("/files/"):]).resolve()
                if review_dir not in target.parents or not target.is_file():
                    self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                    return
                self._send(target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/people":
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            data["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            tmp = review_dir / "people.json.tmp"
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
            tmp.replace(review_dir / "people.json")  # 先写临时文件再替换，中途关掉也不会写坏
            self._json({"ok": True, "saved_at": data["saved_at"]})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lan", action="store_true", help="允许同一局域网的其他设备打开")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    review_dir = (args.out or out_dir_for(args.video)) / "review"
    if not (review_dir / "cards.json").exists():
        print(f"缺少 {review_dir / 'cards.json'}，先运行：.venv/bin/python -m bballgoal.cards {args.video}")
        return 1
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), make_handler(review_dir))
    url = f"http://localhost:{args.port}/"
    print(f"认人网页：{url}")
    if args.lan:
        print(f"同一 Wi-Fi 的其他设备：http://{socket.gethostbyname(socket.gethostname())}:{args.port}/")
    print("结果自动保存到", review_dir / "people.json", "  按 Ctrl+C 结束")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
