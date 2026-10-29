# bballgoal

基于 [basketball-highlight-editor](https://github.com/fly7632785/basketball-highlight-editor)（MIT）改的全场比赛数据统计版。
目标：全场画面里两队每个人按球衣号码标注，统计出手 / 命中 / 篮板 / 助攻。

新增代码都在 `bballgoal/` 目录，原项目代码保持不动。

## 日常使用：本地网站

```bash
.venv/bin/python -m bballgoal.app          # 打开 http://localhost:8000
.venv/bin/python -m bballgoal.app --lan    # 同一 Wi-Fi 的 iPad / 手机也能打开
```

只在本机运行，视频和结果都不上传。每场比赛一个文件夹 `output/<比赛>/`：
`raw/`（原始 360 文件）、`full.mp4`（整场分析画面）、`enroll.mp4`（认人用的 8 段）、
`full/`、`enroll/`（各自的结果）、`balls/`（点球和训练的模型）、`clips/`（事件片段）、`reels/`（集锦）、`logs/`。
后台任务是独立进程，关掉网页、重启网站都不影响；网页顶部随时显示后台在做什么、还要多久。

| 步骤 | 系统做什么 | 你做什么 |
|---|---|---|
| ① 导入视频 | 找到相机/电脑上的录像，原始文件放进比赛文件夹；自动选对着球场的镜头 | 点场地边界和两个篮筐 |
| ② 认人 · 名单 | 从整场均匀抽 8 段，跟踪、聚类成卡片（带黄框截图、卡通头像）；同时后台转换、跟踪整场、给远处的人补外观 | 一张一张过：点掉不是他的截图，A/B/R/C/X 选身份、直接打号码、回车下一张。**同队同号自动合并**，裁判/教练/无关的人各一堆；✂ 拆开串人的段 |
| ② 认人 · 整场核对 | 用名单（特征库）认整场每一段，再按时间接力补上跟踪断开的碎片 | **按人批量核对**：一屏是猜成同一个人的截图，点掉错的，回车整组确认；剩下的逐个问（打号码，没有就新建）。答完一批重认一轮（几秒），越问越少；零碎的可以“剩下的不问了” |
| ③ 点球 | 准备 240 张画面（三分之一在篮筐附近）；新比赛自动沿用上一场训练的模型 | 第一场：点球的位置，满 100 张训练这个机位专用的找球模型。之后：可以跳过；想更准就再点 20 张以上，和以前所有比赛的标注一起训练 |
| ④ 事件审核 | 找投篮（进没进、谁投的；认不出人的按篮筐推断哪队），虚拟镜头渲染片段 | 回车确认、改、删、补；“方式”标三分 / 罚篮；填真实比分 |
| ⑤ 集锦 | 投篮集锦、触球集锦（每次拿球）、球队五佳球；游戏式定格介绍、结尾、合成的背景音乐、中文 / English | 选孩子点生成，或者一次给所有人 / 两队生成 |

### 下一场比赛

新建比赛、导入视频，然后按上面的顺序走。会自动带过来的：
- **找球模型**：导入时复制最近一场训练好的（`balls/train.json` 里记着 `inherited_from`），整场跟踪完直接逐帧找球。
- **名单**：第一次打开“认人 · 名单”时，按**球衣颜色**和最近一场对上队（A/B 每场按颜色重新分，你们队不一定还是 A），
  带过来队名和“号码 → 名字”；之后确认身份时填了号码，名字也自动带出。对方换了队就不带。

磁盘：一场约 40 GB（原片 35 GB + 片段和结果）。处理完可以删 `full.mp4`（约 20 GB）：看片段、剪集锦都直接读原片；
要重新找事件时会自动从原片再转一份。跟踪的标注视频默认不生成（`--debug-video` 才出）。

拍摄：相机尽量放在同一个位置（画面范围、找球模型都最匹配），镜头里拍到记分牌。

给人看的视频（事件片段、集锦）都用**虚拟镜头**从 360 原片渲染：普通镜头投影（线是直的）、
镜头提前规划平滑跟随、去噪锐化调色（`bballgoal/vcam.py`）。分析用的 `full.mp4` 是等距柱状展开（各方向分辨率一致）。

## 环境

```bash
brew install ffmpeg python@3.11
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt -r requirements-bballgoal.txt
```

模型首次运行时自动下载到 `models/`：人物检测 `yolo11m.pt`、姿态 `yolo11m-pose.pt`、行人重识别 `osnet_ain_x1_0_msmt17.pt`（BoxMOT）、ResNet50（torchvision）。

## 识别效果探测

把比赛视频（Insta360 Studio 导出的固定全场平面视频，建议 4K 以上）放进 `videos/`：

```bash
.venv/bin/python -m bballgoal.probe videos/比赛.mp4
.venv/bin/python -m bballgoal.probe videos/比赛.mp4 --frames 12 --start 60 --end 240
```

输出在 `output/probe_<视频名>/`：每帧一张带框截图，`contact_sheet.jpg` 是总览，`summary.json` 是检测明细。
橙色是篮球（外加大圈），红色是篮筐，绿色是人。

## 场地设置（每个机位一次）

```bash
.venv/bin/python -m bballgoal.court videos/比赛.mp4
```

弹出窗口里沿比赛区域边缘点一圈（回车结束），再依次点左、右篮筐中心。结果存为 `videos/比赛.court.json`。

## 球员跟踪 + 自动分队

```bash
.venv/bin/python -m bballgoal.track videos/比赛.mp4              # 整段
.venv/bin/python -m bballgoal.track videos/比赛.mp4 --end 60     # 只跑前 60 秒
```

输出在 `output/track_<视频名>/`：`tracks.mp4` 标注视频（两队按球衣颜色画框，裁判白框），`tracks.json` 每帧每人的框 / 轨迹 / 队伍。
M4 上约 3.5 帧/秒（每秒取 10 帧处理），即处理 1 分钟比赛约需 3 分钟。

- 只处理脚底点在比赛区域内的人：场外替补、教练、隔网后面别的场地的人都不管；
- 跟踪用 BoT-SORT（BoxMOT）：位置预测 + OSNet 行人重识别外观一起匹配，交错时不容易串号；
- 分队靠躯干颜色（HSV）聚成两队，裁判靠黑白竖条纹 + 大人身高识别；
- 每 3 帧给每人采一次外观样本存进 `appearance.npz`：OSNet 整体外观，加上姿态模型定位脚踝后取的鞋部特征。

## 球衣号码识别

```bash
.venv/bin/python -m bballgoal.identify videos/比赛.mp4
```

固定每个人（不只看球衣）：
1. 每场比赛单独学一个“区分人”的投影：同一轨迹里的样本是同一人，不同轨迹是不同人，不需要人工标注；
2. 同队轨迹按外观（整体 + 鞋）逐步合并成人：同时出现的不合并，号码确定且不同的不合并，号码相同的优先合并；
3. 每个人用他所有轨迹的号码读数一起投票（macOS Vision 本地识别，不上传）；认不出号码的显示 `A?1`、`A?2`。

实测（同队不同人能否分开，AUC）：只看颜色 0.68 → OSNet 0.84 → 每场学投影后 OSNet 0.96 → 再加鞋 0.97。
头部在 4K 全场画面里只有十几个像素，单独加进来反而变差，交给整体外观覆盖。

输出 `players.json`（每个人的轨迹和出场时间）和 `players.mp4`（框上显示 A#73 / A?2 这样的固定编号）。

OCR 读不出的号码可以人工指定（可选，不是必需）：在视频旁边放 `videos/比赛.labels.json`，
`[{"time": 78.9, "x": 3721, "y": 1545, "player": "A#18"}]`（某一刻画面上这个人身上的一点）。
用时间和位置而不是轨迹号，重新跟踪后依然有效。

## 找球 + 持球人

```bash
.venv/bin/python -m bballgoal.ball videos/比赛.mp4
```

按原视频帧率逐帧找球（`models/bball_model.pt`），去掉总在同一位置的、头上的（光头像球）、
离场上的人太远的（隔壁场地、网后），再用动态规划挑一条连贯的球轨迹，判断每帧球在谁手里。
输出 `ball.json`。目前远处的球只有约 15% 的帧找得到，是持球统计的主要瓶颈。
（basketball_analysis 的 `ball_detector_model.pt` 用 NBA 转播训练，在全场远景上实测几乎找不到球。）

## 个人视频

```bash
.venv/bin/python -m bballgoal.reel videos/比赛.mp4 A#18
```

把这个人在场上的每一段剪出来，镜头跟着他从 4K 原画裁切到 1080p；持球时标黄。输出 `reel_A18.mp4`。

## 360 原始文件（.insv）

```bash
.venv/bin/python -m bballgoal.fisheye videos/raw/VID_..._016.insv videos/raw/VID_..._017.insv -o videos/比赛.mp4
```

不用 Insta360 Studio 导出：自动选对着球场的镜头，鱼眼展开（等距柱状投影，原生分辨率，不放大），
自动定画面范围，多段文件接成一个视频（保留原始帧率和声音）。实测和 Studio 导出的 4K 相比，
像素数相同、细节略好（原始码率 52 Mbps/镜头 vs 导出 25 Mbps），帧率是原生的。

## 认人网页（第一遍之后）

```bash
.venv/bin/python -m bballgoal.cards videos/比赛.mp4    # 生成每个候选人的卡片和截图
.venv/bin/python -m bballgoal.review videos/比赛.mp4   # 打开本地网页（--lan 让同一 Wi-Fi 的 iPad 也能打开）
```

只在本机运行，不上传。每张卡片是一个候选人：填队、号码、名字，确认；混进来的别人点 × 或拖到正确的卡上；
同一个人的两张卡拖 ⠿ 合并；可撤销。结果自动保存到 `review/people.json`。
