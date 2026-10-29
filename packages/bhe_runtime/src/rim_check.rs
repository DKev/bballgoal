//! Frame-by-frame rim crossing check, ported from the desktop
//! `basketball_highlight.rim_check` module. Results must match the desktop
//! verdicts for the same detections.
//!
//! A make must come down from above the rim opening, cross the rim plane
//! inside the rim, reach the net bottom close to the rim center, not bounce
//! up near the rim, and be hidden by the net for at least one frame. A video
//! whose net does not hide the ball needs the net to move instead.
//!
//! A candidate is excluded only on what the detector saw; other failed checks
//! are "unsure" and stay for review. Coordinates are normalized to the rim:
//! x = (x - center_x) / width and y = (y - rim_y) / width.

use serde::Deserialize;
use serde_json::{json, Map, Value};

pub const RIM_CHECK_VERSION: &str = "rim-dense-v2";
pub const MIN_RIM_WIDTH_PX: f64 = 30.0;
pub const CROP_RIM_WIDTHS: f64 = 5.0;
pub const MIN_CROP_SIDE_PX: f64 = 96.0;

const MIN_BALL_CONF: f64 = 0.15;
const STRONG_BALL_CONF: f64 = 0.5;
// The rim opening spans +-0.5; a ball rolling on the rim crosses up to 0.7.
const MAX_CROSS_OFFSET: f64 = 0.7;
const MAX_EXIT_OFFSET: f64 = 0.7;
const MIN_HIDDEN_FRAMES: usize = 1;
const MIN_OCCLUDED_SHARE: f64 = 0.5;
const MIN_OCCLUSION_SAMPLES: usize = 8;
// A visible ball proves a miss only where the net hides nearly every make.
const STRONG_OCCLUDED_SHARE: f64 = 0.8;
// Only a net that moved less than before the crossing proves a miss.
const MIN_NET_MOTION: f64 = 0.0;
// An off-center track must be a falling ball, not a jersey next to the rim.
const MIN_OFF_CENTER_TRACK_POINTS: usize = 10;
const MIN_OFF_CENTER_TRACK_SPAN: f64 = 1.0;

const ZONE_X: f64 = 2.5;
const ZONE_TOP: f64 = -2.5;
const ZONE_BOTTOM: f64 = 3.0;

#[derive(Clone, Debug, Deserialize)]
pub struct Rim {
    pub center_x: f64,
    pub rim_y: f64,
    pub width: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Detection {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub confidence: f64,
    pub xyxy: [f64; 4],
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
    conf: f64,
    i: usize,
}

#[derive(Clone, Debug)]
struct Crossing {
    x_cross: f64,
    entry: bool,
    x_exit: Option<f64>,
    reached_net_bottom: bool,
    rebound: bool,
    hidden_frames: usize,
    track_points: usize,
    track_span: f64,
    cross_frame: usize,
}

/// Python's `round(value, digits)` for the values this module produces.
fn round_to(value: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(digits);
    let scaled = value * scale;
    let rounded = scaled.round();
    // Python rounds exact halves to even.
    if (scaled - scaled.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        (rounded - scaled.signum()) / scale
    } else {
        rounded / scale
    }
}

/// Square crop centered just below the rim so approach and net exit fit.
pub fn tight_crop_box(rim: &Rim, frame_width: f64, frame_height: f64) -> [i64; 4] {
    let side = MIN_CROP_SIDE_PX
        .max(CROP_RIM_WIDTHS * rim.width)
        .min(frame_width)
        .min(frame_height);
    let cx = rim.center_x;
    let cy = rim.rim_y + 0.75 * rim.width;
    let x1 = round_to((cx - side / 2.0).max(0.0).min(frame_width - side), 0) as i64;
    let y1 = round_to((cy - side / 2.0).max(0.0).min(frame_height - side), 0) as i64;
    [x1, y1, x1 + side as i64, y1 + side as i64]
}

fn frame_balls(detections: &[Detection], rim: &Rim, index: usize) -> Vec<Point> {
    detections
        .iter()
        .filter(|item| item.name == "ball" && item.confidence >= MIN_BALL_CONF)
        .filter_map(|item| {
            let [x1, y1, x2, y2] = item.xyxy;
            let x = ((x1 + x2) / 2.0 - rim.center_x) / rim.width;
            let y = ((y1 + y2) / 2.0 - rim.rim_y) / rim.width;
            (x.abs() <= ZONE_X && (ZONE_TOP..=ZONE_BOTTOM).contains(&y)).then_some(Point {
                x,
                y,
                conf: item.confidence,
                i: index,
            })
        })
        .collect()
}

/// Remove detections that sit still for many frames (bolts, logos).
fn drop_static(frames_balls: Vec<Vec<Point>>) -> Vec<Vec<Point>> {
    const RADIUS: f64 = 0.12;
    const MIN_FRAMES: usize = 6;
    let flat: Vec<Point> = frames_balls.iter().flatten().copied().collect();
    frames_balls
        .into_iter()
        .map(|balls| {
            balls
                .into_iter()
                .filter(|ball| {
                    let mut frames: Vec<usize> = flat
                        .iter()
                        .filter(|other| {
                            (other.x - ball.x).abs() <= RADIUS && (other.y - ball.y).abs() <= RADIUS
                        })
                        .map(|other| other.i)
                        .collect();
                    frames.dedup();
                    frames.len() < MIN_FRAMES
                })
                .collect()
        })
        .collect()
}

fn build_tracks(frames_balls: &[Vec<Point>]) -> Vec<Vec<Point>> {
    const PER_FRAME: f64 = 0.7;
    const MAX_GAP: usize = 12;
    const MAX_JUMP: f64 = 2.0;
    let mut tracks: Vec<Vec<Point>> = Vec::new();
    for balls in frames_balls {
        let mut sorted = balls.clone();
        sorted.sort_by(|a, b| {
            b.conf
                .partial_cmp(&a.conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for point in sorted {
            let mut best: Option<(f64, usize)> = None;
            for (index, track) in tracks.iter().enumerate() {
                let last = track[track.len() - 1];
                if point.i <= last.i || point.i - last.i > MAX_GAP {
                    continue;
                }
                let gap = (point.i - last.i) as f64;
                let distance = ((point.x - last.x).powi(2) + (point.y - last.y).powi(2)).sqrt();
                if distance <= (PER_FRAME * gap).min(MAX_JUMP)
                    && best.is_none_or(|(best_distance, _)| distance < best_distance)
                {
                    best = Some((distance, index));
                }
            }
            match best {
                Some((_, index)) => tracks[index].push(point),
                None => tracks.push(vec![point]),
            }
        }
    }
    tracks
        .into_iter()
        .filter(|track| track.len() >= 3)
        .collect()
}

/// Append a fragment that starts right after the crossing; the net often
/// yields a second box that splits one fall into two tracks.
fn continuation(after: &[Point], below: Point, others: &[&Vec<Point>]) -> Vec<Point> {
    let tail = after.last().copied().unwrap_or(below);
    for other in others {
        let first = other[0];
        if below.i <= first.i + 3
            && first.i <= tail.i + 12
            && (-0.1..=1.0).contains(&first.y)
            && (first.x - tail.x).abs() <= 0.8
            && other[other.len() - 1].i > tail.i
        {
            let mut joined = after.to_vec();
            joined.extend(other.iter().filter(|point| point.i > tail.i));
            return joined;
        }
    }
    after.to_vec()
}

fn track_crossings(track_index: usize, tracks: &[Vec<Point>]) -> Vec<Crossing> {
    let track = &tracks[track_index];
    let others: Vec<&Vec<Point>> = tracks
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != track_index)
        .map(|(_, other)| other)
        .collect();
    let mut crossings = Vec::new();
    for k in 0..track.len().saturating_sub(1) {
        let (above, below) = (track[k], track[k + 1]);
        if !(above.y < 0.0 && 0.0 <= below.y) {
            continue;
        }
        let ratio = -above.y / (below.y - above.y);
        let x_cross = above.x + ratio * (below.x - above.x);
        let before = &track[..=k];
        let entry = before
            .iter()
            .any(|point| (-1.2..=-0.1).contains(&point.y) && point.i + 15 >= above.i);
        let after = continuation(&track[k + 1..], below, &others);
        let exits: Vec<&Point> = after
            .iter()
            .filter(|point| point.y >= 0.4 && point.i - below.i <= 20)
            .collect();
        let deep = after
            .iter()
            .any(|point| point.y >= 1.0 && point.i - below.i <= 30);
        let mut rebound = false;
        let mut lowest = below.y;
        for point in &after {
            if point.y > 1.2 {
                break;
            }
            if point.y < lowest - 0.3 {
                rebound = true;
                break;
            }
            lowest = lowest.max(point.y);
        }
        let last_above = before
            .iter()
            .filter(|point| point.y < -0.1)
            .map(|point| point.i)
            .max()
            .unwrap_or(above.i);
        let first_below = after
            .iter()
            .filter(|point| point.y >= 0.6)
            .map(|point| point.i)
            .min()
            .unwrap_or(below.i);
        let path: Vec<&Point> = before.iter().chain(after.iter()).collect();
        let seen = |index: usize| {
            path.iter()
                .rev()
                .find(|point| point.i == index)
                .map_or(0.0, |point| point.conf)
        };
        let hidden = (last_above + 1..first_below)
            .filter(|index| seen(*index) < STRONG_BALL_CONF)
            .count();
        let max_y = path.iter().map(|point| point.y).fold(f64::MIN, f64::max);
        let min_y = path.iter().map(|point| point.y).fold(f64::MAX, f64::min);
        crossings.push(Crossing {
            x_cross: round_to(x_cross, 3),
            entry,
            x_exit: exits.first().map(|point| round_to(point.x, 3)),
            reached_net_bottom: deep,
            rebound,
            hidden_frames: hidden,
            track_points: k + 1 + after.len(),
            track_span: round_to(max_y - min_y, 3),
            cross_frame: below.i,
        });
    }
    crossings
}

/// Whether a ball fell toward the rim from above, close to its center.
fn ball_approached(tracks: &[Vec<Point>]) -> bool {
    tracks.iter().any(|track| {
        let above: Vec<&Point> = track
            .iter()
            .filter(|point| point.x.abs() <= 1.5 && (-2.5..=-0.1).contains(&point.y))
            .collect();
        above.len() >= 3 && above[above.len() - 1].y - above[0].y >= 0.3
    })
}

fn rejection_reason(crossing: Option<&Crossing>) -> Option<&'static str> {
    let Some(crossing) = crossing else {
        return Some("no_crossing");
    };
    if !crossing.entry {
        return Some("no_entry");
    }
    if crossing.x_cross.abs() > MAX_CROSS_OFFSET {
        return Some("off_center");
    }
    let Some(x_exit) = crossing.x_exit.filter(|_| crossing.reached_net_bottom) else {
        return Some("no_exit");
    };
    if x_exit.abs() > MAX_EXIT_OFFSET {
        return Some("exit_off_center");
    }
    if crossing.rebound {
        return Some("rebound");
    }
    if crossing.hidden_frames < MIN_HIDDEN_FRAMES {
        return Some("not_occluded");
    }
    None
}

/// Lower is better: any fragment that proves a make wins over a fragment
/// closer to the center that lost the ball.
fn rank(crossing: &Crossing) -> (i32, f64) {
    const ORDER: [Option<&str>; 7] = [
        Some("no_entry"),
        Some("off_center"),
        Some("no_exit"),
        Some("exit_off_center"),
        Some("rebound"),
        Some("not_occluded"),
        None,
    ];
    let reason = rejection_reason(Some(crossing));
    let position = ORDER.iter().position(|item| *item == reason).unwrap_or(0);
    (-(position as i32), crossing.x_cross.abs())
}

pub fn skipped_result(reason: &str, rim_width: Option<f64>) -> Map<String, Value> {
    let mut result = Map::new();
    result.insert("version".into(), json!(RIM_CHECK_VERSION));
    result.insert("status".into(), json!("skipped"));
    result.insert("reason".into(), json!(reason));
    result.insert(
        "rim_width_px".into(),
        json!(rim_width.map(|width| round_to(width, 2))),
    );
    result
}

/// Python list slicing `series[start:end]` with negative bounds counted from
/// the end, as the desktop code relies on it near the window start.
fn py_slice(series: &[Option<f64>], start: i64, end: i64) -> &[Option<f64>] {
    let len = series.len() as i64;
    let clamp = |bound: i64| (if bound < 0 { bound + len } else { bound }).clamp(0, len) as usize;
    let (start, end) = (clamp(start), clamp(end));
    if start >= end {
        &[]
    } else {
        &series[start..end]
    }
}

/// Net motion right after the crossing minus the motion before it.
/// `series[k]` is the change of the net from frame k-1 to frame k.
pub fn net_motion_score(series: &[Option<f64>], cross_frame: usize) -> Option<f64> {
    let cross_frame = cross_frame as i64;
    let mut before: Vec<f64> = py_slice(series, (cross_frame - 24).max(1), cross_frame - 5)
        .iter()
        .flatten()
        .copied()
        .collect();
    let mut after: Vec<f64> = py_slice(series, cross_frame + 1, cross_frame + 21)
        .iter()
        .flatten()
        .copied()
        .collect();
    if before.is_empty() || after.is_empty() {
        return None;
    }
    before.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    after.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = before.len() / 2;
    let median = if before.len() % 2 == 1 {
        before[middle]
    } else {
        (before[middle - 1] + before[middle]) / 2.0
    };
    let top = &after[after.len().saturating_sub(5)..];
    Some(round_to(
        top.iter().sum::<f64>() / top.len() as f64 - median,
        3,
    ))
}

/// Judge one candidate from per-frame detections in source coordinates.
/// `frames` holds consecutive source frames.
pub fn assess_rim_crossing(
    frames: &[Vec<Detection>],
    rim: &Rim,
    net_motion_series: Option<&[Option<f64>]>,
) -> Map<String, Value> {
    if rim.width < MIN_RIM_WIDTH_PX {
        return skipped_result("small_rim", Some(rim.width));
    }
    if frames.is_empty() {
        return skipped_result("no_frames", Some(rim.width));
    }
    let frames_balls = drop_static(
        frames
            .iter()
            .enumerate()
            .map(|(index, detections)| frame_balls(detections, rim, index))
            .collect(),
    );
    let tracks = build_tracks(&frames_balls);
    let crossings: Vec<Crossing> = (0..tracks.len())
        .flat_map(|index| track_crossings(index, &tracks))
        .collect();
    let mut best: Option<&Crossing> = None;
    for crossing in &crossings {
        if best.is_none_or(|current| rank(crossing) < rank(current)) {
            best = Some(crossing);
        }
    }
    let reason = rejection_reason(best);
    let mut result = Map::new();
    result.insert("version".into(), json!(RIM_CHECK_VERSION));
    result.insert(
        "status".into(),
        json!(if reason.is_none() { "made" } else { "rejected" }),
    );
    result.insert("reason".into(), json!(reason));
    result.insert("rim_width_px".into(), json!(round_to(rim.width, 2)));
    result.insert("frames".into(), json!(frames.len()));
    result.insert("ball_approached".into(), json!(ball_approached(&tracks)));
    if let Some(best) = best {
        result.insert("x_cross".into(), json!(best.x_cross));
        result.insert("entry".into(), json!(best.entry));
        result.insert("x_exit".into(), json!(best.x_exit));
        result.insert("reached_net_bottom".into(), json!(best.reached_net_bottom));
        result.insert("rebound".into(), json!(best.rebound));
        result.insert("hidden_frames".into(), json!(best.hidden_frames));
        result.insert("track_points".into(), json!(best.track_points));
        result.insert("track_span".into(), json!(best.track_span));
        result.insert("cross_frame".into(), json!(best.cross_frame));
    }
    let net_motion = match (best, net_motion_series) {
        (Some(best), Some(series)) if !series.is_empty() => {
            net_motion_score(series, best.cross_frame)
        }
        _ => None,
    };
    result.insert("net_motion".into(), json!(net_motion));
    result
}

fn status(result: &Map<String, Value>) -> Option<&str> {
    result.get("status").and_then(Value::as_str)
}

fn reason(result: &Map<String, Value>) -> Option<&str> {
    result.get("reason").and_then(Value::as_str)
}

fn number(result: &Map<String, Value>, key: &str) -> Option<f64> {
    result.get(key).and_then(Value::as_f64)
}

/// Drop the occlusion rule for a video whose net does not hide the ball.
/// `results` are the rim checks of every candidate of one video. Without the
/// rule a central crossing must move the net instead.
pub fn apply_video_occlusion(results: Vec<Map<String, Value>>) -> Vec<Map<String, Value>> {
    let judged = |result: &Map<String, Value>| matches!(status(result), Some("made" | "rejected"));
    let central: Vec<&Map<String, Value>> = results
        .iter()
        .filter(|result| judged(result) && matches!(reason(result), None | Some("not_occluded")))
        .collect();
    let share = (!central.is_empty()).then(|| {
        central
            .iter()
            .filter(|result| reason(result).is_none())
            .count() as f64
            / central.len() as f64
    });
    let use_rule =
        central.len() < MIN_OCCLUSION_SAMPLES || share.unwrap_or(0.0) >= MIN_OCCLUDED_SHARE;
    results
        .into_iter()
        .map(|mut result| {
            if !judged(&result) {
                return result;
            }
            result.insert(
                "occluded_share".into(),
                json!(share.map(|value| round_to(value, 3))),
            );
            result.insert("occlusion_rule".into(), json!(use_rule));
            if !use_rule && matches!(reason(&result), None | Some("not_occluded")) {
                let still =
                    number(&result, "net_motion").is_some_and(|motion| motion < MIN_NET_MOTION);
                result.insert(
                    "status".into(),
                    json!(if still { "rejected" } else { "made" }),
                );
                result.insert(
                    "reason".into(),
                    json!(if still { Some("net_still") } else { None }),
                );
            }
            result
        })
        .collect()
}

/// Turn a rejection that no detection proves into "unsure". `refined` tells
/// whether the refine stage saw a crossing.
pub fn keep_unless_seen(mut result: Map<String, Value>, refined: bool) -> Map<String, Value> {
    if status(&result) != Some("rejected") {
        return result;
    }
    let proven = match reason(&result) {
        Some("no_crossing" | "no_entry" | "no_exit") => {
            !refined
                && !result
                    .get("ball_approached")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
        }
        Some("off_center") => {
            number(&result, "track_points").unwrap_or(0.0) >= MIN_OFF_CENTER_TRACK_POINTS as f64
                && number(&result, "track_span").unwrap_or(0.0) >= MIN_OFF_CENTER_TRACK_SPAN
        }
        Some("not_occluded") => {
            number(&result, "occluded_share").unwrap_or(0.0) >= STRONG_OCCLUDED_SHARE
        }
        _ => true,
    };
    if !proven {
        result.insert("status".into(), json!("unsure"));
    }
    result
}

#[derive(Deserialize)]
struct CandidateInput {
    rim: Option<Rim>,
    #[serde(default)]
    frames: Option<Vec<Vec<Detection>>>,
    #[serde(default)]
    net_motion_series: Option<Vec<Option<f64>>>,
    #[serde(default)]
    refined: bool,
}

#[derive(Deserialize)]
struct VideoInput {
    candidates: Vec<CandidateInput>,
}

/// Judge every candidate of one video, in the order the desktop engine uses:
/// per-candidate check, video occlusion rule, then keep-unless-seen.
pub fn judge_video_json(input: &str) -> Result<String, serde_json::Error> {
    let video: VideoInput = serde_json::from_str(input)?;
    let checks: Vec<Map<String, Value>> = video
        .candidates
        .iter()
        .map(|candidate| match (&candidate.rim, &candidate.frames) {
            (Some(rim), Some(frames)) => {
                assess_rim_crossing(frames, rim, candidate.net_motion_series.as_deref())
            }
            (rim, _) => skipped_result("no_frames", rim.as_ref().map(|rim| rim.width)),
        })
        .collect();
    let results: Vec<Value> = apply_video_occlusion(checks)
        .into_iter()
        .zip(&video.candidates)
        .map(|(check, candidate)| Value::Object(keep_unless_seen(check, candidate.refined)))
        .collect();
    serde_json::to_string(&json!({ "results": results }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RIM: Rim = Rim {
        center_x: 500.0,
        rim_y: 300.0,
        width: 50.0,
    };

    fn ball(x: f64, y: f64, confidence: f64) -> Detection {
        let cx = RIM.center_x + x * RIM.width;
        let cy = RIM.rim_y + y * RIM.width;
        let half = RIM.width * 0.25;
        Detection {
            name: "ball".into(),
            confidence,
            xyxy: [cx - half, cy - half, cx + half, cy + half],
        }
    }

    /// Ball falling from above the rim to below the net, one frame per step.
    fn fall(x_start: f64, x_end: f64, hidden: &[usize]) -> Vec<Vec<Detection>> {
        (0..16)
            .map(|step| {
                let y = -1.5 + step as f64 * 0.25;
                let x = x_start + (x_end - x_start) * step as f64 / 15.0;
                if hidden.contains(&step) {
                    vec![]
                } else {
                    vec![ball(x, y, 0.85)]
                }
            })
            .collect()
    }

    fn make() -> Vec<Vec<Detection>> {
        fall(0.3, -0.2, &[6, 7])
    }

    fn assess(frames: &[Vec<Detection>]) -> Map<String, Value> {
        assess_rim_crossing(frames, &RIM, None)
    }

    fn reason_of(result: &Map<String, Value>) -> Option<&str> {
        reason(result)
    }

    #[test]
    fn ball_hidden_by_net_while_passing_the_rim_is_a_make() {
        let result = assess(&make());
        assert_eq!(status(&result), Some("made"));
        assert_eq!(result["hidden_frames"], json!(2));
    }

    #[test]
    fn ball_that_stays_fully_visible_fell_in_front_of_the_rim() {
        let result = assess(&fall(0.3, -0.2, &[]));
        assert_eq!(reason_of(&result), Some("not_occluded"));
    }

    #[test]
    fn a_ball_rolling_on_the_rim_edge_can_still_be_a_make() {
        let result = assess(&fall(0.8, 0.4, &[6, 7]));
        let x_cross = number(&result, "x_cross").unwrap();
        assert!(0.5 < x_cross && x_cross < 0.7);
        assert_eq!(status(&result), Some("made"));
    }

    #[test]
    fn crossing_outside_the_rim_opening_is_rejected() {
        let result = assess(&fall(0.9, 0.6, &[6, 7]));
        assert_eq!(reason_of(&result), Some("off_center"));
    }

    #[test]
    fn ball_bouncing_back_up_near_the_rim_is_rejected() {
        let mut frames = fall(0.3, -0.2, &[6]);
        let tail = frames.split_off(10);
        frames.push(vec![ball(0.0, 0.25, 0.85)]);
        frames.push(vec![ball(0.0, 0.3, 0.85)]);
        frames.extend(tail);
        assert_eq!(reason_of(&assess(&frames)), Some("rebound"));
    }

    #[test]
    fn ball_lost_before_leaving_the_net_is_rejected() {
        assert_eq!(reason_of(&assess(&make()[..9])), Some("no_exit"));
    }

    #[test]
    fn static_false_detection_does_not_create_a_crossing() {
        let frames: Vec<Vec<Detection>> = (0..20).map(|_| vec![ball(0.1, -0.4, 0.6)]).collect();
        assert_eq!(reason_of(&assess(&frames)), Some("no_crossing"));
    }

    #[test]
    fn small_rims_are_left_for_manual_review() {
        let rim = Rim { width: 17.0, ..RIM };
        let result = assess_rim_crossing(&make(), &rim, None);
        assert_eq!(status(&result), Some("skipped"));
        assert_eq!(reason_of(&result), Some("small_rim"));
    }

    #[test]
    fn tight_crop_stays_inside_the_frame() {
        assert_eq!(tight_crop_box(&RIM, 1920.0, 1080.0), [375, 212, 625, 462]);
        let edge = Rim {
            center_x: 10.0,
            rim_y: 10.0,
            ..RIM
        };
        assert_eq!(tight_crop_box(&edge, 1920.0, 1080.0), [0, 0, 250, 250]);
    }

    fn video(hidden_makes: usize, visible_falls: usize) -> Vec<Map<String, Value>> {
        let mut results: Vec<Map<String, Value>> =
            (0..hidden_makes).map(|_| assess(&make())).collect();
        results.extend((0..visible_falls).map(|_| assess(&fall(0.3, -0.2, &[]))));
        results
    }

    #[test]
    fn occlusion_is_ignored_when_the_net_rarely_hides_the_ball() {
        let judged = apply_video_occlusion(video(2, 8));
        assert!(judged.iter().all(|result| status(result) == Some("made")));
        assert_eq!(judged[0]["occlusion_rule"], json!(false));
        assert_eq!(judged[0]["occluded_share"], json!(0.2));
    }

    #[test]
    fn occlusion_rule_is_kept_when_the_net_hides_makes_or_too_few_crossings() {
        let judged = apply_video_occlusion(video(9, 2));
        assert_eq!(reason_of(&judged[10]), Some("not_occluded"));
        let judged = apply_video_occlusion(video(1, 6));
        assert_eq!(reason_of(&judged[6]), Some("not_occluded"));
    }

    fn net(push: f64) -> Vec<Option<f64>> {
        let mut series = vec![None];
        series.extend(std::iter::repeat_n(Some(1.0), 36));
        series.extend(std::iter::repeat_n(Some(push), 30));
        series
    }

    fn late_fall(push: f64, hidden: &[usize]) -> Map<String, Value> {
        let mut frames: Vec<Vec<Detection>> = (0..30).map(|_| vec![]).collect();
        frames.extend(fall(0.3, -0.2, hidden));
        assess_rim_crossing(&frames, &RIM, Some(&net(push)))
    }

    #[test]
    fn net_motion_is_the_push_after_the_crossing_over_the_quiet_before() {
        let result = late_fall(9.0, &[]);
        assert_eq!(result["cross_frame"], json!(36));
        assert_eq!(number(&result, "net_motion"), Some(8.0));
        assert_eq!(net_motion_score(&net(9.0), 5), None);
    }

    #[test]
    fn a_still_net_rejects_the_crossing_when_the_net_does_not_hide_the_ball() {
        let mut results = vec![late_fall(9.0, &[6, 7])];
        results.extend((0..7).map(|_| late_fall(9.0, &[])));
        results.push(late_fall(0.5, &[]));
        results.push(late_fall(1.5, &[]));
        let judged = apply_video_occlusion(results);
        assert!(judged[..8]
            .iter()
            .all(|result| status(result) == Some("made")));
        assert_eq!(reason_of(&judged[8]), Some("net_still"));
        assert_eq!(status(&judged[9]), Some("made"));
    }

    fn rejected(reason: &str, features: Value) -> Map<String, Value> {
        let mut result = Map::new();
        result.insert("status".into(), json!("rejected"));
        result.insert("reason".into(), json!(reason));
        if let Value::Object(features) = features {
            result.extend(features);
        }
        result
    }

    fn kept(result: Map<String, Value>, refined: bool) -> String {
        status(&keep_unless_seen(result, refined))
            .unwrap()
            .to_string()
    }

    #[test]
    fn an_unseen_ball_excludes_only_when_refine_did_not_see_a_crossing_either() {
        for reason in ["no_crossing", "no_entry", "no_exit"] {
            assert_eq!(kept(rejected(reason, json!({})), true), "unsure");
            assert_eq!(kept(rejected(reason, json!({})), false), "rejected");
            let approached = json!({"ball_approached": true});
            assert_eq!(kept(rejected(reason, approached), false), "unsure");
        }
    }

    #[test]
    fn a_ball_falling_toward_the_rim_is_noticed_even_when_it_is_lost_at_the_rim() {
        let result = assess(&fall(0.3, -0.2, &[])[..6]);
        assert_eq!(reason_of(&result), Some("no_crossing"));
        assert_eq!(result["ball_approached"], json!(true));
    }

    #[test]
    fn an_off_center_crossing_excludes_only_on_a_long_falling_track() {
        let short = json!({"track_points": 9, "track_span": 2.0});
        let jersey = json!({"track_points": 12, "track_span": 0.45});
        let fall_track = json!({"track_points": 10, "track_span": 1.0});
        assert_eq!(kept(rejected("off_center", short), true), "unsure");
        assert_eq!(kept(rejected("off_center", jersey), true), "unsure");
        assert_eq!(kept(rejected("off_center", fall_track), true), "rejected");
        let result = assess(&fall(1.2, 0.9, &[]));
        assert_eq!(number(&result, "track_span"), Some(3.75));
    }

    #[test]
    fn a_visible_ball_excludes_only_where_the_net_hides_nearly_every_make() {
        let far = json!({"occluded_share": 0.7});
        let close = json!({"occluded_share": 0.86});
        assert_eq!(kept(rejected("not_occluded", far), true), "unsure");
        assert_eq!(kept(rejected("not_occluded", close), true), "rejected");
    }

    #[test]
    fn seen_misses_and_makes_are_kept_as_judged() {
        for reason in ["exit_off_center", "rebound", "net_still"] {
            assert_eq!(kept(rejected(reason, json!({})), true), "rejected");
        }
        let mut made = Map::new();
        made.insert("status".into(), json!("made"));
        assert_eq!(keep_unless_seen(made.clone(), false), made);
    }

    #[test]
    fn python_rounding_sends_halves_to_even() {
        assert_eq!(round_to(0.5, 0), 0.0);
        assert_eq!(round_to(1.5, 0), 2.0);
        assert_eq!(round_to(-2.5, 0), -2.0);
        assert_eq!(round_to(0.1234, 3), 0.123);
    }
}
