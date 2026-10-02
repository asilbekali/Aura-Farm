"""
Aura Farm - watches you through the webcam and plays an aura edit
when you pull off an aura move (beautiful smile, mentalist look, cool pose).

Run:  .venv/bin/python aura_farm.py          (q = quit, space = end round now)
"""

import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import collections
import math
import random
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from aura_edit import AuraEdit
from music import MusicLibrary
from styles import STYLES, StylePicker

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
EDITS = ROOT / "edits"
CAPTURES = ROOT / "captures"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

# Each vibe maps to a folder inside edits/. Empty folders fall back to any edit.
VIBES = {
    "smile": "Beautiful smile",
    "mentalist": "Mentalist look",
    "pose": "Cool pose",
}
VIBE_COLORS = {  # BGR
    "smile": (180, 105, 255),
    "mentalist": (255, 80, 160),
    "pose": (60, 200, 255),
}

# Tuning
ROUND_SECONDS = 5      # every round, your best move decides which edit plays
MIN_AURA = 0.35        # best move must at least reach this, or the round is skipped
EMA_ALPHA = 0.25
DETECT_WIDTH = 480     # detection runs on a downscaled frame for speed
CLIP_SECONDS = 4.0     # how much of your move gets saved to captures/


def norm(x, lo, hi):
    """Map x from [lo, hi] to [0, 1], clamped."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# --------------------------------------------------------------------------- detection

@dataclass
class Detector:
    face: vision.FaceLandmarker
    pose: vision.PoseLandmarker
    prev_center: tuple | None = None
    speed: float = 0.0

    @classmethod
    def create(cls):
        face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task"), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1,
        ))
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task"), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        ))
        return cls(face, pose)

    def close(self):
        self.face.close()
        self.pose.close()

    def scores(self, frame_bgr, ts_ms):
        """Return ({vibe: 0..1}, face_landmarks or None, pose_landmarks or None)."""
        h, w = frame_bgr.shape[:2]
        if w > DETECT_WIDTH:  # landmarks are normalized, so a smaller image gives the same result faster
            frame_bgr = cv2.resize(frame_bgr, (DETECT_WIDTH, int(h * DETECT_WIDTH / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        face_res = self.face.detect_for_video(image, ts_ms)
        pose_res = self.pose.detect_for_video(image, ts_ms)

        out = {v: 0.0 for v in VIBES}
        if not face_res.face_landmarks:
            self.prev_center = None
            return out, None, None

        lm = face_res.face_landmarks[0]
        bs = {c.category_name: c.score for c in face_res.face_blendshapes[0]}
        pose_lm = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

        # Head geometry
        left_eye, right_eye = lm[33], lm[263]
        roll = math.degrees(math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
        face_w = math.dist((lm[234].x, lm[234].y), (lm[454].x, lm[454].y)) or 1e-6
        yaw = (lm[1].x - (lm[234].x + lm[454].x) / 2) / face_w  # ~ -0.5..0.5

        # Calmness: aura is smooth and controlled, not shaky
        center = (lm[1].x, lm[1].y)
        if self.prev_center is not None:
            v = math.dist(center, self.prev_center) / face_w
            self.speed = 0.7 * self.speed + 0.3 * v
        self.prev_center = center
        calm = 1.0 - 0.6 * norm(self.speed, 0.02, 0.10)

        smile_l, smile_r = bs.get("mouthSmileLeft", 0), bs.get("mouthSmileRight", 0)
        smile = (smile_l + smile_r) / 2
        smirk = abs(smile_l - smile_r)
        squint = (bs.get("eyeSquintLeft", 0) + bs.get("eyeSquintRight", 0)) / 2
        brow_down = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
        side_eye = max(bs.get("eyeLookOutLeft", 0), bs.get("eyeLookOutRight", 0),
                       bs.get("eyeLookInLeft", 0), bs.get("eyeLookInRight", 0))
        jaw_open = bs.get("jawOpen", 0)
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2

        # 1) Beautiful smile: real smile, not a screaming laugh, eyes open
        out["smile"] = norm(smile, 0.35, 0.80) * (1 - 0.5 * norm(jaw_open, 0.3, 0.7)) \
            * (1 - norm(blink, 0.5, 0.8)) * calm

        # 2) Mentalist look: narrowed eyes, smirk or side-eye, head tilt / turn, no big grin
        look = (0.30 * norm(squint, 0.15, 0.55)
                + 0.25 * max(norm(smirk, 0.08, 0.35), norm(side_eye, 0.35, 0.75))
                + 0.25 * max(norm(abs(roll), 5, 18), norm(abs(yaw), 0.08, 0.25))
                + 0.20 * norm(brow_down, 0.1, 0.5))
        out["mentalist"] = look * (1 - norm(smile, 0.55, 0.85)) * (1 - norm(jaw_open, 0.2, 0.5)) \
            * (1 - norm(blink, 0.6, 0.85)) * calm

        # 3) Cool pose: hand on chin / jawline / hair swipe while keeping a straight face
        if pose_lm is not None:
            sh_w = math.dist((pose_lm[11].x, pose_lm[11].y), (pose_lm[12].x, pose_lm[12].y)) or face_w * 2
            nose = (pose_lm[0].x, pose_lm[0].y)
            hand = 0.0
            for wrist in (15, 16, 19, 20):  # wrists + index fingers
                p = pose_lm[wrist]
                if p.visibility > 0.5:
                    hand = max(hand, 1 - norm(math.dist((p.x, p.y), nose) / sh_w, 0.45, 1.0))
            out["pose"] = hand * (1 - 0.5 * norm(jaw_open, 0.25, 0.6)) * calm

        return out, lm, pose_lm


# --------------------------------------------------------------------------- edits

def find_edits(vibe):
    def videos(folder):
        return [p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS] if folder.exists() else []
    return videos(EDITS / vibe) or videos(EDITS)


class Player:
    """Plays an edit (with sound) via ffplay, in its own window, on top."""

    def __init__(self, fullscreen):
        self.fullscreen = fullscreen
        self.proc = None
        self.last = {}

    @property
    def playing(self):
        return self.proc is not None and self.proc.poll() is None

    def play(self, vibe):
        edits = find_edits(vibe)
        if not edits:
            return None
        # Avoid repeating the same edit twice in a row
        choices = [e for e in edits if e != self.last.get(vibe)] or edits
        edit = random.choice(choices)
        self.last[vibe] = edit
        cmd = ["ffplay", "-autoexit", "-loglevel", "error", "-alwaysontop",
               "-window_title", f"AURA - {VIBES[vibe]}", str(edit)]
        if self.fullscreen:
            cmd.insert(1, "-fs")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        return edit

    def stop(self):
        if self.playing:
            self.proc.terminate()


# --------------------------------------------------------------------------- UI

def put_text(frame, text, org, scale, color, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


_glow_cache = {}


def aura_glow(shape, color):
    """Blurred colored border, rendered once per frame size and color (blurring every frame is slow)."""
    key = (shape, color)
    if key not in _glow_cache:
        h, w = shape[:2]
        small = np.zeros((h // 4, w // 4, 3), np.uint8)
        cv2.rectangle(small, (0, 0), (w // 4 - 1, h // 4 - 1), color, 12)
        small = cv2.GaussianBlur(small, (0, 0), 6)
        _glow_cache[key] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return _glow_cache[key]


def draw_ui(frame, smoothed, round_best, state, time_left, live_vibe, face_lm, round_secs=ROUND_SECONDS):
    h, w = frame.shape[:2]
    color = VIBE_COLORS[live_vibe]

    # Aura glow around the frame grows as you build up aura
    level = smoothed[live_vibe]
    if level > 0.3:
        cv2.addWeighted(frame, 1.0, aura_glow(frame.shape, color), min(1.5, level * 2), 0, dst=frame)

    if face_lm is not None and level > 0.3:
        xs = [p.x * w for p in face_lm]
        ys = [p.y * h for p in face_lm]
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        r = int((max(xs) - min(xs)) * (0.6 + 0.3 * level))
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, color, 3, cv2.LINE_AA)
        cv2.addWeighted(overlay, level, frame, 1 - level, 0, dst=frame)

    # Meters: bar = live score, white tick = your best this round
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (290, 30 + 32 * len(VIBES)), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.5, frame, 0.5, 0, dst=frame)
    for i, (vibe, label) in enumerate(VIBES.items()):
        y = 35 + i * 32
        cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (150, y - 12), (280, y), (80, 80, 80), -1)
        cv2.rectangle(frame, (150, y - 12), (150 + int(130 * smoothed[vibe]), y), VIBE_COLORS[vibe], -1)
        bx = 150 + int(130 * round_best[vibe])
        cv2.line(frame, (bx, y - 15), (bx, y + 3), (255, 255, 255), 2)

    # Countdown
    if state == "farming":
        secs = f"{int(math.ceil(time_left))}s"
        (tw, _), _ = cv2.getTextSize(secs, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
        put_text(frame, secs, (w - tw - 20, 50), 1.4, (255, 255, 255), 3)
        cv2.rectangle(frame, (0, h - 8), (int(w * (1 - time_left / round_secs)), h), color, -1)

    # Status
    top = max(round_best, key=round_best.get)
    if state == "playing":
        text = f"AURA FARMED: {VIBES[top]}"
    elif round_best[top] >= MIN_AURA:
        text = f"Best so far: {VIBES[top]} {round_best[top]:.2f}"
    else:
        text = "Show me your aura..."
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    put_text(frame, text, ((w - tw) // 2, h - 25), 0.9, VIBE_COLORS[top] if round_best[top] else color)


# --------------------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser(description="Aura Farm")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--interval", type=float, default=ROUND_SECONDS, help="seconds per round")
    ap.add_argument("--fullscreen", action="store_true", help="play edits fullscreen")
    ap.add_argument("--no-capture", action="store_true", help="don't save your aura moments")
    args = ap.parse_args()
    round_secs = args.interval

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. Install it with: brew install ffmpeg")
    for vibe in VIBES:
        (EDITS / vibe).mkdir(parents=True, exist_ok=True)
    if not find_edits("smile"):
        print("No videos in edits/, so Aura Farm will make phonk edits out of YOUR best moves. "
              "(Drop .mp4 edits into edits/smile, edits/mentalist, edits/pose to use those instead.)")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. On macOS allow camera access for your "
                         "terminal in System Settings > Privacy & Security > Camera.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = fps if 5 < fps < 121 else 30

    detector = Detector.create()
    player = Player(args.fullscreen)
    music = MusicLibrary()
    music.scan_in_background()  # finds beats/drops of new tracks in music/ (cached after the first time)
    styles = StylePicker()
    tracks = len(music.files())
    print(f"{len(STYLES)} edit styles, {tracks} tracks in music/" +
          ("" if tracks else " (run `python fetch_music.py` to download free phonk; using generated beats)"))
    # Warm up the edit renderer so the first real edit starts instantly
    AuraEdit([np.zeros((72, 128, 3), np.uint8)] * 2, [None] * 2, "smile", 0, STYLES[0]).frame_at(1.0)
    smoothed = {v: 0.0 for v in VIBES}
    history = collections.deque(maxlen=int(CLIP_SECONDS * fps))  # (frame, face center)
    t0 = time.monotonic()
    cv2.namedWindow("Aura Farm", cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty("Aura Farm", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def new_round():
        return {v: 0.0 for v in VIBES}, None, time.monotonic() + round_secs

    # best = (score, vibe, best_frame, clip) of the strongest moment this round
    round_best, best, round_end = new_round()
    state = "farming"  # farming -> playing -> farming
    edit = None        # AuraEdit being shown in our window (when edits/ has no videos)
    print(f"Round started: show your best aura move in the next {round_secs:.0f}s")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            face_lm = None
            if state == "farming":  # skip detection while an edit plays so the video gets the CPU
                raw, face_lm, _ = detector.scores(frame, int((now - t0) * 1000))
                for v in VIBES:
                    smoothed[v] = (1 - EMA_ALPHA) * smoothed[v] + EMA_ALPHA * raw[v]
            center = ((face_lm[10].x + face_lm[152].x) / 2, (face_lm[10].y + face_lm[152].y) / 2) \
                if face_lm is not None else None
            history.append((frame.copy(), center))
            live_vibe = max(smoothed, key=smoothed.get)
            key = cv2.waitKey(1) & 0xFF

            if state == "farming":
                for v in VIBES:
                    round_best[v] = max(round_best[v], smoothed[v])
                if best is None or smoothed[live_vibe] > best[0]:
                    best = (smoothed[live_vibe], live_vibe, frame.copy(), list(history))

                if now >= round_end or key == ord(" "):
                    score, vibe, best_frame, clip = best
                    if score >= MIN_AURA or key == ord(" "):
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        if not args.no_capture:
                            CAPTURES.mkdir(exist_ok=True)
                            cv2.imwrite(str(CAPTURES / f"{stamp}_{vibe}_best.jpg"), best_frame)
                        video = player.play(vibe)
                        if video:
                            print(f"[AURA] {VIBES[vibe]} ({score:.2f}) -> {video.name}")
                        else:
                            style = styles.pick(vibe, score)
                            track = music.pick(vibe)
                            edit = AuraEdit([f for f, _ in clip], [c for _, c in clip], vibe, score,
                                            style, track)
                            edit.play(time.monotonic())
                            edit.stamp = stamp
                            print(f"[AURA] {VIBES[vibe]} ({score:.2f}) -> style #{style.id} {style.name}, "
                                  f"music: {track['title'] if track else 'generated phonk'}")
                        state = "playing"
                    else:
                        print(f"No aura this round (best {score:.2f}). Try again!")
                        round_best, best, round_end = new_round()

            elif state == "playing":
                done = edit.finished(now) if edit else not player.playing
                if done:
                    if edit:
                        edit.stop()
                        if not args.no_capture:
                            out = CAPTURES / f"{edit.stamp}_{edit.vibe}_edit.mp4"
                            threading.Thread(target=edit.save, args=(out,)).start()  # not daemon: finishes saving on quit
                            print(f"Saving your edit -> captures/{out.name}")
                        edit = None
                    state = "farming"
                    smoothed = {v: 0.0 for v in VIBES}
                    round_best, best, round_end = new_round()
                    print(f"Next round: {round_secs:.0f}s")

            if edit is not None and state == "playing":
                view = edit.frame_at(now - edit.start)
            else:
                view = frame.copy()
                draw_ui(view, smoothed, round_best, state, max(0.0, round_end - now),
                        live_vibe, face_lm, round_secs)
            cv2.imshow("Aura Farm", view)
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if edit:
            edit.stop()
        player.stop()
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
