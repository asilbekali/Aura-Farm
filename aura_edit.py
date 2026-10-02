"""
Auto aura edit: turns your best moment into a phonk edit in one of 100 styles (styles.py),
synced to the beats of a phonk track from music/ (or a phonk beat synthesized on the spot).
Rendered live frame by frame, so it starts instantly with no encoding wait.
"""

import bisect
import math
import random
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from styles import FONTS, GRADES

SR = 44100
EDIT_HEIGHT = 720
BEATS = 16        # whole edit length in beats
DROP_BEAT = 2     # intro on your best frame before the drop
FREEZE_BEAT = 12  # main part until here, then freeze-frame on your peak
SYNTH_BPM = {"mentalist": (118, 132), "smile": (125, 138), "pose": (138, 155)}
VIBE_GLOW = {"mentalist": (255, 60, 140), "smile": (190, 120, 255), "pose": (255, 200, 60)}  # BGR


# --------------------------------------------------------------------------- phonk beat (fallback)

# --------------------------------------------------------------------------- phonk beat

def _env(n, decay):
    return np.exp(-np.arange(n) / SR * decay)


def _kick():
    t = np.arange(int(0.35 * SR)) / SR
    f = 45 + 110 * np.exp(-t * 30)
    return np.tanh(2.5 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7))


def _snare(rng):
    n = int(0.22 * SR)
    noise = np.diff(rng.standard_normal(n + 1)) * 0.5
    tone = np.sin(2 * np.pi * 185 * np.arange(n) / SR)
    return (noise * _env(n, 22) + 0.6 * tone * _env(n, 35)) * 0.8


def _hat(rng):
    n = int(0.05 * SR)
    return np.diff(rng.standard_normal(n + 1)) * _env(n, 90) * 0.25


def _cowbell(ratio):
    # The classic phonk cowbell: two detuned square waves, short decay, pitched per note
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    sq = np.sign(np.sin(2 * np.pi * 587 * ratio * t)) + np.sign(np.sin(2 * np.pi * 845 * ratio * t))
    sq = np.convolve(sq, np.ones(4) / 4, mode="same")  # soften the edges a bit
    return sq * (0.6 * _env(n, 9) + 0.4 * _env(n, 40)) * 0.22


def _bass(freq, length):
    # Distorted 808 with a small pitch drop at the start
    n = int(length * SR)
    t = np.arange(n) / SR
    f = freq * (1 + 0.5 * np.exp(-t * 25))
    return np.tanh(3 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 2.2)) * 0.55


def phonk_track(bpm, seed):
    rng = np.random.default_rng(seed)
    spb = 60 / bpm
    step = spb / 4
    n = int(SR * spb * BEATS) + SR // 2
    out = np.zeros(n)

    def add(sig, t, gain=1.0):
        i = int(t * SR)
        if i < n:
            m = min(len(sig), n - i)
            out[i:i + m] += sig[:m] * gain

    # Intro riser
    rlen = int(DROP_BEAT * spb * SR)
    riser = np.diff(rng.standard_normal(rlen + 1)) * np.linspace(0, 1, rlen) ** 3 * 0.35
    add(riser, 0)

    # One-bar cowbell motif in a minor key, repeated with a variation on the last bar
    scale = [0, 3, 5, 7, 10, 12, 15]
    rhythm = [0, 3, 6, 8, 10, 11, 14]
    motif = [scale[i] for i in rng.integers(0, len(scale), len(rhythm))]
    root = 49.0 * 2 ** (int(rng.integers(-3, 4)) / 12)  # 808 in a random key around G1
    bass_notes = [0, 0, -2, 3]

    drop = DROP_BEAT * spb
    for bar in range(math.ceil((BEATS - DROP_BEAT) / 4)):
        bar_t = drop + bar * 4 * spb
        last = bar == 2
        for s in range(16):
            t = bar_t + s * step
            if t >= BEATS * spb:
                break
            if s in (0, 7, 10):
                add(_kick(), t)
                semi = bass_notes[bar % 4]
                add(_bass(root * 2 ** (semi / 12), step * (7 if s == 0 else 3)), t)
            if s in (4, 12):
                add(_snare(rng), t)
            if s % 2 == 0 or (last and s >= 12):
                add(_hat(rng), t, 0.7 if s % 4 else 1.0)
            if s in rhythm:
                note = motif[rhythm.index(s)] + (5 if last and s > 8 else 0)
                add(_cowbell(2 ** (note / 12)), t)

    out = np.tanh(out * 1.4)
    out /= max(1e-6, np.abs(out).max()) / 0.9
    return out


def write_wav(samples, path):
    data = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def play_audio(path):
    if shutil.which("afplay"):
        return subprocess.Popen(["afplay", str(path)])
    return subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
                            stdin=subprocess.DEVNULL)


# --------------------------------------------------------------------------- building blocks

def grade_lut(name):
    mono, sat, contrast, lift, sh, hi = GRADES[name]
    x = np.arange(256) / 255
    c = 1 / (1 + np.exp(-contrast * (x - 0.5)))
    c = (c - c.min()) / (c.max() - c.min())
    chans = [np.clip(c * 255 + lift + sh[i] * (1 - c) ** 2 * 1.6 + hi[i] * c ** 2 * 1.2, 0, 255) for i in range(3)]
    return mono, sat, np.stack(chans, axis=1).astype(np.uint8).reshape(1, 256, 3)


def apply_grade(img, grade):
    mono, sat, lut = grade
    if mono:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif sat != 1.0:
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, sat, gray, 1 - sat, 0)
    return cv2.LUT(img, lut)


def load_font(name, size):
    try:
        return ImageFont.truetype(FONTS[name], size)
    except (OSError, KeyError):
        for fallback in ("impact", "futura", "avenir"):
            try:
                return ImageFont.truetype(FONTS[fallback], size)
            except OSError:
                pass
        return ImageFont.load_default(size)


def text_sprite(text, font, glow=None, fill=(255, 255, 255), stroke=4):
    """Pre-rendered RGBA text (as BGRA numpy) with outline and optional glow."""
    l, t, r, b = font.getbbox(text, stroke_width=stroke)
    pad = 30 if glow else 6
    w, h = r - l + 2 * pad, b - t + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if glow:
        g = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(g).text((pad - l, pad - t), text, font=font, fill=(glow[2], glow[1], glow[0], 255),
                               stroke_width=stroke + 4, stroke_fill=(glow[2], glow[1], glow[0], 255))
        img = Image.alpha_composite(img, g.filter(ImageFilter.GaussianBlur(12)))
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=fill + (255,),
                             stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def blend_sprite(img, sprite, cx, cy, scale=1.0, alpha=1.0):
    if alpha <= 0.01:
        return
    if abs(scale - 1) > 0.01:
        sprite = cv2.resize(sprite, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    sh, sw = sprite.shape[:2]
    H, W = img.shape[:2]
    x0, y0 = int(cx - sw / 2), int(cy - sh / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + sw), min(H, y0 + sh)
    if xa >= xb or ya >= yb:
        return
    s = sprite[ya - y0:yb - y0, xa - x0:xb - x0]
    a = (s[..., 3:4].astype(np.float32) / 255) * alpha
    roi = img[ya:yb, xa:xb]
    roi[:] = (roi * (1 - a) + s[..., :3] * a).astype(np.uint8)


# --------------------------------------------------------------------------- the edit

class AuraEdit:
    def __init__(self, clip, centers, vibe, score, style, music=None, seed=None):
        """clip: frames leading up to (and ending on) your peak; centers: normalized face centers;
        style: styles.Style; music: MusicLibrary.pick() result or None to synthesize a beat."""
        self.vibe, self.score, self.style = vibe, score, style
        self.seed = seed if seed is not None else random.randrange(1 << 30)
        rng = np.random.default_rng(self.seed)

        # --- music + beat grid
        if music:
            self.wav = Path(music["wav"])
            beats, strength = music["beats"], music["strength"]
            self.credit = music["credit"]
        else:
            bpm = float(rng.uniform(*SYNTH_BPM[vibe]))
            self.wav = Path(tempfile.gettempdir()) / f"aura_phonk_{self.seed}.wav"
            write_wav(phonk_track(bpm, self.seed), self.wav)
            beats = [k * 60 / bpm for k in range(BEATS + 1)]
            strength = [1.0 if k % 2 == 0 else 0.75 for k in range(BEATS + 1)]
            self.credit = None
        self.beats = list(beats)
        s = np.asarray(strength, np.float32)
        self.strength = list(0.45 + 0.55 * s / (s.max() + 1e-9))
        self.drop_t, self.freeze_t = beats[DROP_BEAT], beats[FREEZE_BEAT]
        self.duration = beats[-1]

        # --- frames
        h, w = clip[-1].shape[:2]
        scale = EDIT_HEIGHT / h if h > EDIT_HEIGHT else 1.0
        self.size = W, H = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
        self.frames = [cv2.resize(f, self.size, interpolation=cv2.INTER_AREA) for f in clip] \
            if (W, H) != (w, h) else list(clip)
        cs = [c for c in centers[-30:] if c is not None]
        cx, cy = (np.median([c[0] for c in cs]), np.median([c[1] for c in cs])) if cs else (0.5, 0.4)
        self.center = (float(cx * W), float(cy * H))

        # --- look
        self.grade = grade_lut(style.grade)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = np.clip(1.25 - 0.55 * d ** 2, 0.25, 1)[..., None]
        self.grain = [rng.integers(0, 16, (H, W, 1), dtype=np.uint8).repeat(3, axis=2) for _ in range(4)]
        self.glow = VIBE_GLOW[vibe]
        ov = set(style.overlays)
        if ov & {"vhs", "scanlines"}:
            lines = np.ones((H, 1, 1), np.float32)
            lines[::3] = 0.72
            self.scan = lines
        if "leak" in ov:
            small = np.zeros((H // 4, W // 2, 3), np.uint8)
            for _ in range(5):
                c = [(40, 120, 255), (80, 60, 255), (60, 200, 255), (200, 80, 255)][int(rng.integers(0, 4))]
                cv2.circle(small, (int(rng.integers(0, W // 2)), int(rng.integers(0, H // 4))),
                           int(rng.integers(H // 16, H // 6)), c, -1)
            self.leak = cv2.resize(cv2.GaussianBlur(small, (0, 0), H / 24), (W * 2, H))
        if "sparkle" in ov:
            self.sparks = rng.uniform(0, 1, (45, 4))  # x, y, phase, size
        if "speedlines" in ov:
            lines = np.zeros((H, W), np.uint8)
            for a in rng.uniform(0, 2 * np.pi, 70):
                r0 = rng.uniform(0.25, 0.4) * W
                p0 = (int(self.center[0] + r0 * math.cos(a)), int(self.center[1] + r0 * math.sin(a)))
                p1 = (int(self.center[0] + 1.5 * W * math.cos(a)), int(self.center[1] + 1.5 * W * math.sin(a)))
                cv2.line(lines, p0, p1, 255, int(rng.integers(1, 4)), cv2.LINE_AA)
            self.speed = cv2.cvtColor(cv2.GaussianBlur(lines, (3, 3), 0), cv2.COLOR_GRAY2BGR)
        if "burn" in ov:
            edge = np.clip((d - 0.75) * 2.2, 0, 1)[..., None]
            self.burn = (edge * np.array([20, 110, 255], np.float32)).astype(np.uint8)

        # --- text
        fsize = int(H * (0.105 if len(style.phrase) <= 12 else 0.085))
        font = load_font(style.font, fsize)
        self.phrase = text_sprite(style.phrase, font, glow=self.glow)
        if style.anim == "type":
            self.typed = [text_sprite(style.phrase[:i], font, glow=self.glow) if style.phrase[:i].strip() else None
                          for i in range(len(style.phrase) + 1)]
        if style.anim == "glitch":
            self.phrase_cyan = text_sprite(style.phrase, font, fill=(255, 255, 0))
            self.phrase_red = text_sprite(style.phrase, font, fill=(40, 0, 255))
        outro = style.outro.format(aura=int(score * 1000))
        self.outro = text_sprite(outro, load_font(style.font, int(H * 0.095)), glow=self.glow)
        self.vhs_label = text_sprite("PLAY  " + style.name.upper(), load_font("avenir", int(H * 0.035)), stroke=2) \
            if "vhs" in ov else None
        self.credit_sprite = text_sprite(f"music: {self.credit}"[:90], load_font("avenir", int(H * 0.024)),
                                         stroke=2) if self.credit else None

        self.audio = None
        self.start = None

    # -- playback
    def play(self, now):
        self.audio = play_audio(self.wav)
        self.start = now

    def finished(self, now):
        return self.start is not None and now - self.start >= self.duration

    def stop(self):
        if self.audio and self.audio.poll() is None:
            self.audio.terminate()

    # -- timeline helpers
    def _beat(self, t):
        k = max(0, bisect.bisect_right(self.beats, t) - 1)
        return k, t - self.beats[k]

    def _source_index(self, t, k):
        """Which clip frame to show during the main part, per structure."""
        n = len(self.frames)
        main = self.freeze_t - self.drop_t
        p = min(1.0, max(0.0, (t - self.drop_t) / main))
        slow_k = min(n, max(2, int(main * 30 * 0.5)))
        st = self.style.structure
        if st == "speed_ramp":
            return (1 - (1 - p) ** 3) * (n - 1)
        if st == "stutter":
            pb = min(1.0, max(0.0, (self.beats[k] - self.drop_t) / main))
            return min(n - 1, n - slow_k + pb * (slow_k - 1) + int((t - self.beats[k]) * 30) % 4)
        if st == "boomerang":
            phase = p * 3
            tri = 1 - abs((phase % 2) - 1)
            return n - 1 - slow_k + slow_k * tri if phase < 3 else n - 1
        if st == "freeze_cuts":
            cuts = FREEZE_BEAT - DROP_BEAT
            j = min(cuts - 1, max(0, k - DROP_BEAT))
            return (n - 1) * ((j + 1) / cuts) ** 0.6
        if st == "rewind":
            if p < 0.3:
                return (n - 1) * (1 - p / 0.3)
            return (n - 1) * ((p - 0.3) / 0.7)
        return n - slow_k + p * (slow_k - 1)  # slowmo

    # -- rendering (a pure function of t, so it can be re-rendered to a file)
    def frame_at(self, t):
        st, W, H = self.style, *self.size
        I = st.intensity
        fx = st.fx
        k, dt = self._beat(t)
        hit = self.strength[k] * math.exp(-dt * 8) if k >= DROP_BEAT else 0.0
        peak = self.frames[-1]
        part = "intro" if t < self.drop_t else "main" if t < self.freeze_t else "outro"

        # --- pick the source frame + camera
        rewinding = False
        if part == "intro":
            img = peak
            p = t / max(1e-6, self.drop_t)
            zoom = 1.0 + 0.12 * p if st.intro == "push" else 1.05
        elif part == "main":
            idx = self._source_index(t, k)
            rewinding = st.structure == "rewind" and (t - self.drop_t) / (self.freeze_t - self.drop_t) < 0.3
            img = self.frames[int(round(min(len(self.frames) - 1, max(0, idx))))]
            zoom = 1.08
        else:
            img = peak
            p = (t - self.freeze_t) / max(1e-6, self.duration - self.freeze_t)
            zoom = (1.5 - 0.35 * p ** 0.7) if st.zoom_out else (1.15 + 0.35 * p ** 1.5)

        img = apply_grade(img, self.grade)

        angle, sx, sy = 0.0, 0.0, 0.0
        if "zoom" in fx:
            zoom += 0.09 * hit * I
        if "shake" in fx:
            sx, sy = 18 * hit * I * math.sin(t * 93), 18 * hit * I * math.cos(t * 71)
        if "tilt" in fx:
            angle = 3.5 * hit * I * (1 if k % 2 else -1)
        cx, cy = self.center
        M = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
        M[0, 2] += sx + (W / 2 - cx) * (zoom - 1) * 0.6
        M[1, 2] += sy + (H / 2 - cy) * (zoom - 1) * 0.6
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)

        if "zoomblur" in fx and hit > 0.15:
            Z = cv2.getRotationMatrix2D((cx, cy), 0, 1 + 0.06 * hit * I)
            img = cv2.addWeighted(img, 0.6, cv2.warpAffine(img, Z, (W, H), borderMode=cv2.BORDER_REFLECT), 0.4, 0)

        if part == "intro":
            if st.intro == "blur":
                b = int(30 * (1 - t / self.drop_t)) * 2 + 1
                img = cv2.GaussianBlur(img, (b, b), 0)
            elif st.intro == "blackout":
                img = (img * min(1.0, (t / self.drop_t) ** 2 * 0.6)).astype(np.uint8)
            elif st.intro == "flash" and int(t * 12) % 2 == 0:
                img = cv2.addWeighted(img, 0.5, np.full_like(img, 255), 0.5, 0)
            else:
                img = (img * 0.75).astype(np.uint8)

        # --- bloom
        small = cv2.resize(img, (W // 4, H // 4))
        small = cv2.GaussianBlur(cv2.threshold(small, 200, 255, cv2.THRESH_TOZERO)[1], (0, 0), 6)
        img = cv2.addWeighted(img, 1.0, cv2.resize(small, (W, H)), 0.3, 0)

        # --- beat effects
        shift = int(12 * hit * I) if "rgb" in fx else 0
        if rewinding:
            shift = max(shift, 6)
        if shift:
            b, g, r = cv2.split(img)
            img = cv2.merge([np.roll(b, -shift, axis=1), g, np.roll(r, shift, axis=1)])
        if "glitch" in fx and hit > 0.3:
            grng = np.random.default_rng(k * 1009 + int(t * 20))
            for _ in range(6):
                y0 = int(grng.integers(0, H - 10))
                hh = int(grng.integers(6, 50))
                img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], int(grng.integers(-70, 70) * hit), axis=1)
        if "strobe" in fx and hit > 0.45 and int(t * 20) % 2 == 0:
            img = (img * 0.3).astype(np.uint8)
        if "invert" in fx and k >= DROP_BEAT and k % 4 == 0 and dt < 0.06:
            img = 255 - img

        # --- overlays
        ov = st.overlays
        if "leak" in ov:
            off = int((t * 0.12 % 1) * W)
            leak = self.leak[:, off:off + W]
            if leak.shape[1] == W:
                img = cv2.addWeighted(img, 1.0, leak, 0.35 + 0.3 * hit, 0)
        if "speedlines" in ov and hit > 0.05:
            img = cv2.addWeighted(img, 1.0, self.speed, 0.55 * hit * I, 0)
        if "burn" in ov:
            a = 0.55 + 0.3 * math.sin(t * 2.3) + (0.6 if part != "main" else 0)
            img = cv2.addWeighted(img, 1.0, self.burn, max(0.0, a), 0)
        if "sparkle" in ov:
            for x, y, ph, sz in self.sparks:
                tw = math.sin(t * 5 + ph * 30)
                if tw > 0.3:
                    px, py = int(x * W), int((y - t * 0.03 * (0.5 + sz)) % 1 * H)
                    r = int((3 + 9 * sz) * tw)
                    col = (200, 240, 255) if ph > 0.5 else (255, 255, 255)
                    cv2.line(img, (px - r, py), (px + r, py), col, 1, cv2.LINE_AA)
                    cv2.line(img, (px, py - r), (px, py + r), col, 1, cv2.LINE_AA)
                    cv2.circle(img, (px, py), max(1, r // 4), col, -1, cv2.LINE_AA)
        if hasattr(self, "scan"):
            img = (img * self.scan).astype(np.uint8)
        if "vhs" in ov:
            by = int((t * 140) % H)
            img[by:by + 10] = np.roll(img[by:by + 10], 14, axis=1)
            blend_sprite(img, self.vhs_label, 30 + self.vhs_label.shape[1] / 2, int(H * 0.08) + 22)

        img = (img * self.vignette).astype(np.uint8)
        img = cv2.subtract(cv2.add(img, self.grain[int(t * 30) % 4]), 8)

        # --- flashes: always on the drop and the freeze, plus on bars when the style has "flash"
        flashes = [self.drop_t, self.freeze_t]
        if "flash" in fx:
            flashes += [self.beats[b] for b in range(DROP_BEAT + 4, FREEZE_BEAT, 4)]
        for ft in flashes:
            if 0 <= t - ft < 0.4:
                a = (1 - (t - ft) / 0.4) * (1.0 if ft in (self.drop_t, self.freeze_t) else 0.6)
                img = cv2.addWeighted(img, 1 - a, np.full_like(img, 255), a, 0)

        if "letterbox" in ov:
            bar = int(H * 0.1)
            img[:bar] = 0
            img[H - bar:] = 0

        # --- text
        self._draw_text(img, t, k, hit, part)

        if t > self.duration - 0.6:  # fade out
            img = (img * max(0.0, (self.duration - t) / 0.6)).astype(np.uint8)
        return img

    def _draw_text(self, img, t, k, hit, part):
        st, W, H = self.style, *self.size
        y = int(H * 0.78)
        t_in = self.beats[min(DROP_BEAT + 2, len(self.beats) - 1)]
        if part == "main" and t >= t_in:
            age = t - t_in
            if st.anim == "pop":
                blend_sprite(img, self.phrase, W / 2, y, 1 + 0.5 * math.exp(-age * 10) + 0.08 * hit)
            elif st.anim == "type":
                n = min(len(self.typed) - 1, int(age / 0.06))
                if self.typed[n] is not None:
                    blend_sprite(img, self.typed[n], W / 2, y)
            elif st.anim == "slide":
                e = 1 - (1 - min(1.0, age / 0.3)) ** 3
                blend_sprite(img, self.phrase, W / 2 + (1 - e) * W, y)
            elif st.anim == "flicker":
                on = age > 0.4 or int(age * 25) % 3 != 0
                if on and not (hit > 0.7 and int(t * 30) % 2):
                    blend_sprite(img, self.phrase, W / 2, y)
            else:  # glitch
                if hit > 0.25:
                    o = int(14 * hit)
                    blend_sprite(img, self.phrase_cyan, W / 2 - o, y, alpha=0.8)
                    blend_sprite(img, self.phrase_red, W / 2 + o, y, alpha=0.8)
                blend_sprite(img, self.phrase, W / 2 + (int(t * 97) % 7 - 3) * hit * 3, y)
        elif part == "outro":
            age = t - self.freeze_t
            blend_sprite(img, self.outro, W / 2, y, 1 + 0.4 * math.exp(-age * 9) + 0.06 * hit)
        if self.credit_sprite is not None and t >= self.drop_t:
            cs = self.credit_sprite
            blend_sprite(img, cs, 16 + cs.shape[1] / 2, H - 14 - cs.shape[0] / 2, alpha=0.75)

    # -- export
    def save(self, path, fps=30):
        """Render the edit with its music to an mp4 (runs in a background thread)."""
        W, H = self.size
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-i", str(self.wav), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(int(self.duration * fps)):
            proc.stdin.write(self.frame_at(i / fps).tobytes())
        proc.stdin.close()
        proc.wait()
