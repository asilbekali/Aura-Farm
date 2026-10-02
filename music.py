"""
Music for aura edits.

- Any audio file in music/ (downloaded with fetch_music.py, or your own) is analyzed once:
  tempo + beats are detected and the hardest-hitting drops are cut into short segments
  (cached in music/.cache). Edits then sync their effects to the real beats.
- If music/ is empty, a phonk beat is synthesized on the spot (aura_edit.phonk_track).
"""

import json
import random
import subprocess
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
MUSIC = ROOT / "music"
CACHE = MUSIC / ".cache"
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".oga", ".ogx", ".flac", ".aac", ".opus", ".webm"}

SR = 22050
HOP = 512
SEG_BEATS = 16       # an edit is 16 beats long
DROP_AT = 2          # the drop lands on beat 2 of the segment
SEGMENTS_PER_TRACK = 3

# Which tracks suit which vibe, by words in the title/genre
VIBE_WORDS = {
    "mentalist": ["memphis", "atmospher", "dark", "night", "horror", "slowed", "nightmare", "ufo", "mind"],
    "smile": ["house", "chill", "funk", "song", "dream", "love", "happy"],
    "pose": ["drift", "gym", "aggressive", "hard", "brazil", "krush", "sport", "punch", "beat"],
}


# --------------------------------------------------------------------------- analysis

def decode(path, sr=SR):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32)


def onset_envelope(y):
    n = 1 + (len(y) - 1024) // HOP
    if n < 10:
        return np.zeros(1), np.zeros(1)
    idx = np.arange(1024)[None, :] + HOP * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(y[idx] * np.hanning(1024), axis=1))
    logmag = np.log1p(100 * mag)
    flux = np.maximum(0, np.diff(logmag, axis=0, prepend=logmag[:1])).sum(axis=1)
    flux -= np.convolve(flux, np.ones(16) / 16, mode="same")  # remove slow trend
    flux = np.maximum(flux, 0)
    bass = mag[:, 1:12].sum(axis=1)  # ~20-250 Hz: kicks and 808s
    return flux / (flux.max() + 1e-9), bass / (bass.max() + 1e-9)


def estimate_period(env):
    """Beat period in frames, preferring 120-160 BPM (phonk territory)."""
    fps = SR / HOP
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    bpm = 60 * fps / np.maximum(lags, 1)
    valid = (bpm >= 95) & (bpm <= 185)
    prior = np.exp(-0.5 * (np.log2(bpm / 140) / 0.5) ** 2)
    score = np.where(valid, ac * prior, -np.inf)
    return int(np.argmax(score))


def track_beats(env, period, tightness=100):
    """Dynamic-programming beat tracker (Ellis 2007)."""
    n = len(env)
    score = env.copy()
    back = -np.ones(n, int)
    offsets = np.arange(-2 * period, -period // 2)
    penalty = -tightness * np.log(-offsets / period) ** 2
    for i in range(n):
        prev = i + offsets
        ok = prev >= 0
        if ok.any():
            cand = score[prev[ok]] + penalty[ok]
            j = int(np.argmax(cand))
            score[i] += cand[j]
            back[i] = prev[ok][j]
    i = int(np.argmax(score[-period:])) + n - period
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    return np.array(beats[::-1])


def find_drops(beat_energy, count):
    """Start beats for segments whose drop (beat DROP_AT) hits hardest."""
    nb = len(beat_energy)
    scores = []
    for b in range(0, nb - SEG_BEATS):
        before = beat_energy[b:b + DROP_AT].mean()
        after = beat_energy[b + DROP_AT:b + SEG_BEATS].mean()
        scores.append(after * 1.0 + (after - before) * 1.5)
    order = np.argsort(scores)[::-1]
    picked = []
    for b in order:
        if all(abs(b - p) >= SEG_BEATS for p in picked):
            picked.append(int(b))
        if len(picked) == count:
            break
    return picked


def analyze(path):
    y = decode(path)
    env, bass = onset_envelope(y)
    if len(env) < 200:
        return []
    period = estimate_period(env)
    beats = track_beats(env, period)
    if len(beats) < SEG_BEATS + 1:
        return []
    win = max(2, period // 2)
    energy = np.array([env[b:b + win].mean() + bass[b:b + win].mean() for b in beats])
    energy /= energy.max() + 1e-9
    segs = []
    starts = find_drops(energy, SEGMENTS_PER_TRACK)
    loudness = {b: energy[b + DROP_AT:b + SEG_BEATS].mean() for b in starts}
    starts = [b for b in starts if loudness[b] >= 0.6 * max(loudness.values())]  # skip quiet parts
    for k, b in enumerate(starts):
        frames = beats[b:b + SEG_BEATS + 1]
        t0 = frames[0] * HOP / SR
        times = [round(f * HOP / SR - t0, 4) for f in frames]
        dur = times[-1] + 0.25
        wav = CACHE / f"{path.stem}_{k}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
                        "-ac", "2", "-ar", "44100", "-af", f"afade=t=out:st={dur - 0.6:.3f}:d=0.6",
                        str(wav)], check=True)
        strengths = [round(float(env[f:f + 3].max()), 3) for f in frames]
        segs.append({"wav": wav.name, "beats": times, "strength": strengths})
    return segs


# --------------------------------------------------------------------------- library

class MusicLibrary:
    def __init__(self):
        MUSIC.mkdir(exist_ok=True)
        CACHE.mkdir(exist_ok=True)
        self.index_path = CACHE / "index.json"
        try:
            self.index = json.loads(self.index_path.read_text())
        except (OSError, ValueError):
            self.index = {}
        self.credits = self._load_credits()
        self.recent = []
        self.lock = threading.Lock()
        self.ready = threading.Event()

    def _load_credits(self):
        try:
            return json.loads((MUSIC / "credits.json").read_text())
        except (OSError, ValueError):
            return {}

    def files(self):
        return sorted(p for p in MUSIC.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)

    def scan(self, log=print):
        """Analyze new/changed tracks. Safe to run in a background thread."""
        for path in self.files():
            key, mtime = path.name, path.stat().st_mtime
            entry = self.index.get(key)
            if entry and entry.get("mtime") == mtime and all((CACHE / s["wav"]).exists() for s in entry["segments"]):
                continue
            try:
                log(f"  analyzing {path.name} ...")
                segs = analyze(path)
            except Exception as e:  # a broken file shouldn't stop the app
                log(f"  skipped {path.name}: {e}")
                segs = []
            with self.lock:
                self.index[key] = {"mtime": mtime, "segments": segs}
                self.index_path.write_text(json.dumps(self.index, indent=1))
        self.credits = self._load_credits()
        self.ready.set()

    def scan_in_background(self):
        threading.Thread(target=self.scan, kwargs={"log": lambda m: None}, daemon=True).start()

    def pick(self, vibe):
        """Return a segment dict (wav path, beats, strength, title, credit) or None."""
        with self.lock:
            options = []
            for name, entry in self.index.items():
                if not (MUSIC / name).exists():
                    continue
                for seg in entry["segments"]:
                    options.append((name, seg))
        if not options:
            return None
        words = VIBE_WORDS.get(vibe, [])

        def matches(name):
            text = (name + " " + self.credits.get(name, {}).get("title", "")).lower()
            return any(w in text for w in words)

        fresh = [o for o in options if o[0] not in self.recent] or options
        preferred = [o for o in fresh if matches(o[0])]
        name, seg = random.choice(preferred if preferred and random.random() < 0.75 else fresh)
        self.recent = (self.recent + [name])[-min(3, max(1, len(self.index) - 1)):]
        info = self.credits.get(name, {})
        title = info.get("title") or Path(name).stem
        credit = f"{title} - {info['creator']} ({info['license']})" if info.get("creator") else title
        return {"wav": CACHE / seg["wav"], "beats": seg["beats"], "strength": seg["strength"],
                "title": title, "credit": credit}
