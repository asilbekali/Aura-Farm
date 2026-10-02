"""
100 aura edit styles, generated from building blocks (color grade, structure, beat effects,
overlays, text). Each style leans toward a vibe, so your face decides the kind of edit:
  mentalist -> dark, glitchy, cold-blooded
  smile     -> warm, dreamy, sparkly
  pose      -> hard, cold, aggressive
Run `python styles.py` to list them all.
"""

import random
from dataclasses import dataclass

# name: (mono, saturation, contrast, lift, shadow tint BGR, highlight tint BGR)
GRADES = {
    "Noir":         (True, 0.0, 9, 0, (6, 0, -4), (4, 0, -2)),
    "Red Noir":     (True, 0.0, 10, -4, (-8, -8, 14), (-4, -4, 10)),
    "Bleach":       (False, 0.3, 10, 0, (4, 2, 0), (0, 0, 0)),
    "Ice":          (False, 0.3, 7, 4, (30, 10, -10), (24, 12, -6)),
    "Cold Steel":   (False, 0.45, 8, 0, (18, 6, -10), (8, 4, -4)),
    "Teal Orange":  (False, 1.15, 6, 0, (24, 10, -14), (-22, 4, 26)),
    "Crimson":      (False, 0.6, 9, -6, (-6, -10, 18), (-10, -6, 22)),
    "Matrix":       (False, 0.5, 8, -4, (-6, 18, -10), (-10, 24, -6)),
    "Purple Haze":  (False, 0.9, 6, 4, (30, -10, 18), (16, -4, 20)),
    "Neon":         (False, 1.6, 7, 0, (36, -14, 30), (30, 24, -12)),
    "Golden Hour":  (False, 1.1, 5, 6, (-14, 0, 14), (-24, 8, 30)),
    "Warm Dream":   (False, 1.05, 3, 14, (6, 0, 12), (-8, 4, 18)),
    "Sepia":        (True, 0.0, 6, 4, (-20, -2, 14), (-26, 0, 20)),
    "Rose":         (False, 0.9, 4, 10, (16, 0, 20), (8, 6, 22)),
}

STRUCTURES = {  # how your clip is played between the drop and the freeze-frame
    "Slow-Mo": "slowmo", "Stutter": "stutter", "Boomerang": "boomerang",
    "Freeze Cuts": "freeze_cuts", "Speed Ramp": "speed_ramp", "Rewind": "rewind",
}
BEAT_FX = ["zoom", "shake", "tilt", "rgb", "flash", "strobe", "glitch", "zoomblur", "invert"]
OVERLAYS = ["vhs", "leak", "sparkle", "speedlines", "burn", "letterbox", "scanlines"]
FONTS = {  # macOS fonts, with fallbacks handled by the renderer
    "impact": "/System/Library/Fonts/Supplemental/Impact.ttf",
    "futura": "/System/Library/Fonts/Supplemental/Futura.ttc",
    "bodoni": "/System/Library/Fonts/Supplemental/Bodoni 72.ttc",
    "didot": "/System/Library/Fonts/Supplemental/Didot.ttc",
    "copperplate": "/System/Library/Fonts/Supplemental/Copperplate.ttc",
    "avenir": "/System/Library/Fonts/Avenir Next Condensed.ttc",
    "blackletter": "/System/Library/Fonts/Supplemental/Trattatello.ttf",
    "arialblack": "/System/Library/Fonts/Supplemental/Arial Black.ttf",
}
TEXT_ANIMS = ["pop", "glitch", "type", "slide", "flicker"]
INTROS = ["push", "blackout", "blur", "flash"]

PHRASES = {
    "mentalist": ["THE MENTALIST", "MIND GAMES", "CALCULATED", "LOCKED IN", "VILLAIN ARC",
                  "I KNEW IT", "CHECKMATE", "READ YOUR MIND", "PLANNED", "TOO SMART"],
    "smile": ["AURA +1000", "MAIN CHARACTER", "THAT SMILE", "HEARTBREAKER", "GOLDEN",
              "RIZZ", "UNREAL", "GLOW UP", "TOO PURE", "HIM"],
    "pose": ["AURA +1000", "UNBOTHERED", "DIFFERENT BREED", "COLD", "SIGMA",
             "BUILT DIFFERENT", "LEGEND", "NO CAP", "HIM", "UNTOUCHABLE"],
}
OUTROS = ["AURA FARMED", "AURA: MAX", "+{aura} AURA", "AURA LEVEL {aura}"]

# What each vibe tends to get
VIBE_POOLS = {
    "mentalist": dict(grades=["Noir", "Red Noir", "Bleach", "Crimson", "Matrix", "Purple Haze", "Cold Steel"],
                      fx=["glitch", "rgb", "zoom", "invert", "strobe", "tilt", "zoomblur", "flash"],
                      overlays=["vhs", "scanlines", "letterbox", "burn"],
                      fonts=["bodoni", "didot", "copperplate", "blackletter", "futura"],
                      anims=["glitch", "type", "flicker", "slide"]),
    "smile": dict(grades=["Golden Hour", "Warm Dream", "Rose", "Purple Haze", "Teal Orange", "Sepia", "Neon"],
                  fx=["zoom", "flash", "zoomblur", "rgb", "tilt", "shake"],
                  overlays=["leak", "sparkle", "burn", "letterbox"],
                  fonts=["futura", "avenir", "didot", "bodoni", "impact"],
                  anims=["pop", "slide", "type", "flicker"]),
    "pose": dict(grades=["Cold Steel", "Ice", "Teal Orange", "Bleach", "Neon", "Noir", "Crimson"],
                 fx=["shake", "zoom", "strobe", "flash", "rgb", "glitch", "zoomblur", "tilt"],
                 overlays=["speedlines", "letterbox", "vhs", "scanlines", "leak"],
                 fonts=["impact", "arialblack", "futura", "avenir", "copperplate"],
                 anims=["pop", "glitch", "slide", "flicker"]),
}
ADJECTIVES = ["Midnight", "Phantom", "Velvet", "Savage", "Silent", "Electric", "Royal", "Shadow",
              "Frozen", "Crystal", "Toxic", "Golden", "Cosmic", "Ruthless", "Lunar", "Infernal",
              "Silk", "Static", "Venom", "Halo"]


@dataclass(frozen=True)
class Style:
    id: int
    name: str
    vibe: str
    grade: str
    structure: str
    fx: tuple
    overlays: tuple
    font: str
    anim: str
    intro: str
    phrase: str
    outro: str
    intensity: float  # 0.7 (smooth) .. 1.3 (violent)
    zoom_out: bool    # outro zooms out instead of in


def build_styles(count=100, seed=2026):
    rng = random.Random(seed)
    vibes = ["mentalist", "smile", "pose"]
    styles, seen = [], set()
    while len(styles) < count:
        vibe = vibes[len(styles) % 3]
        pool = VIBE_POOLS[vibe]
        grade = rng.choice(pool["grades"])
        structure = rng.choice(list(STRUCTURES))
        fx = tuple(sorted(rng.sample(pool["fx"], rng.randint(2, 4))))
        key = (vibe, grade, structure, fx)
        if key in seen:
            continue
        seen.add(key)
        overlays = tuple(sorted(rng.sample(pool["overlays"], rng.randint(1, 2))))
        name = f"{rng.choice(ADJECTIVES)} {grade} {structure}"
        styles.append(Style(
            id=len(styles) + 1, name=name, vibe=vibe, grade=grade, structure=STRUCTURES[structure],
            fx=fx, overlays=overlays, font=rng.choice(pool["fonts"]), anim=rng.choice(pool["anims"]),
            intro=rng.choice(INTROS), phrase=rng.choice(PHRASES[vibe]), outro=rng.choice(OUTROS),
            intensity=round(rng.uniform(0.75, 1.3), 2), zoom_out=rng.random() < 0.25))
    return styles


STYLES = build_styles()


class StylePicker:
    """Picks a style for the vibe, never repeating the recent ones."""

    def __init__(self, memory=12):
        self.recent = []
        self.memory = memory

    def pick(self, vibe, score):
        own = [s for s in STYLES if s.vibe == vibe and s.id not in self.recent]
        other = [s for s in STYLES if s.vibe != vibe and s.id not in self.recent]
        # Mostly your vibe's styles; a stronger move unlocks more intense ones
        pool = own if own and random.random() < 0.85 else (other or own)
        weights = [1 + max(0.0, score - 0.6) * 4 * (s.intensity - 0.7) for s in pool]
        style = random.choices(pool, weights=weights)[0]
        self.recent = (self.recent + [style.id])[-self.memory:]
        return style


if __name__ == "__main__":
    for s in STYLES:
        print(f"#{s.id:03d} {s.name:34s} {s.vibe:9s} {s.structure:11s} fx={','.join(s.fx):28s} "
              f"ov={','.join(s.overlays):20s} {s.font}/{s.anim} \"{s.phrase}\"")
