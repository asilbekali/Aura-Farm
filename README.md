# Aura Farm

Opens your webcam and runs 5-second rounds. During a round it watches your face and
pose and remembers your best aura move. When the timer hits 0 it instantly plays an
aura edit of **you**, then starts the next round.

| Vibe | How to trigger |
|---|---|
| Beautiful smile | real, calm smile, eyes open |
| Mentalist look | narrowed eyes + smirk or side-eye + head tilt/turn, no grin |
| Cool pose | hand on chin / temple / jawline, hair swipe, straight face |

Shaky/fast movement lowers the score — aura is calm and controlled.
If your best score in a round is below 0.35, the round is skipped.

## Edits: 100 styles, real phonk
- **100 edit styles** (`python styles.py` lists them): 14 color grades, 6 structures
  (slow-mo, stutter, boomerang, freeze cuts, speed ramp, rewind), beat effects (zoom punch,
  shake, tilt, RGB split, glitch, strobe, zoom blur, invert, flash), overlays (VHS, scanlines,
  light leaks, sparkles, speed lines, film burn, letterbox), fonts and text animations.
  Your vibe picks the family (mentalist → dark/glitchy, smile → warm/dreamy, pose → hard/cold),
  a stronger move unlocks more intense styles, and recent styles aren't repeated.
- **Music** comes from `music/`. Every track is analyzed once to find its tempo, beats and
  biggest drops, and the edit's effects hit on the real beats. With no tracks, a phonk beat
  (cowbell + 808) is generated instead.
- Each edit is saved to `captures/` as an mp4 with sound.
- Your own `.mp4` edits in `edits/smile`, `edits/mentalist`, `edits/pose` play instead, if present.

## Get free phonk

    python fetch_music.py                     # default phonk searches
    python fetch_music.py "slowed phonk"      # your own search

Downloads Creative Commons phonk from [Openverse](https://openverse.org) (Freesound,
Wikimedia Commons, Jamendo...), no account needed. Credits go to `CREDITS.md` and are shown
in each edit. You can also drop your own `.mp3`/`.wav` files into `music/`.

## Run

    cd ~/Desktop/AuraFarm
    source .venv/bin/activate
    python aura_farm.py              # or --fullscreen, --interval 8, --camera 1, --no-capture

Keys: `q` quit, `space` end the round now.

## Fresh setup (another machine)

    brew install ffmpeg
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    python fetch_music.py

Note: mediapipe 1.0.x crashes on macOS ("Service is unavailable"); keep 0.10.35.
