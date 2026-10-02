"""
Download free, Creative Commons licensed phonk tracks into music/ via Openverse
(https://openverse.org - searches Freesound, Wikimedia Commons, Jamendo, ...).
No account or API key needed. Credits are saved to music/credits.json and CREDITS.md.

Run:  .venv/bin/python fetch_music.py               (default phonk searches)
      .venv/bin/python fetch_music.py "slowed phonk" "dark trap"
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from music import MUSIC, MusicLibrary

API = "https://api.openverse.org/v1/audio/"
QUERIES = ["phonk", "drift phonk", "phonk house", "memphis phonk", "brazilian phonk",
           "gym phonk", "aggressive phonk", "krushfunk", "funk phonk", "cowbell phonk"]
# "nd" (no derivatives) licenses forbid cutting tracks into edits, so they're skipped
OK_LICENSES = {"cc0", "pdm", "by", "by-sa", "by-nc", "by-nc-sa"}
MIN_SECONDS = 30
HEADERS = {"User-Agent": "AuraFarm/1.0 (personal fun project)"}


def search(query):
    url = API + "?" + urllib.parse.urlencode({"q": query, "page_size": 20})
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
        return json.load(r).get("results", [])


def usable(r):
    text = (r["title"] + " " + " ".join(t["name"] for t in r.get("tags") or [])).lower()
    return (r["license"] in OK_LICENSES
            and (r.get("duration") or 0) >= MIN_SECONDS * 1000
            and "sample" not in r["title"].lower()
            and not r["title"].startswith("LL-Q")  # pronunciation recordings
            and any(w in text for w in ("phonk", "krushfunk", "drift", "memphis")))


def download(url, dest, tries=4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as resp:
                dest.write_bytes(resp.read())
            return True
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                print(f"   download failed: {e}")
                return False
            time.sleep(10 * (attempt + 1))  # rate limited: wait and retry
        except Exception as e:
            print(f"   download failed: {e}")
            return False


def filename(r):
    ext = Path(urllib.parse.urlparse(r["url"]).path).suffix.lower() or "." + r["filetype"]
    ext = ".ogg" if ext in (".ogx", ".oga") else ext
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{r['creator']}_{r['title']}").strip("_")[:80]
    return safe + ext


def main():
    MUSIC.mkdir(exist_ok=True)
    credits_path = MUSIC / "credits.json"
    credits = json.loads(credits_path.read_text()) if credits_path.exists() else {}
    seen = {c["id"] for c in credits.values()}

    for q in sys.argv[1:] or QUERIES:
        try:
            results = search(q)
        except Exception as e:
            print(f"search '{q}' failed: {e}")
            continue
        for r in results:
            if r["id"] in seen or not usable(r):
                continue
            seen.add(r["id"])
            name = filename(r)
            dest = MUSIC / name
            print(f"[{r['license']}] {r['title'][:60]} - {r['creator']}")
            if not download(r["url"], dest):
                continue
            credits[name] = {"id": r["id"], "title": r["title"], "creator": r["creator"],
                             "license": f"CC {r['license'].upper()}" if r["license"] not in ("cc0", "pdm")
                             else r["license"].upper(),
                             "license_url": r.get("license_url"), "source": r.get("foreign_landing_url")}
            credits_path.write_text(json.dumps(credits, indent=1))

    lines = ["# Music credits\n", "Tracks in music/ downloaded via Openverse. Thank you to the artists!\n"]
    for name, c in sorted(credits.items()):
        lines.append(f"- **{c['title']}** by {c['creator']} - {c['license']} ({c['license_url']}) - {c['source']}")
    (Path(__file__).parent / "CREDITS.md").write_text("\n".join(lines) + "\n")

    print(f"\n{len(credits)} tracks in music/. Finding the drops and beats (one time)...")
    MusicLibrary().scan()
    print("Done! Run: python aura_farm.py")


if __name__ == "__main__":
    main()
