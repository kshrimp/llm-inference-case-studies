"""Build the deterministic speech WAVs for the omni pipeline probe (local, no GPU).

Source: LibriVox "The Art of War" (Sun Tzu, tr. Lionel Giles), section 1
(chapters 1-2), read by Moira Fogarty. Public domain in the US per LibriVox.
One continuous passage starting at CLIP_START_S gives nested 10 / 30 / 90 s
clips (each shorter clip is a prefix of the longer). Every clip has VARIANTS
copies whose *decoded samples* differ by +-1 LSB at one position, so the audio
encoder cache (keyed on decoded samples) misses on each measured request.

The text control uses the written form of the same passage: Project Gutenberg
eBook #132 (Lionel Giles translation, public domain in the US), from
"5, 6. _The Moral Law_ causes" onward, whitespace-normalized.

Usage: uv run python projects/omni-pipeline-probe/make_inputs.py [--download]
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly

SOURCE_URL = ("https://www.archive.org/download/art_of_war_librivox/"
              "art_of_war_01-02_sun_tzu_64kb.mp3")
SOURCE = {
    "librivox_book_id": 119, "librivox_url": "https://librivox.org/the-art-of-war-by-sun-tzu/",
    "archive_item": "https://archive.org/details/art_of_war_librivox",
    "url": SOURCE_URL, "title": "The Art of War, section 1: 01 Laying Plans / 02 Waging War",
    "author": "Sun Tzu", "translator": "Lionel Giles", "narrator": "Moira Fogarty",
    "license": "Public domain in the US (LibriVox); archive.org licenseurl creativecommons.org/licenses/publicdomain/",
    "bytes": 4055167,
    "sha256": "6cb289fa9a13cfc94b18e8a89159da687c60db88c2b6377311f9bc5664d5492f",
    "archive_md5": "c788727db63e9eeddaf8c26912c9bb62",
    "archive_sha1": "9b9e7c673fe27d50c703f92efcabeb695d9722f7",
}
TEXT_SOURCE = {
    "url": "https://www.gutenberg.org/cache/epub/132/pg132.txt",
    "title": "The Art of War (Project Gutenberg eBook #132), tr. Lionel Giles",
    "license": "Public domain in the US (Project Gutenberg)",
    "bytes": 342105,
    "sha256": "2301ae1ce20875381a316b2fd0ac5ada00f71bacdc732458a11b3674b358e360",
    "start_marker": "5, 6. _The Moral Law_ causes",
    "words": 600,
}
ROOT = Path("results/omni-pipeline-probe/inputs")
TEXT_SRC = ROOT / "text" / "pg132.txt"
PASSAGE = ROOT / "text" / "passage.txt"
MP3 = ROOT / "source" / "art_of_war_01-02_sun_tzu_64kb.mp3"
WAV_DIR = ROOT / "wav"
MANIFEST = Path(__file__).with_name("inputs_manifest.json")
TARGET_SR = 16000
# Inside the 63.85-65.95 s pause found by a 50 ms RMS scan, well after the
# LibriVox preamble, so every clip starts between words.
CLIP_START_S = 65.0
DURATIONS_S = (10, 30, 90)
VARIANTS = 8
VARIANT_BASE_INDEX = 8000  # variant k changes sample VARIANT_BASE_INDEX + k (k >= 1)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decoded_sha256(pcm: np.ndarray) -> str:
    """Hash of the int16 sample sequence (little-endian), independent of the WAV header."""
    return sha256_bytes(pcm.astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="fetch the MP3 if missing")
    args = parser.parse_args()
    if not MP3.is_file():
        if not args.download:
            sys.exit(f"{MP3} missing; rerun with --download")
        MP3.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, MP3)
    data = MP3.read_bytes()
    if len(data) != SOURCE["bytes"] or sha256_bytes(data) != SOURCE["sha256"]:
        sys.exit("source MP3 does not match the recorded size/SHA-256")

    if not TEXT_SRC.is_file():
        if not args.download:
            sys.exit(f"{TEXT_SRC} missing; rerun with --download")
        TEXT_SRC.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TEXT_SOURCE["url"], TEXT_SRC)
    raw = TEXT_SRC.read_bytes()
    if len(raw) != TEXT_SOURCE["bytes"] or sha256_bytes(raw) != TEXT_SOURCE["sha256"]:
        sys.exit("text source does not match the recorded size/SHA-256")
    book = raw.decode("utf-8")
    start_at = book.index(TEXT_SOURCE["start_marker"])
    words = book[start_at:].split()[:TEXT_SOURCE["words"]]
    PASSAGE.write_text(" ".join(words) + "\n")
    passage = {"file": str(PASSAGE), "words": len(words),
               "sha256": sha256_bytes(PASSAGE.read_bytes())}

    audio, sr = sf.read(MP3, dtype="float64", always_2d=False)
    if audio.ndim != 1:
        sys.exit(f"expected mono source, got shape {audio.shape}")
    ratio = Fraction(TARGET_SR, sr)
    resampled = resample_poly(audio, ratio.numerator, ratio.denominator)
    pcm_all = np.clip(np.round(resampled * 32767.0), -32768, 32767).astype(np.int16)
    start = int(CLIP_START_S * TARGET_SR)
    longest = start + max(DURATIONS_S) * TARGET_SR
    if longest > len(pcm_all):
        sys.exit("source too short for the longest clip")

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    clips, seen = [], set()
    for duration in DURATIONS_S:
        base = pcm_all[start:start + duration * TARGET_SR].copy()
        for k in range(VARIANTS):
            pcm = base.copy()
            changed = None
            if k:
                changed = VARIANT_BASE_INDEX + k
                pcm[changed] += 1 if pcm[changed] < 32767 else -1
            path = WAV_DIR / f"aow_{duration:03d}s_v{k:02d}.wav"
            sf.write(path, pcm, TARGET_SR, subtype="PCM_16")
            back, back_sr = sf.read(path, dtype="int16")
            if back_sr != TARGET_SR or not np.array_equal(back, pcm):
                sys.exit(f"read-back mismatch for {path}")
            digest = decoded_sha256(back)
            if digest in seen:
                sys.exit(f"duplicate decoded samples for {path}")
            seen.add(digest)
            clips.append({
                "file": str(path), "duration_s": duration, "variant": k,
                "num_samples": int(len(pcm)), "sample_rate": TARGET_SR,
                "source_offset_s": CLIP_START_S, "source_end_s": CLIP_START_S + duration,
                "changed_sample_index": changed, "file_sha256": sha256_bytes(path.read_bytes()),
                "decoded_int16_sha256": digest,
            })

    manifest = {
        "schema_version": 1, "source": SOURCE, "text_source": TEXT_SOURCE,
        "text_passage": passage,
        "processing": {
            "decoder": f"soundfile {sf.__version__} / libsndfile {sf.__libsndfile_version__}",
            "source_sample_rate": sr, "resample": f"scipy {scipy.__version__} resample_poly"
            f" up={ratio.numerator} down={ratio.denominator}", "numpy": np.__version__,
            "quantization": "round(x * 32767), clip to int16, PCM_16 WAV, mono",
            "clip_start_s": CLIP_START_S, "nesting": "each shorter clip is a prefix of the longer",
            "variant_rule": f"variant k>=1 adds +1 LSB (or -1 at +32767) to sample {VARIANT_BASE_INDEX}+k",
            "note": "Re-decoding with another MP3 decoder may change samples; compare decoded_int16_sha256.",
        },
        "clips": clips,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(clips)} WAVs to {WAV_DIR} and {MANIFEST}")


if __name__ == "__main__":
    main()
