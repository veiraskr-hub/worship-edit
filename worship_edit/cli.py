"""명령어 버전.

  worship-edit check  <프로젝트.fcpxmld> <오디오.mp3>
  worship-edit separate <오디오.mp3> <목소리.wav>
  worship-edit build <프로젝트.fcpxmld> <오디오.mp3> --vocals 목소리.wav --lyrics 가사.txt -o 결과.fcpxml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from . import fcpxml
from .align import VocalTrack, display_spans, read_anchors, snap_lines
from .analyze import Audio, Structure, detect_structure
from .cuts import CutSettings, make_cuts


def audio_length(path: str) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def check(project: str, audio: str) -> dict:
    prj = fcpxml.load(project)
    roles = fcpxml.classify_roles(prj.angles)
    al = audio_length(audio)
    return {
        "project_length": round(prj.length, 2),
        "audio_length": round(al, 2),
        "length_ok": abs(al - prj.length) < 1.0,
        "roles": {n: fcpxml.ROLE_KO.get(r, r) for n, r in roles.items()},
        "unknown": [n for n, r in roles.items() if r == "unknown"],
    }


def build(project: str, audio: str, vocals: str, lyrics: str | None, out: str, name: str | None = None,
          structure: str | None = None, roles_json: str | None = None) -> dict:
    prj = fcpxml.load(project)
    roles = json.loads(roles_json) if roles_json else fcpxml.classify_roles(prj.angles)
    roles = {n: r for n, r in roles.items() if r in fcpxml.ROLES}
    au = Audio(audio)
    st = Structure.from_dict(json.load(open(structure))) if structure else detect_structure(au)
    report = {"structure": st.to_dict()}
    lines = []
    if lyrics:
        vt = VocalTrack(vocals)
        lines = snap_lines(vt, read_anchors(lyrics))
        st.chorus = [ln.time for ln in lines if ln.chorus]
        report["lyrics"] = {"lines": len(lines), "voice_start_ok": vt.starts_ok([l.time for l in lines])}
    cuts, rep = make_cuts(prj, roles, au, st, CutSettings())
    report["cuts"] = rep
    text = fcpxml.write_cuts(prj, [(c.start, c.end, c.angle) for c in cuts])
    if lines:
        text = fcpxml.write_titles(prj, text, display_spans(lines))
    text = fcpxml.rename(text, name or out.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    fcpxml.save(text, out)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="worship-edit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("project"); c.add_argument("audio")
    s = sub.add_parser("separate"); s.add_argument("audio"); s.add_argument("out")
    a = sub.add_parser("analyze"); a.add_argument("audio")
    b = sub.add_parser("build")
    b.add_argument("project"); b.add_argument("audio")
    b.add_argument("--vocals", required=True); b.add_argument("--lyrics")
    b.add_argument("--structure"); b.add_argument("--roles"); b.add_argument("--name")
    b.add_argument("-o", "--out", required=True)
    ns = ap.parse_args(argv)
    if ns.cmd == "check":
        print(json.dumps(check(ns.project, ns.audio), ensure_ascii=False, indent=2))
    elif ns.cmd == "separate":
        from .separate import separate_vocals
        separate_vocals(ns.audio, ns.out, progress=lambda f, t: print(f"\r{f*100:5.1f}% {t:5.0f}s", end="", file=sys.stderr))
        print(file=sys.stderr)
    elif ns.cmd == "analyze":
        print(json.dumps(detect_structure(Audio(ns.audio)).to_dict(), ensure_ascii=False, indent=2))
    elif ns.cmd == "build":
        rep = build(ns.project, ns.audio, ns.vocals, ns.lyrics, ns.out, ns.name, ns.structure, ns.roles)
        print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
