"""명령어 버전.

  worship-edit check  <프로젝트.fcpxmld> <오디오.mp3>
  worship-edit separate <오디오.mp3> <목소리.wav>
  worship-edit build <프로젝트.fcpxmld> <오디오.mp3> --vocals 목소리.wav --lyrics 가사.txt -o 결과.fcpxml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import fcpxml
from .align import VocalTrack, display_spans, read_anchors, snap_lines
from .analyze import Audio, Structure, detect_structure
from .cuts import CutSettings, make_cuts


def audio_length(path: str) -> float:
    from .media import duration
    return duration(path)


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
    roles = {n: r for n, r in roles.items() if fcpxml.is_camera_role(r)}
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
    au = sub.add_parser("auto", help="처음부터 끝까지 자동 (오디오→분리→받아쓰기→가사 정렬→컷→타이틀)")
    au.add_argument("project"); au.add_argument("audio", nargs="?", default="", help="생략하면 XML이 가리키는 원본 녹화본에서 꺼냄")
    au.add_argument("--songs", required=True, help="부른 순서대로 곡 제목, 쉼표로 구분 (가사 보관함에 있어야 함)")
    au.add_argument("--library", help="가사 보관함 폴더 (기본: ~/Library/Application Support/WorshipEdit/lyrics)")
    au.add_argument("--vocals"); au.add_argument("--srt"); au.add_argument("--name")
    au.add_argument("-o", "--out", required=True)
    pr = sub.add_parser("premiere", help="결과 FCPXML → 프리미어용 XML + 가사 SRT")
    pr.add_argument("result"); pr.add_argument("-o", "--out")
    ly = sub.add_parser("lyrics", help="가사 보관함: 목록 보기 / 곡 추가")
    ly.add_argument("action", choices=["list", "add"]); ly.add_argument("file", nargs="?")
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
    elif ns.cmd == "auto":
        from .library import LIB_DIR, find, load_library
        from .pipeline import Job, run
        lib = load_library(ns.library or LIB_DIR)
        songs = []
        for t in [x.strip() for x in ns.songs.split(",") if x.strip()]:
            s = find(lib, t)
            if not s:
                sys.exit(f"가사 보관함에 '{t}'이(가) 없어요. `worship-edit lyrics add 가사파일.txt`로 먼저 넣어 주세요.")
            songs.append(s)
        job = Job(ns.project, songs, ns.out, audio=ns.audio, name=ns.name or "", vocals=ns.vocals or "", srt=ns.srt or "")
        res = run(job, progress=lambda st, f, m: print(f"\r[{st}] {'' if f is None else f'{f*100:5.1f}%'} {m}", end="", file=sys.stderr))
        print(file=sys.stderr)
        print(json.dumps(res.report, ensure_ascii=False, indent=2))
    elif ns.cmd == "premiere":
        from .premiere import export
        base = os.path.splitext(ns.out or ns.result.rstrip("/"))[0]
        rep = export(ns.result, base + "_프리미어.xml", base + "_가사.srt")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    elif ns.cmd == "lyrics":
        from .library import LIB_DIR, load_library, parse, save_song
        if ns.action == "list":
            for k, s in load_library().items():
                print(f"{s.title}  ({sum(len(v) for v in s.sections.values())}줄, 진행: {' '.join(s.order)})")
        else:
            s = parse(open(ns.file, encoding="utf-8").read())
            print("저장:", save_song(s))


if __name__ == "__main__":
    main()
