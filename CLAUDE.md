# worship-edit 작업 메모

- 사용자: Splendor (한국어로 소통). 주일 예배 찬양 실황 편집 자동화 앱.
- 엔진: `worship_edit/` (fcpxml, analyze, separate, align, autoalign, transcribe, library, media, pipeline, cuts, premiere, cli). 앱 화면: `gui.py` (PySide6), 진입점 `app_main.py`.
- 앱 빌드: `.github/workflows/build-macos.yml` (macos-14, PyInstaller) → Releases `latest`에 zip. 빌드 뒤 `--selftest`로 모듈·ffmpeg·Whisper 점검. 화면 점검은 컨테이너에서 `QT_QPA_PLATFORM=offscreen`.
- 입력은 XML 하나: 오디오는 `media.audio_source`가 XML의 원본 녹화본 경로에서 꺼낸다. 결과 확인은 `python3 tests/verify_xml.py 결과.fcpxml`.
- 가사(저작권)는 절대 저장소에 커밋하지 않는다. 테스트용 가사·오디오·XML도 커밋 금지(.gitignore).
- 규칙·스타일 값이 바뀌면 README의 규칙 표와 `cuts.py` 상단 주석, `fcpxml.TITLE_STYLE`을 함께 고친다.
- 구조 자동 감지(analyze.detect_structure)는 아직 추정치: 조용한 피아노 인트로·멘트 위 패드·약한 드럼 카운트에서 틀릴 수 있어 앱에서 사용자가 검토·수정하게 한다.
- 가사 자동 정렬(autoalign) 평가: 2026-11-02 기준 세 주(11/1, 10/25, 10/11) 수동 위치 대비 ±1초 일치 66~88%. 평가용 데이터(오디오·가사·수동 위치)는 사용자 Mac에만 있고 저장소에는 없다.
- Whisper(whisper.cpp)는 노래에서 단어 타임스탬프가 부정확하다. 구간 단위 결과 + 목소리 소절 시작점으로 위치를 잡는다.
