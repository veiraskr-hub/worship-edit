# worship-edit 작업 메모

- 사용자: Splendor (한국어로 소통). 주일 예배 찬양 실황 편집 자동화 앱.
- 엔진: `worship_edit/` (fcpxml, analyze, separate, align, autoalign, transcribe, library, media, pipeline, cuts, premiere(출력), premiere_in(프리미어 멀티캠 XML 입력), cli). 앱 화면: `gui.py` (PySide6), 진입점 `app_main.py`.
- 앱 빌드: `.github/workflows/build-macos.yml` (macos-15, PyInstaller) → Releases `latest`에 zip. 빌드 뒤 `--selftest`로 모듈·ffmpeg·Whisper 점검. 화면 점검은 컨테이너에서 `QT_QPA_PLATFORM=offscreen`.
- 입력은 XML 하나: 오디오는 `media.audio_source`가 XML의 원본 녹화본 경로에서 꺼낸다. 결과 확인은 `python3 tests/verify_xml.py 결과.fcpxml`.
- 가사(저작권)는 절대 저장소에 커밋하지 않는다. 테스트용 가사·오디오·XML도 커밋 금지(.gitignore).
- 규칙·스타일 값이 바뀌면 README의 규칙 표와 `cuts.py` 상단 주석, `fcpxml.TITLE_STYLE`을 함께 고친다.
- 구조 자동 감지(analyze.detect_structure)는 아직 추정치: 조용한 피아노 인트로·멘트 위 패드·약한 드럼 카운트에서 틀릴 수 있어 앱에서 사용자가 검토·수정하게 한다.
- 가사 자동 정렬(autoalign) 평가: 2026-11-02 기준 세 주(11/1, 10/25, 10/11) 수동 위치 대비 ±1초 일치 66~88%. 평가용 데이터(오디오·가사·수동 위치)는 사용자 Mac에만 있고 저장소에는 없다.
- 가사 자막 기본은 실황 녹화본 하단 가사 슬라이드 OCR(`slides.py`, 맥 Vision / tesseract). 11/8 평가: 슬라이드 경계·순서 수동 판독과 일치, 부르지 않은 슬라이드 1개 정확히 제외. 실패 시 받아쓰기 정렬(autoalign)로.
- 11/15: 자막 뒤에 흰 건반·아이패드·선풍기가 비쳐 OCR 오타(맥 Vision 줄 정확도 56%), 첫 줄만 같은 슬라이드 합쳐짐, 곡 중간 슬라이드 9장 '안 부름'으로 빠짐 → 배경 지우기·두 줄 비교·곡 중간 유지로 고침(슬라이드 시작 70/71 일치). 줄 글자 정확도는 맥 Vision으로 다시 재야 함(컨테이너 tesseract는 51%).
- Whisper(whisper.cpp)는 노래에서 단어 타임스탬프가 부정확하다. 구간 단위 결과 + 목소리 소절 시작점으로 위치를 잡는다.
