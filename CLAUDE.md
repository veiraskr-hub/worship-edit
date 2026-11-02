# worship-edit 작업 메모

- 사용자: Splendor (한국어로 소통). 주일 예배 찬양 실황 편집 자동화 앱.
- 엔진: `worship_edit/` (fcpxml, analyze, separate, align, cuts, cli). 결과 확인은 `python3 tests/verify_xml.py 결과.fcpxml`.
- 가사(저작권)는 절대 저장소에 커밋하지 않는다. 테스트용 가사·오디오·XML도 커밋 금지(.gitignore).
- 규칙·스타일 값이 바뀌면 README의 규칙 표와 `cuts.py` 상단 주석, `fcpxml.TITLE_STYLE`을 함께 고친다.
- 구조 자동 감지(analyze.detect_structure)는 아직 추정치: 조용한 피아노 인트로·멘트 위 패드·약한 드럼 카운트에서 틀릴 수 있어 앱에서 사용자가 검토·수정하게 한다.
