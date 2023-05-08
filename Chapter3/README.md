# 제 3고지

## 25단계. 계산 그래프 시각화(1)
- Graphviz 사용
- `brew install graphviz`
- DOT 언어 배우기
- `digraph g{ \ x \ y \ }` 구조
- 각 줄마다 노드에 대한 정보가 담김
- 변수를 원, 함수를 사각형으로 표현

## 26단계. 계산 그래프 시각화(2)
- dezero/utils.py에 계산 그래프 시각화 함수 구현
- `_dot_var` 함수: 보조 함수, `get_dot_graph` 함수 전용으로 사용할 예정
    - Variable 인스턴스를 건네면 인스턴스 내용을 DOT 언어로 작성된 문자열로 바꿔서 반환
- `_dot_func` : dezero 함수를 DOT 언어로 변환
    - 함수와 입력 변수의 관계, 함수와 출력 변수의 관계 또한 DOT 언어로 기술
- `get_dot_graph` : Variable 클래스의 backward 메서드와 거의 같으나, 미분값을 전파하는게 아니라 DOT 언어로 기술한 문자열을 txt에 추가
- generation 값으로 정렬하는 코드는 주석처리 (노드의 추적 순서는 중요하지 않음)
- `plot_dot_graph` : 이미지 변환까지 한번에 수행
    - `os.path.expanduser('~')`: 사용자의 홈 디렉토리를 뜻하는 ~를 절대 경로로 풀어줌
    - `to_file`에 저장할 이미지 파일의 이름 지정

- 이 챕터의 아쉬운 점: 페이지가 바뀔 때 indentation 표기에 대해 더 신경써줬으면..