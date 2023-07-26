# 제5고지

## 52단계. GPU 지원
- cupy: GPU를 사용해 병렬 계산을 해주는 라이브러리
- `pip install cupy` 
- `cp.get_array_module` : 주어진 데이터에 적합한 모듈을 돌려줌
- Variable class
    - 데이터가 cupy.ndarray 가 넘어와도 대응할 수 있도록 수정
    - 기울기를 자동으로 보완하는 부분 수정
    - 데이터를 gpu 혹은 cpu로 전송해주는 기능 추가

