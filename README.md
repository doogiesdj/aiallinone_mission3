# Mini NPU Simulator

NumPy·pandas 같은 외부 라이브러리 없이 파이썬 표준 라이브러리 `json`, `time`만으로 **MAC(Multiply-Accumulate) 연산을 이중 반복문으로 직접 구현**한 패턴 판별 시뮬레이터입니다. 입력 패턴이 **Cross(+)** 인지 **X** 인지를 두 필터와의 MAC 점수로 판별합니다.

---

## 1. 프로젝트 개요

NPU의 핵심 연산 단위인 `Accumulator ← Accumulator + (A × B)`(`out += a * b`)를 코드 수준에서 재현합니다.

- **동적 N×N 처리**: 행렬 크기를 하드코딩하지 않고 `3×3`, `5×5`, `13×13`, `25×25`를 같은 로직으로 처리
- **Epsilon 동점 판정**: IEEE 754 부동소수점 오차를 고려해 점수 차이가 `EPSILON = 1e-9` 미만이면 `UNDECIDED`
- **3갈래 실패 진단**: 데이터/스키마 오류, 로직 오류, 수치 비교 오류를 사용자 정의 예외로 분리
- **성능 측정**: I/O를 제외한 순수 MAC 연산 시간만 측정해 $O(N^2)$ 복잡도 확인

---

## 2. 시작하기

### 요구 사항
- **Python 3.10 이상** (`int | None` 타입 표기 사용)
- 외부 패키지 설치 불필요 — 표준 라이브러리 `json`, `time`만 사용

### 실행
`main.py`와 `data.json`을 같은 폴더에 두고 실행합니다.

```bash
python main.py
```

Windows 콘솔에서 한글이 깨지면 UTF-8 출력을 지정합니다.

```bash
set PYTHONIOENCODING=utf-8      # cmd
$env:PYTHONIOENCODING="utf-8"   # PowerShell
```

### 메뉴

| 번호 | 기능 | 설명 |
| :---: | :--- | :--- |
| 1 | 사용자 입력 모드 | `3×3` Cross 필터, X 필터, 판별 대상 패턴을 직접 입력해 판별 |
| 2 | `data.json` 일괄 판정 모드 | JSON의 모든 케이스를 평가하고 케이스별 `PASS`/`FAIL`과 실패 사유 출력 |
| 3 | 성능 분석 | 크기별 순수 MAC 연산을 10회 반복한 평균 시간(ms)과 연산 횟수(N²) 출력 |
| 0 | 종료 | |

---

## 3. 사용 예시

### 모드 1 — 사용자 입력

각 행을 공백 또는 쉼표로 구분해 입력합니다. 열 개수가 맞지 않거나 숫자가 아니면 안내 후 해당 행을 다시 입력받습니다.

```text
[필터 1 (Cross 필터)] 3×3 행렬을 입력하세요
  1행 입력 > 0 1 0
  2행 입력 > 1 1 1
  3행 입력 > 0 1 0
...
--- 판별 결과 ---
Cross 필터 MAC 점수 : 5
X     필터 MAC 점수 : 1
점수 차이 (|Δ|)     : 4 (기준 epsilon = 1e-09)
최종 판별 결과      : Cross
```

### 모드 2 — `data.json` 일괄 판정

```text
[01] case_01_3x3_cross   | PASS | 3×3   | 판별=Cross     (Cross=5, X=1)
[04] case_04_13x13_cross | PASS | 13×13 | 판별=Cross     (Cross=25, X=1)
[08] case_08_3x3_tie_fp  | PASS | 3×3   | 판별=UNDECIDED (Cross=0.3, X=0.3)
[09] case_09_err_schema  | FAIL | [데이터/스키마 오류] 크기 불일치: 입력=2×2, Cross필터=5×5, X필터=5×5
[10] case_10_err_logic   | FAIL | [로직 오류] 패턴 판별 불일치 (기대=X, 판별=Cross, Cross=9, X=1)
[11] case_11_err_epsilon | FAIL | [수치 비교 오류] epsilon 동점 경계 불일치 (기대=Cross, 판별=UNDECIDED, ...)
------------------------------------------------------------
평가 요약: 총 11건 | 통과(PASS) 8건 | 실패(FAIL) 3건
```

한 케이스에서 오류가 나도 프로그램은 멈추지 않고 해당 케이스만 `FAIL`로 표시한 뒤 다음 케이스로 넘어갑니다.

---

## 4. `data.json` 형식

루트는 케이스 리스트이거나 `{"filters": ..., "cases": [...]}` 형태의 객체입니다. 키 이름은 아래처럼 여러 별칭을 허용합니다.

| 항목 | 허용 키 | 비고 |
| :--- | :--- | :--- |
| 케이스 목록 | `cases`, `test_cases`, `tests`, `data` | 루트가 객체일 때 |
| 케이스 이름 | `id`, `name` | 없으면 `Case-번호` |
| 입력 패턴 | `input`, `pattern`, `matrix`, `data` | N×N 숫자 행렬 |
| 기대 정답 | `expected`, `label`, `answer`, `target` | |
| 필터 | 케이스의 `filters`/`filter`, 또는 `filter_cross` + `filter_x`, 없으면 루트 `filters` | 루트 `filters`는 크기별 매핑(`{"5": {...}}`)도 지원 |
| 기대 점수 (선택) | `expected_scores` | `{"cross": 9, "x": 1}` 형태, epsilon 기준 비교 |

**라벨 정규화**: `'+'`, `'cross'` → `Cross`, `'x'` → `X`, `'undecided'` → `UNDECIDED` (대소문자·앞뒤 공백 무관)

---

## 5. 구현 원칙

1. **MAC 직접 구현 (`compute_mac`)**: 누적 변수를 `acc = 0.0`으로 초기화한 뒤 이중 `for`문으로 `acc += row_a[j] * row_b[j]`를 수행합니다.
2. **입력 검증 (`parse_finite_float`, `validate_square_matrix`)**: 정사각 행렬 여부와 숫자 변환을 검사하며, `bool`, `NaN`, `Inf`는 거부합니다.
3. **Epsilon 비교 (`classify_pattern`)**: `abs(score_cross - score_x) < 1e-9`이면 `UNDECIDED`, 아니면 점수가 큰 쪽으로 판별합니다.
4. **실패 원인 분리**: 예외 계층 `SimulatorError` 아래 세 가지로 구분합니다.

   | 예외 | 분류 | 발생 조건 |
   | :--- | :--- | :--- |
   | `DataSchemaError` | 데이터/스키마 오류 | 파일·키 누락, 크기 불일치, 숫자 파싱 실패 |
   | `LogicError` | 로직 오류 | 판별 라벨이 기대 라벨과 다름 |
   | `NumericalComparisonError` | 수치 비교 오류 | `UNDECIDED` 경계 불일치, 기대 점수 오차 초과 |

5. **안전한 파일 처리 (`load_json_file`)**: `try-except-else-finally`로 예외와 무관하게 파일을 닫습니다.
6. **전각 문자 정렬 (`display_width`, `pad_display`)**: 한글·한자 등은 콘솔에서 2칸을 차지하므로, 유니코드 범위로 표시 폭을 계산해 케이스 이름에 한글이 섞여도 결과 표의 열을 맞춥니다. 표준 라이브러리 제약상 `unicodedata` 대신 범위 상수(`WIDE_CHAR_RANGES`)를 직접 사용합니다.

---

## 6. 성능 분석 ($O(N^2)$ 검증)

### 측정 경계
- **포함**: `compute_mac()` 호출 구간만 `time.perf_counter()`로 측정
- **제외**: JSON 읽기, 행렬 생성, 콘솔 출력 등 I/O·준비 작업
- **방식**: 크기별 10회 반복 후 평균

### 측정 결과 (Python 3.14.0, Windows 11)

| 행렬 크기 | 연산 횟수 (N²) | 평균 시간 (ms) | 시간 배율 | 연산량 배율 |
| :--- | :---: | :---: | :---: | :---: |
| 3 × 3 | 9 | 0.001130 | 1.00x | 1.00x |
| 5 × 5 | 25 | 0.001840 | 1.63x | 2.78x |
| 13 × 13 | 169 | 0.008400 | 7.43x | 18.78x |
| 25 × 25 | 625 | 0.026770 | 23.69x | 69.44x |

> 측정값은 실행 환경과 시점에 따라 달라집니다.

### 해석
- `compute_mac`은 N×N의 모든 칸을 한 번씩 방문하므로 연산 횟수가 정확히 N²에 비례하는 $O(N^2)$ 알고리즘입니다.
- 작은 크기에서는 시간 배율이 연산량 배율보다 낮습니다. 함수 호출, 크기 검증 같은 **고정 비용**이 전체 시간에서 차지하는 비중이 크기 때문입니다.
- 크기가 커질수록 고정 비용 비중이 줄어 `13×13 → 25×25` 구간에서는 연산량 증가(약 3.70배)와 시간 증가(약 3.19배)가 가까워집니다.

---

## 7. 테스트 데이터 구성

`data.json`의 11개 케이스는 정상 판별, 동점 처리, 3가지 실패 진단을 모두 확인하도록 구성되어 있습니다.

| 케이스 | 목적 | 결과 |
| :--- | :--- | :---: |
| `case_01` ~ `case_07` | `3×3`~`25×25` Cross/X 판별 및 라벨 정규화 | PASS |
| `case_08_3x3_tie_fp` | `0.1 + 0.2` vs `0.3` 부동소수점 동점 → `UNDECIDED` | PASS |
| `case_09_err_schema` | 입력 `2×2` vs 필터 `5×5` 크기 불일치 | FAIL (데이터/스키마) |
| `case_10_err_logic` | Cross 패턴에 기대값 `x` 지정 | FAIL (로직) |
| `case_11_err_epsilon` | 점수 차이 `5e-10 < 1e-9`인데 기대값 `Cross` | FAIL (수치 비교) |

FAIL 3건은 오류 진단 기능을 확인하기 위해 일부러 넣은 케이스입니다.

---

## 8. 디렉토리 구조

```text
.
├── main.py       # 시뮬레이터 본체 (메뉴, MAC 연산, 판정, 성능 측정)
├── data.json     # 일괄 판정용 테스트 케이스 11건
├── README.md     # 프로젝트 설명
└── .gitignore
```
