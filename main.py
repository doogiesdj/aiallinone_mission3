"""
Mini NPU Simulator (미션 3)
===========================
외부 벡터화·수치 계산 라이브러리를 일절 사용하지 않고,
파이썬 표준 라이브러리(json, time)만을 활용하여 다음을 구현합니다.

1. 동적 N×N 행렬 구조 (크기 하드코딩 금지: 3×3, 5×5, 13×13, 25×25 모두 동일 로직 처리)
2. 모드 1: 사용자 입력 모드 (3×3 필터 2개 + 입력 패턴 검증 및 오류 시 재입력)
3. 라벨 정규화 함수 분리 ('+' / 'cross' -> 'Cross', 'x' -> 'X')
4. 반복문 기반 MAC(Multiply-Accumulate) 연산 직접 구현 (out += a * b)
5. 부동소수점 허용오차(epsilon = 1e-9) 기반 비교 및 동점(UNDECIDED) 판정
6. 모드 2: data.json 일괄 판정 모드 (케이스 단위 예외 처리로 중단 방지 및 3갈래 실패 진단)
7. 성능 분석: 입출력(I/O) 시간을 제외한 순수 MAC 연산만 크기별 10회 반복 측정(ms) 및 O(N²) 확인
"""

import json
import time

# ---------------------------------------------------------------------------
# 전역 상수 정의
# ---------------------------------------------------------------------------
# IEEE 754 부동소수점 연산은 이진수 근사 표현으로 미세한 오차가 누적될 수 있으므로,
# '==' 직접 비교 대신 허용오차(epsilon) 이내의 차이를 동등/동점으로 판정합니다.
EPSILON = 1e-9

# 성능 분석 시 크기별 반복 측정 횟수 및 기본 벤치마크 대상 크기
BENCHMARK_REPEATS = 10
BENCHMARK_SIZES = (3, 5, 13, 25)

# 기본 JSON 파일 경로
DEFAULT_DATA_FILE = "data.json"

# 모드 2 출력에서 크기 열(예: '25×25')의 최소 표시 너비
SIZE_LABEL_WIDTH = 5


# ---------------------------------------------------------------------------
# 사용자 정의 예외 계층 (실패 원인을 3갈래로 분리 진단)
# ---------------------------------------------------------------------------
class SimulatorError(Exception):
    """Mini NPU Simulator에서 발생하는 모든 도메인 예외의 기본 클래스."""

    category = "일반 오류"


class DataSchemaError(SimulatorError):
    """1) 데이터/스키마 오류: 파일 형식, 필수 키 누락, 행렬 크기 불일치, 숫자 변환 실패 등."""

    category = "데이터/스키마 오류"


class LogicError(SimulatorError):
    """2) 로직 오류: 예측된 패턴 라벨이 기대 라벨(expected)과 일치하지 않는 경우 등."""

    category = "로직 오류"


class NumericalComparisonError(SimulatorError):
    """3) 수치 비교 오류: epsilon 경계(UNDECIDED) 불일치 또는 수치적 비교 검증 실패."""

    category = "수치 비교 오류"


# ---------------------------------------------------------------------------
# 유틸리티: 수치 검증 및 라벨 정규화
# ---------------------------------------------------------------------------
def parse_finite_float(value) -> float:
    """
    입력값을 유한한 실수(float)로 변환(파싱)합니다.
    - 불리언(bool)은 파이썬에서 int의 하위 타입이므로 숫자로 오인하지 않도록 먼저 차단합니다.
    - NaN, Inf, -Inf는 수치 연산 및 epsilon 비교를 왜곡하므로 거부합니다.
    """
    if isinstance(value, bool):
        raise DataSchemaError(f"불리언 값({value!r})은 숫자로 사용할 수 없습니다.")

    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise DataSchemaError(f"숫자 파싱 실패: {value!r}을(를) 실수로 변환할 수 없습니다.") from exc

    # 표준 라이브러리(json, time) 외 모듈을 쓰지 않고 NaN / Inf 여부 검사
    if num != num or num == float("inf") or num == float("-inf"):
        raise DataSchemaError(f"유효하지 않은 부동소수점 값입니다: {value!r}")

    return num


# ---------------------------------------------------------------------------
# 유틸리티: 콘솔 표시 폭 계산 (한글·한자 등 전각 문자는 2칸 차지)
# ---------------------------------------------------------------------------
# 터미널에서 2칸으로 표시되는 대표적인 전각(Wide/Fullwidth) 유니코드 범위
WIDE_CHAR_RANGES = (
    (0x1100, 0x115F),    # 한글 자모 (초성)
    (0x2E80, 0x303E),    # CJK 부수, 기호 및 구두점
    (0x3041, 0x33FF),    # 히라가나, 가타카나, 한글 호환 자모 등
    (0x3400, 0x4DBF),    # CJK 통합 한자 확장 A
    (0x4E00, 0x9FFF),    # CJK 통합 한자
    (0xA000, 0xA4CF),    # 이 문자
    (0xAC00, 0xD7A3),    # 한글 완성형 음절
    (0xF900, 0xFAFF),    # CJK 호환 한자
    (0xFE30, 0xFE4F),    # CJK 호환 형태
    (0xFF00, 0xFF60),    # 전각 ASCII
    (0xFFE0, 0xFFE6),    # 전각 기호
    (0x20000, 0x3FFFD),  # CJK 통합 한자 확장 B 이후
)


def display_width(text: str) -> int:
    """문자열이 콘솔에서 차지하는 칸 수를 계산합니다 (전각 문자 2칸, 그 외 1칸)."""
    width = 0
    for ch in text:
        code = ord(ch)
        is_wide = any(start <= code <= end for start, end in WIDE_CHAR_RANGES)
        width += 2 if is_wide else 1
    return width


def pad_display(text, width: int) -> str:
    """화면 표시 폭 기준으로 오른쪽에 공백을 채워 왼쪽 정렬합니다 (str.ljust의 전각 대응판)."""
    text = str(text)
    return text + " " * max(width - display_width(text), 0)


def get_first_present(mapping: dict, keys: tuple[str, ...], default=None):
    """
    keys 순서대로 mapping에 실제로 존재하는 첫 번째 키의 값을 반환합니다.
    'a.get(k1) or a.get(k2)' 방식은 빈 리스트·0·빈 문자열을 '키 없음'으로 오인하므로,
    값의 참/거짓이 아니라 키 존재 여부로 판단합니다. 단, JSON null(None)은 없는 것으로 봅니다.
    """
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def normalize_label(label) -> str:
    """
    패턴 및 필터 라벨을 표준 형식으로 정규화하는 전용 함수입니다.
    - '+', 'cross' (대소문자 및 앞뒤 공백 무관) -> 'Cross'
    - 'x' -> 'X'
    - 'undecided' -> 'UNDECIDED'
    규격에 없는 라벨이면 DataSchemaError를 발생시킵니다.
    """
    if not isinstance(label, str):
        raise DataSchemaError(f"라벨은 문자열이어야 합니다: {label!r}")

    cleaned = label.strip().lower()
    if cleaned in ("+", "cross"):
        return "Cross"
    if cleaned == "x":
        return "X"
    if cleaned == "undecided":
        return "UNDECIDED"

    raise DataSchemaError(f"지원하지 않는 라벨입니다: {label!r} (허용: '+', 'cross', 'x', 'UNDECIDED')")


# ---------------------------------------------------------------------------
# 행렬 검증 및 MAC(Multiply-Accumulate) 핵심 연산
# ---------------------------------------------------------------------------
def validate_square_matrix(matrix, name: str = "행렬", expected_size: int | None = None) -> list[list[float]]:
    """
    2차원 정사각 행렬(N×N)인지 검증하고, 모든 원소를 float으로 정규화해 반환합니다.
    행렬 크기를 특정 숫자(3, 5 등)로 고정하지 않고 실제 입력 길이를 기준으로 동적 검증합니다.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise DataSchemaError(f"{name}은(는) 비어 있지 않은 2차원 리스트여야 합니다.")

    row_count = len(matrix)
    if expected_size is not None and row_count != expected_size:
        raise DataSchemaError(
            f"{name}의 행 개수가 일치하지 않습니다: 기대={expected_size}, 실제={row_count}"
        )

    normalized: list[list[float]] = []
    for row_idx, row in enumerate(matrix):
        if not isinstance(row, list):
            raise DataSchemaError(f"{name}의 {row_idx + 1}번째 행이 리스트가 아닙니다.")
        if len(row) != row_count:
            raise DataSchemaError(
                f"{name}의 {row_idx + 1}번째 행 열 개수가 정사각 크기({row_count})와 다릅니다: 실제={len(row)}"
            )

        parsed_row: list[float] = []
        for col_idx, cell in enumerate(row):
            try:
                parsed_row.append(parse_finite_float(cell))
            except DataSchemaError as exc:
                raise DataSchemaError(
                    f"{name} ({row_idx + 1}행 {col_idx + 1}열) 오류: {exc}"
                ) from exc
        normalized.append(parsed_row)

    return normalized


def compute_mac(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> float:
    """
    하드웨어 NPU의 MAC(Multiply-Accumulate: Accumulator <- Accumulator + A * B) 연산을
    외부 라이브러리 없이 이중 반복문으로 직접 구현합니다.

    - 두 행렬은 동일한 N×N 정사각 크기여야 합니다.
    - 누적 변수(acc)는 '+=' 누적 전에 반드시 0.0으로 초기화합니다.
    """
    size_a = len(matrix_a)
    size_b = len(matrix_b)
    if size_a == 0 or size_a != size_b:
        raise DataSchemaError(
            f"MAC 연산 행렬 크기가 일치하지 않습니다: 필터={size_a}×{size_a}, 입력={size_b}×{size_b}"
        )

    # 행렬 내 각 행의 열 길이까지 정합성 확인 (직접 호출 시 안전장치)
    for idx in range(size_a):
        if len(matrix_a[idx]) != size_a or len(matrix_b[idx]) != size_a:
            raise DataSchemaError(
                f"MAC 연산 대상의 {idx + 1}번째 행 열 개수가 {size_a}와 일치하지 않습니다."
            )

    # 누적합 변수 초기화 (필수)
    acc = 0.0
    for i in range(size_a):
        row_a = matrix_a[i]
        row_b = matrix_b[i]
        for j in range(size_a):
            # 위치별 곱셈 결과를 누적기에 합산 (out += a * b)
            acc += row_a[j] * row_b[j]

    return acc


def classify_pattern(
    pattern_matrix: list[list[float]],
    cross_filter: list[list[float]],
    x_filter: list[list[float]],
    epsilon: float = EPSILON,
) -> tuple[str, float, float, float]:
    """
    입력 패턴과 두 필터(Cross, X)에 대해 각각 MAC 연산을 수행하고,
    부동소수점 허용오차(epsilon) 기준으로 패턴을 판별합니다.

    반환값: (판별_라벨, Cross_점수, X_점수, 점수_차이_절댓값)
    """
    score_cross = compute_mac(pattern_matrix, cross_filter)
    score_x = compute_mac(pattern_matrix, x_filter)

    # 부동소수점 정밀도 오차를 고려하여 두 점수 차이의 절댓값이 epsilon 미만이면 동점 처리
    diff = abs(score_cross - score_x)
    if diff < epsilon:
        predicted = "UNDECIDED"
    elif score_cross > score_x:
        predicted = "Cross"
    else:
        predicted = "X"

    return predicted, score_cross, score_x, diff


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 모드 (3×3 필터 2개 + 패턴)
# ---------------------------------------------------------------------------
def read_matrix_from_user(title: str, size: int = 3) -> list[list[float]]:
    """
    사용자로부터 size×size 크기의 행렬을 한 줄씩 입력받습니다.
    - 공백 또는 쉼표(,) 구분자를 모두 허용합니다.
    - 행/열 수 불일치나 숫자 파싱 실패가 발생하면 오류 원인을 안내하고 재입력을 받습니다.
    """
    print(f"\n[{title}] {size}×{size} 행렬을 입력하세요 (각 행의 숫자 {size}개를 공백 또는 쉼표로 구분):")

    while True:
        rows: list[list[float]] = []
        valid_matrix = True

        for r in range(size):
            while True:
                raw_line = input(f"  {r + 1}행 입력 > ").strip()
                if not raw_line:
                    print("  [입력 오류] 빈 줄은 입력할 수 없습니다. 다시 입력해 주세요.")
                    continue

                # 쉼표를 공백으로 치환해 공백/쉼표 혼합 입력도 깔끔하게 분리
                tokens = raw_line.replace(",", " ").split()
                if len(tokens) != size:
                    print(
                        f"  [입력 오류] 열 개수가 일치하지 않습니다 "
                        f"(기대: {size}개, 입력: {len(tokens)}개). {r + 1}행을 다시 입력해 주세요."
                    )
                    continue

                try:
                    parsed_row = [parse_finite_float(tok) for tok in tokens]
                except DataSchemaError as exc:
                    print(f"  [입력 오류] {exc} {r + 1}행을 다시 입력해 주세요.")
                    continue

                rows.append(parsed_row)
                break

        # 전체 행 수 최종 검증 (안전 확인)
        try:
            validated = validate_square_matrix(rows, name=title, expected_size=size)
        except DataSchemaError as exc:
            print(f"  [검증 실패] {exc} 행렬 전체를 다시 입력합니다.")
            valid_matrix = False

        if valid_matrix:
            return validated


def run_interactive_mode(default_size: int = 3) -> None:
    """
    모드 1 실행 함수:
    기본 3×3 크기의 Cross 필터, X 필터, 입력 패턴을 사용자로부터 받아 MAC 연산으로 판별합니다.
    """
    print("\n" + "=" * 60)
    print(f"모드 1: 사용자 입력 모드 ({default_size}×{default_size} 필터 2개 + 패턴)")
    print("=" * 60)

    cross_filter = read_matrix_from_user("필터 1 (Cross 필터)", size=default_size)
    x_filter = read_matrix_from_user("필터 2 (X 필터)", size=default_size)
    pattern = read_matrix_from_user("판별 대상 패턴", size=default_size)

    predicted, score_cross, score_x, diff = classify_pattern(
        pattern, cross_filter, x_filter, epsilon=EPSILON
    )

    print("\n--- 판별 결과 ---")
    print(f"Cross 필터 MAC 점수 : {score_cross:.12g}")
    print(f"X     필터 MAC 점수 : {score_x:.12g}")
    print(f"점수 차이 (|Δ|)     : {diff:.12g} (기준 epsilon = {EPSILON})")
    print(f"최종 판별 결과      : {predicted}")


# ---------------------------------------------------------------------------
# 모드 2: data.json 일괄 판정 모드 (유연한 스키마 해석 + 케이스 단위 예외 처리)
# ---------------------------------------------------------------------------
def load_json_file(file_path: str):
    """
    try-except-else-finally 구조를 사용해 JSON 파일을 안전하게 읽고 자원을 정리합니다.
    """
    file_obj = None
    try:
        file_obj = open(file_path, "r", encoding="utf-8")
        data = json.load(file_obj)
    except FileNotFoundError as exc:
        raise DataSchemaError(f"파일을 찾을 수 없습니다: '{file_path}'") from exc
    except json.JSONDecodeError as exc:
        raise DataSchemaError(
            f"JSON 구문 파싱 실패 ({exc.lineno}행 {exc.colno}열): {exc.msg}"
        ) from exc
    except OSError as exc:
        raise DataSchemaError(f"파일 읽기 중 OS 오류가 발생했습니다: {exc}") from exc
    else:
        return data
    finally:
        # 예외 발생 여부와 무관하게 열려 있는 파일 자원을 반드시 해제
        if file_obj is not None:
            file_obj.close()


def extract_filters_from_mapping(raw_filters, context_name: str) -> dict[str, list[list[float]]]:
    """
    필터 딕셔너리(또는 리스트)에서 라벨 정규화 함수를 거쳐 Cross / X 필터 행렬을 추출합니다.
    예: {"cross": [[...]], "x": [[...]]} 또는 {"+": [[...]], "x": [[...]]}
    """
    extracted: dict[str, list[list[float]]] = {}

    if isinstance(raw_filters, dict):
        for raw_key, raw_matrix in raw_filters.items():
            norm_key = normalize_label(raw_key)
            if norm_key not in ("Cross", "X"):
                raise DataSchemaError(
                    f"{context_name}: 필터 라벨은 Cross('+'/'cross') 또는 X('x')여야 합니다: {raw_key!r}"
                )
            extracted[norm_key] = validate_square_matrix(
                raw_matrix, name=f"{context_name} [{norm_key} 필터]"
            )
    elif isinstance(raw_filters, list):
        for idx, item in enumerate(raw_filters):
            if not isinstance(item, dict):
                raise DataSchemaError(
                    f"{context_name}: 필터 리스트의 {idx + 1}번째 항목이 객체가 아닙니다."
                )
            raw_label = get_first_present(item, ("label", "name", "type"))
            raw_matrix = get_first_present(item, ("matrix", "weights", "filter"))
            if raw_label is None or raw_matrix is None:
                raise DataSchemaError(
                    f"{context_name}: 필터 항목에 라벨 또는 행렬 필드가 누락되었습니다."
                )
            norm_key = normalize_label(raw_label)
            if norm_key not in ("Cross", "X"):
                raise DataSchemaError(f"{context_name}: 잘못된 필터 라벨입니다: {raw_label!r}")
            extracted[norm_key] = validate_square_matrix(
                raw_matrix, name=f"{context_name} [{norm_key} 필터]"
            )
    else:
        raise DataSchemaError(f"{context_name}: 필터 구조가 딕셔너리 또는 리스트가 아닙니다.")

    if "Cross" not in extracted or "X" not in extracted:
        raise DataSchemaError(f"{context_name}: Cross와 X 필터 2개가 모두 필요합니다.")

    size_cross = len(extracted["Cross"])
    size_x = len(extracted["X"])
    if size_cross != size_x:
        raise DataSchemaError(
            f"{context_name}: 두 필터의 크기가 다릅니다 (Cross={size_cross}×{size_cross}, X={size_x}×{size_x})."
        )

    return extracted


def resolve_case_components(
    case_item, global_filters
) -> tuple[str, list[list[float]], list[list[float]], list[list[float]], str, dict | None]:
    """
    단일 테스트 케이스에서 (식별자, 입력 패턴, Cross 필터, X 필터, 기대 라벨, 기대 점수)를 추출·검증합니다.
    누락된 키나 잘못된 크기/타입이 있으면 DataSchemaError를 발생시킵니다.
    """
    if not isinstance(case_item, dict):
        raise DataSchemaError("테스트 케이스 항목이 JSON 객체(dict) 형식이 아닙니다.")

    case_id = str(get_first_present(case_item, ("id", "name"), default="이름없음"))

    # 1. 입력 패턴 행렬 추출 (일반적인 키 이름 지원)
    raw_pattern = None
    for key in ("input", "pattern", "matrix", "data"):
        if key in case_item:
            raw_pattern = case_item[key]
            break
    if raw_pattern is None:
        raise DataSchemaError("입력 패턴 필드('input', 'pattern', 'matrix' 중 하나)가 누락되었습니다.")

    pattern_matrix = validate_square_matrix(raw_pattern, name=f"케이스({case_id}) 입력 패턴")
    pattern_size = len(pattern_matrix)

    # 2. 기대 라벨 추출 및 정규화 함수 호출
    raw_expected = None
    for key in ("expected", "label", "answer", "target"):
        if key in case_item:
            raw_expected = case_item[key]
            break
    if raw_expected is None:
        raise DataSchemaError("기대 정답 필드('expected', 'label', 'answer' 중 하나)가 누락되었습니다.")

    expected_label = normalize_label(raw_expected)

    # 3. 필터 2개 추출 (케이스 내부 filter/filters 우선, 없으면 전역 filters에서 크기 매칭)
    filters_dict = None
    if "filters" in case_item:
        filters_dict = extract_filters_from_mapping(case_item["filters"], f"케이스({case_id})")
    elif "filter" in case_item:
        filters_dict = extract_filters_from_mapping(case_item["filter"], f"케이스({case_id})")
    elif "filter_cross" in case_item and "filter_x" in case_item:
        filters_dict = {
            "Cross": validate_square_matrix(case_item["filter_cross"], name="filter_cross"),
            "X": validate_square_matrix(case_item["filter_x"], name="filter_x"),
        }
    elif global_filters is not None:
        # 전역 filters가 크기별 매핑({"5": {"cross": ..., "x": ...}}) 또는 단일 매핑({"cross": ..., "x": ...})인지 확인
        size_key = str(pattern_size)
        if isinstance(global_filters, dict) and size_key in global_filters:
            filters_dict = extract_filters_from_mapping(
                global_filters[size_key], f"전역 필터({size_key}×{size_key})"
            )
        else:
            filters_dict = extract_filters_from_mapping(global_filters, "전역 필터")

    if filters_dict is None:
        raise DataSchemaError("케이스 또는 전역 설정에 비교용 필터 2개('filters'/'filter')가 없습니다.")

    cross_filter = filters_dict["Cross"]
    x_filter = filters_dict["X"]

    # 입력 패턴과 필터 크기가 일치하는지 검사 (N×N 동적 일치 검증)
    if len(cross_filter) != pattern_size or len(x_filter) != pattern_size:
        raise DataSchemaError(
            f"크기 불일치: 입력={pattern_size}×{pattern_size}, "
            f"Cross필터={len(cross_filter)}×{len(cross_filter)}, "
            f"X필터={len(x_filter)}×{len(x_filter)}"
        )

    expected_scores = case_item.get("expected_scores")
    return case_id, pattern_matrix, cross_filter, x_filter, expected_label, expected_scores


def evaluate_single_case(case_item, global_filters) -> tuple[str, int, str, float, float]:
    """
    단일 케이스를 평가합니다.
    실패 시 3가지 예외(DataSchemaError / NumericalComparisonError / LogicError) 중 하나로 구분하여 발생시킵니다.
    성공 시 (케이스ID, 크기N, 예측라벨, Cross점수, X점수)를 반환합니다.
    """
    case_id, pattern, cross_filter, x_filter, expected_label, expected_scores = (
        resolve_case_components(case_item, global_filters)
    )
    size_n = len(pattern)

    predicted, score_cross, score_x, diff = classify_pattern(
        pattern, cross_filter, x_filter, epsilon=EPSILON
    )

    # 선택적으로 기대 점수(expected_scores)가 제공된 경우 epsilon 기반 수치 비교 검증
    if expected_scores is not None:
        if not isinstance(expected_scores, dict):
            raise DataSchemaError("expected_scores 필드는 딕셔너리여야 합니다.")
        for raw_k, raw_v in expected_scores.items():
            norm_k = normalize_label(raw_k)
            exp_val = parse_finite_float(raw_v)
            actual_val = score_cross if norm_k == "Cross" else score_x
            if abs(actual_val - exp_val) >= EPSILON:
                raise NumericalComparisonError(
                    f"{norm_k} 점수 수치 오차 초과 (기대={exp_val}, 실제={actual_val}, "
                    f"|Δ|={abs(actual_val - exp_val):.3e} >= {EPSILON})"
                )

    # 라벨 일치 여부 판정 및 실패 원인 분리
    if predicted != expected_label:
        # 동점(UNDECIDED)과 관련된 불일치는 epsilon 경계/수치 비교 오류로 분류
        if predicted == "UNDECIDED" or expected_label == "UNDECIDED":
            raise NumericalComparisonError(
                f"epsilon 동점 경계 불일치 (기대={expected_label}, 판별={predicted}, "
                f"Cross={score_cross:.6g}, X={score_x:.6g}, |Δ|={diff:.3e})"
            )
        # 그 외 오분류는 로직/분류 불일치로 분류
        raise LogicError(
            f"패턴 판별 불일치 (기대={expected_label}, 판별={predicted}, "
            f"Cross={score_cross:.6g}, X={score_x:.6g})"
        )

    return case_id, size_n, predicted, score_cross, score_x


def new_batch_summary() -> dict:
    """모드 2 평가 결과를 집계할 빈 요약 딕셔너리를 생성합니다."""
    return {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors_by_category": {
            DataSchemaError.category: 0,
            LogicError.category: 0,
            NumericalComparisonError.category: 0,
        },
        "measured_sizes": set(),
    }


def parse_batch_root(raw_root) -> tuple[list, object]:
    """
    JSON 루트 구조를 해석해 (케이스 리스트, 전역 필터)를 반환합니다.
    리스트 형태 또는 {"cases": [...], "filters": ...} 형태를 모두 지원합니다.
    """
    if isinstance(raw_root, list):
        return raw_root, None
    if not isinstance(raw_root, dict):
        raise DataSchemaError("JSON 최상위 구조는 리스트 또는 객체여야 합니다.")

    global_filters = get_first_present(raw_root, ("filters", "filter"))
    cases = get_first_present(raw_root, ("cases", "test_cases", "tests", "data"))
    if not isinstance(cases, list):
        raise DataSchemaError("JSON 루트 객체에 테스트 케이스 리스트('cases')가 없습니다.")
    return cases, global_filters


def build_display_ids(cases: list) -> list[str]:
    """각 케이스의 출력용 이름(id → name → 'Case-번호' 순)을 만듭니다."""
    return [
        str(get_first_present(item, ("id", "name"), default=f"Case-{idx}"))
        if isinstance(item, dict)
        else f"Case-{idx}"
        for idx, item in enumerate(cases, start=1)
    ]


def evaluate_and_report_case(
    idx: int, case_item, global_filters, display_id: str, id_width: int, summary: dict
) -> None:
    """
    단일 케이스를 평가해 PASS/FAIL 한 줄을 출력하고 summary 집계를 갱신합니다.
    케이스 단위로 예외를 격리하므로 실패해도 다음 케이스 평가가 계속됩니다.
    """
    try:
        case_id, size_n, predicted, score_cross, score_x = evaluate_single_case(
            case_item, global_filters
        )
    except SimulatorError as exc:
        summary["failed"] += 1
        summary["errors_by_category"][exc.category] = (
            summary["errors_by_category"].get(exc.category, 0) + 1
        )
        print(f"[{idx:02d}] {pad_display(display_id, id_width)} | FAIL | [{exc.category}] {exc}")
        return

    summary["passed"] += 1
    summary["measured_sizes"].add(size_n)
    size_label = f"{size_n}×{size_n}"
    print(
        f"[{idx:02d}] {pad_display(case_id, id_width)} | PASS | {size_label:<{SIZE_LABEL_WIDTH}} | "
        f"판별={predicted:<9} (Cross={score_cross:.4g}, X={score_x:.4g})"
    )


def print_batch_summary(summary: dict) -> None:
    """모드 2 평가 요약(총계, PASS/FAIL 수, 실패 원인 분류)을 출력합니다."""
    print("-" * 60)
    print(
        f"평가 요약: 총 {summary['total']}건 | "
        f"통과(PASS) {summary['passed']}건 | 실패(FAIL) {summary['failed']}건"
    )
    if summary["failed"] > 0:
        print("실패 원인 분류:")
        for cat_name, count in summary["errors_by_category"].items():
            print(f"  - {cat_name}: {count}건")


def run_batch_mode(file_path: str = DEFAULT_DATA_FILE) -> dict:
    """
    모드 2 실행 함수:
    data.json 파일의 모든 케이스를 일괄 판정합니다.
    개별 케이스에서 스키마 오류나 크기 불일치가 발생해도 프로그램이 중단되지 않고
    해당 케이스만 FAIL과 사유를 출력한 뒤 다음 케이스로 진행합니다.
    """
    print("\n" + "=" * 60)
    print(f"모드 2: data.json 일괄 판정 모드 (대상: {file_path})")
    print("=" * 60)

    summary = new_batch_summary()

    try:
        raw_root = load_json_file(file_path)
    except DataSchemaError as exc:
        print(f"[파일 오류] {exc}")
        return summary

    try:
        cases, global_filters = parse_batch_root(raw_root)
    except DataSchemaError as exc:
        print(f"[{exc.category}] {exc}")
        return summary

    summary["total"] = len(cases)
    display_ids = build_display_ids(cases)
    # 가장 긴 케이스 이름의 화면 폭에 맞춰 이름 열 너비를 결정 (한글은 2칸)
    id_width = max((display_width(label) for label in display_ids), default=0)

    for idx, case_item in enumerate(cases, start=1):
        evaluate_and_report_case(
            idx, case_item, global_filters, display_ids[idx - 1], id_width, summary
        )

    print_batch_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# 성능 분석: I/O를 제외한 순수 MAC 연산 구간만 크기별 10회 측정
# ---------------------------------------------------------------------------
def build_synthetic_matrices(size: int) -> tuple[list[list[float]], list[list[float]]]:
    """
    성능 측정용 N×N 입력 행렬과 필터 행렬을 사전에 생성합니다.
    (시간 측정 구간에 행렬 생성 비용이 섞이지 않도록 타이머 밖에서 호출합니다.)
    """
    pattern = [[float((i + j) % 2) for j in range(size)] for i in range(size)]
    cross_filter = [
        [1.0 if (i == size // 2 or j == size // 2) else 0.0 for j in range(size)]
        for i in range(size)
    ]
    return pattern, cross_filter


def benchmark_mac_performance(
    sizes: tuple[int, ...] = BENCHMARK_SIZES, repeats: int = BENCHMARK_REPEATS
) -> list[dict]:
    """
    크기별 순수 MAC 연산 시간을 repeats회 반복 측정해 평균 시간(ms)과 연산 횟수(N²)를 반환합니다.
    - 측정 경계: 행렬 생성과 콘솔 출력(I/O)은 타이머 밖으로 완전히 분리하고,
      오직 compute_mac() 호출 구간만 time.perf_counter()로 측정합니다.
    """
    results: list[dict] = []

    for size in sizes:
        # 타이머 시작 전 데이터 준비 (측정에서 제외)
        pattern, filt = build_synthetic_matrices(size)
        elapsed_ms_list: list[float] = []

        for _ in range(repeats):
            start_t = time.perf_counter()
            # 순수 MAC 연산만 수행
            _ = compute_mac(pattern, filt)
            end_t = time.perf_counter()
            elapsed_ms_list.append((end_t - start_t) * 1000.0)

        avg_ms = sum(elapsed_ms_list) / repeats
        results.append(
            {
                "size": size,
                "ops": size * size,
                "avg_ms": avg_ms,
                "repeats": repeats,
            }
        )

    return results


def print_performance_report(results: list[dict]) -> None:
    """성능 측정 결과를 콘솔 표(O(N²) 복잡도 비교용)로 출력합니다."""
    print("\n" + "=" * 60)
    print(f"성능 분석 결과 (순수 MAC 연산 {BENCHMARK_REPEATS}회 반복 평균, I/O 제외)")
    print("=" * 60)
    print(
        f"{pad_display('행렬 크기 (N×N)', 16)} | {pad_display('연산 횟수 (N²)', 14)} | "
        f"{pad_display('평균 시간 (ms)', 14)} | 기준 대비 배율"
    )
    print("-" * 60)

    base_time = results[0]["avg_ms"] if results and results[0]["avg_ms"] > 0 else None
    for row in results:
        size_str = f"{row['size']} × {row['size']}"
        ratio_str = f"{row['avg_ms'] / base_time:.2f}x" if base_time else "-"
        print(
            f"{size_str:<16} | {row['ops']:<14} | {row['avg_ms']:<14.6f} | {ratio_str}"
        )


# ---------------------------------------------------------------------------
# 메인 메뉴 진입점
# ---------------------------------------------------------------------------
def main() -> None:
    """Mini NPU Simulator 대화형 CLI 진입점."""
    while True:
        print("\n" + "=" * 60)
        print("Mini NPU Simulator (표준 라이브러리 json, time만 사용)")
        print("=" * 60)
        print("1. 모드 1: 사용자 입력 모드 (3×3 필터 2개 + 패턴 판별)")
        print("2. 모드 2: data.json 일괄 판정 모드 (5×5, 13×13, 25×25 등)")
        print("3. 성능 분석: 크기별 순수 MAC 연산 10회 평균 시간(ms) 및 N² 측정")
        print("0. 종료")

        choice = input("실행할 번호를 선택하세요 > ").strip()

        if choice == "1":
            run_interactive_mode(default_size=3)
        elif choice == "2":
            custom_path = input(
                f"JSON 파일 경로 입력 (엔터 시 기본값 '{DEFAULT_DATA_FILE}') > "
            ).strip()
            target_path = custom_path if custom_path else DEFAULT_DATA_FILE
            run_batch_mode(target_path)
        elif choice == "3":
            bench_results = benchmark_mac_performance()
            print_performance_report(bench_results)
        elif choice == "0":
            print("Mini NPU Simulator를 종료합니다.")
            break
        else:
            print("[안내] 올바른 메뉴 번호(0, 1, 2, 3)를 입력해 주세요.")


if __name__ == "__main__":
    main()
