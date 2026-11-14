"""
Mini NPU Simulator 보너스 과제 모듈
===================================
1. 패턴 생성기: 크기 N을 입력하면 N×N 십자가(Cross)와 X 패턴을 자동 생성합니다.
   생성한 패턴은 모드 1(필터 A/B 자동 생성)과 성능 분석(측정용 입력)에서 재활용합니다.
2. 시뮬레이터 최적화(메모리 접근): 2차원 배열을 길이 N²의 1차원 배열로 변환해 MAC을 계산하고,
   최적화 전(2차원)/후(1차원) 성능을 동일 입력·동일 반복 횟수로 비교합니다.

main.py와 같은 제약을 따라 표준 라이브러리 time만 사용합니다.
main.py의 compute_mac은 순환 import를 피하기 위해 함수 인자로 전달받습니다.
"""

import time

# 2차원/1차원 MAC 결과가 같은지 확인할 때 쓰는 허용오차 (main.py의 EPSILON과 같은 값)
MATCH_TOLERANCE = 1e-9

# 최적화 비교 대상 크기: 미션 기본 크기 + 차이가 잘 드러나는 큰 크기
COMPARE_SIZES = (3, 5, 13, 25, 51, 101)
COMPARE_REPEATS = 10

# 패턴 미리보기에 쓰는 기호 (미션 안내문과 같은 □/■ 표기)
FILLED_CELL = "■"
EMPTY_CELL = "□"


# ---------------------------------------------------------------------------
# 1. 패턴 생성기
# ---------------------------------------------------------------------------
def validate_pattern_size(size) -> int:
    """패턴 크기는 가운데 칸이 존재하도록 3 이상의 홀수 정수여야 합니다."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"패턴 크기는 정수여야 합니다: {size!r}")
    if size < 3 or size % 2 == 0:
        raise ValueError(f"패턴 크기는 3 이상의 홀수여야 합니다 (가운데 칸 필요): {size}")
    return size


def generate_cross_pattern(size: int) -> list[list[float]]:
    """가운데 행과 가운데 열이 1.0이고 나머지는 0.0인 N×N 십자가(Cross) 패턴을 만듭니다."""
    validate_pattern_size(size)
    mid = size // 2
    return [[1.0 if (i == mid or j == mid) else 0.0 for j in range(size)] for i in range(size)]


def generate_x_pattern(size: int) -> list[list[float]]:
    """두 대각선(i == j, i + j == N - 1)이 1.0이고 나머지는 0.0인 N×N X 패턴을 만듭니다."""
    validate_pattern_size(size)
    return [
        [1.0 if (i == j or i + j == size - 1) else 0.0 for j in range(size)]
        for i in range(size)
    ]


def render_pattern(matrix: list[list[float]]) -> str:
    """패턴을 □/■ 기호로 그린 문자열을 반환합니다 (0이 아닌 칸은 ■)."""
    return "\n".join(
        " ".join(FILLED_CELL if value != 0 else EMPTY_CELL for value in row) for row in matrix
    )


def read_pattern_size() -> int:
    """패턴 크기 N을 입력받습니다. 잘못된 입력이면 안내 후 다시 입력받습니다."""
    while True:
        raw = input("생성할 패턴 크기 N 입력 (3 이상의 홀수) > ").strip()
        try:
            return validate_pattern_size(int(raw))
        except ValueError as exc:
            print(f"  [입력 오류] {exc if raw.lstrip('-').isdigit() else '숫자를 입력하세요.'}")


def run_pattern_generator() -> None:
    """보너스 1: N을 입력받아 Cross/X 패턴을 생성하고 화면에 그려 줍니다."""
    size = read_pattern_size()
    print(f"\n[{size}×{size} Cross 패턴]\n{render_pattern(generate_cross_pattern(size))}")
    print(f"\n[{size}×{size} X 패턴]\n{render_pattern(generate_x_pattern(size))}")
    print("\n※ 생성한 패턴은 모드 1(필터 A/B 자동 생성)과 성능 분석의 측정용 입력으로도 쓰입니다.")


def ask_generated_filters(size: int) -> tuple[list[list[float]], list[list[float]]] | None:
    """
    모드 1에서 필터 A/B를 직접 입력하는 대신 생성기로 만들지 묻습니다.
    'y'이면 (A=Cross 패턴, B=X 패턴)을 반환하고, 그 외에는 None을 반환해 직접 입력하게 합니다.
    """
    answer = input("필터 A/B를 패턴 생성기로 자동 생성할까요? (A=Cross, B=X) [y/N] > ").strip().lower()
    if answer not in ("y", "yes"):
        return None
    return generate_cross_pattern(size), generate_x_pattern(size)


# ---------------------------------------------------------------------------
# 2. 시뮬레이터 최적화: 2차원 → 1차원 배열
# ---------------------------------------------------------------------------
def flatten_matrix(matrix: list[list[float]]) -> list[float]:
    """N×N 2차원 배열을 행 우선(row-major) 순서의 길이 N² 1차원 배열로 변환합니다."""
    return [value for row in matrix for value in row]


def compute_mac_flat(flat_a: list[float], flat_b: list[float]) -> float:
    """
    1차원 배열 두 개의 MAC 연산을 단일 반복문으로 수행합니다.
    2차원 버전과 달리 행 리스트를 꺼내는 단계가 없어 인덱스 접근이 한 번으로 줄어듭니다.
    """
    if len(flat_a) != len(flat_b) or not flat_a:
        raise ValueError(f"1차원 MAC 길이가 일치하지 않습니다: {len(flat_a)} vs {len(flat_b)}")
    acc = 0.0
    for k in range(len(flat_a)):
        acc += flat_a[k] * flat_b[k]
    return acc


def average_ms(func, args: tuple, repeats: int) -> float:
    """func(*args) 호출 구간만 repeats회 측정해 평균 시간(ms)을 반환합니다."""
    total = 0.0
    for _ in range(repeats):
        start_t = time.perf_counter()
        func(*args)
        total += time.perf_counter() - start_t
    return total / repeats * 1000.0


def compare_mac_layouts(compute_mac_2d, sizes=COMPARE_SIZES, repeats: int = COMPARE_REPEATS) -> list[dict]:
    """
    크기별로 같은 입력(X 패턴 × Cross 필터)에 대해 2차원/1차원 MAC 시간을 같은 횟수로 측정합니다.
    1차원 변환(flatten)은 '데이터 준비' 단계로 보고 타이머 밖에서 수행합니다.
    두 방식의 결과가 같은지도 함께 검증합니다.
    """
    rows: list[dict] = []
    for size in sizes:
        pattern, filt = generate_x_pattern(size), generate_cross_pattern(size)
        flat_pattern, flat_filter = flatten_matrix(pattern), flatten_matrix(filt)

        same_result = abs(compute_mac_2d(pattern, filt) - compute_mac_flat(flat_pattern, flat_filter))
        ms_2d = average_ms(compute_mac_2d, (pattern, filt), repeats)
        ms_1d = average_ms(compute_mac_flat, (flat_pattern, flat_filter), repeats)
        rows.append({
            "size": size, "ops": size * size, "ms_2d": ms_2d, "ms_1d": ms_1d,
            "match": same_result < MATCH_TOLERANCE,
        })
    return rows


def print_layout_comparison(rows: list[dict], repeats: int = COMPARE_REPEATS) -> None:
    """최적화 전/후 비교 결과를 표로 출력합니다. 개선율 = (전 - 후) / 전."""
    print("\n" + "=" * 72)
    print(f"최적화 전/후 비교 (2차원 vs 1차원, 동일 입력, 각 {repeats}회 반복 평균, I/O 제외)")
    print("=" * 72)
    # 한글은 콘솔에서 2칸을 차지하므로 제목 줄은 화면 폭 기준으로 공백을 직접 맞춥니다.
    print("크기      | 연산 횟수    | 2차원 (ms)   | 1차원 (ms)   |   개선율 | 결과 일치")
    print("-" * 72)
    for row in rows:
        size_str = f"{row['size']}×{row['size']}"
        gain = (row["ms_2d"] - row["ms_1d"]) / row["ms_2d"] * 100 if row["ms_2d"] > 0 else 0.0
        match_str = "O" if row["match"] else "X (불일치)"
        print(
            f"{size_str:<9} | {row['ops']:<12} | {row['ms_2d']:<12.6f} | {row['ms_1d']:<12.6f} | "
            f"{gain:>7.1f}% | {match_str}"
        )


def run_bonus_menu(compute_mac_2d) -> None:
    """보너스 과제 하위 메뉴: 패턴 생성기 / 최적화 전후 비교."""
    while True:
        print("\n--- 보너스 과제 ---")
        print("1. 패턴 생성기 (N×N Cross / X 자동 생성)")
        print("2. 최적화 전/후 성능 비교 (2차원 vs 1차원 배열)")
        print("0. 메인 메뉴로 돌아가기")
        choice = input("번호 선택 > ").strip()
        if choice == "1":
            run_pattern_generator()
        elif choice == "2":
            print_layout_comparison(compare_mac_layouts(compute_mac_2d))
        elif choice == "0":
            return
        else:
            print("[안내] 올바른 번호(0, 1, 2)를 입력해 주세요.")
