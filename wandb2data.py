import re
import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

try:
    import pandas as pd  # required for DataFrame
except ImportError as e:
    raise ImportError("pandas가 필요합니다. `pip install pandas`로 설치하세요.") from e


# =============================================================================
# wandb2data.py
#
# 목적:
#   오프라인 환경에서 W&B(.wandb) 로그 파일(텍스트 형태)을 읽어,
#   tqdm 진행 라인에 반복적으로 포함되는 학습 지표를 추출한다.
#
# 추출 항목:
#   - step        : 현재 스텝 (예: 3/100000에서 3)
#   - total_step  : 총 스텝   (예: 3/100000에서 100000)
#   - s_per_it    : iteration당 소요 시간(초). "Xs/it" 또는 "Yit/s"에서 변환
#   - accuracy    : accuracy=...
#   - ce          : ce=...
#
# 제공 기능(분리):
#   1) parse_wandb_logs_to_df(path_log) -> DataFrame 반환 (저장 없음)
#   2) save_df_to_csv(df, path_csv) -> CSV 저장
# =============================================================================


# -------------------------
# Regex patterns
# -------------------------
STEP_TOTAL_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
S_PER_IT_RE = re.compile(r"\b([0-9]*\.?[0-9]+)\s*s/it\b")
IT_PER_S_RE = re.compile(r"\b([0-9]*\.?[0-9]+)\s*it/s\b")
ACC_RE = re.compile(r"\baccuracy\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\b")
CE_RE = re.compile(r"\bce\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\b")

LIKELY_TQDM_LINE = re.compile(r"\bStep:\s*.*\|\s*\d+\s*/\s*\d+\s*\[")


def iter_wandb_files(path_log: Union[str, Path]) -> Iterator[Path]:
    """
    path_log가
      - 단일 .wandb 파일이면 해당 파일만 yield
      - 디렉토리면 하위 포함 *.wandb 전부 yield
    """
    p = Path(path_log)
    if p.is_file():
        if p.suffix == ".wandb":
            yield p
        else:
            raise ValueError(f"파일이지만 .wandb가 아닙니다: {p}")
    elif p.is_dir():
        for fp in sorted(p.rglob("*.wandb")):
            yield fp
    else:
        raise FileNotFoundError(f"경로가 존재하지 않습니다: {p}")


def read_text_lines(fp: Path) -> Iterator[str]:
    """
    .wandb 파일은 텍스트처럼 보이지만, 깨진/바이너리 조각이 섞일 수 있으므로
    errors='ignore'로 최대한 라인 기반으로 복구한다.
    """
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line.rstrip("\n")


def parse_tqdm_line(line: str) -> Optional[Dict[str, Optional[float]]]:
    """
    tqdm 진행 라인에서 필요한 값만 추출해 dict로 반환.
    매칭 실패/유효값 부족 시 None.

    반환 키:
      step(int), total_step(int), s_per_it(float|None), accuracy(float|None), ce(float|None)
    """
    if not LIKELY_TQDM_LINE.search(line):
        return None

    m_st = STEP_TOTAL_RE.search(line)
    if not m_st:
        return None

    step = int(m_st.group(1))
    total_step = int(m_st.group(2))

    s_per_it: Optional[float] = None
    m_spi = S_PER_IT_RE.search(line)
    if m_spi:
        s_per_it = float(m_spi.group(1))
    else:
        m_ips = IT_PER_S_RE.search(line)
        if m_ips:
            it_per_s = float(m_ips.group(1))
            if it_per_s > 0:
                s_per_it = 1.0 / it_per_s

    m_acc = ACC_RE.search(line)
    m_ce = CE_RE.search(line)

    if not (m_acc or m_ce):
        return None

    accuracy = float(m_acc.group(1)) if m_acc else None
    ce = float(m_ce.group(1)) if m_ce else None

    return {
        "step": step,
        "total_step": total_step,
        "s_per_it": s_per_it,
        "accuracy": accuracy,
        "ce": ce,
    }


def extract_rows(path_log: Union[str, Path], dedup_by_step: bool = True) -> List[Dict]:
    """
    path_log 내 .wandb 로그(파일 또는 디렉토리)에서 파싱 가능한 모든 row를 수집.

    dedup_by_step=True:
      같은 step이 여러 번 등장하면 최초 1개만 유지.
    """
    rows: List[Dict] = []
    seen_steps = set()

    for fp in iter_wandb_files(path_log):
        for line in read_text_lines(fp):
            parsed = parse_tqdm_line(line)
            if parsed is None:
                continue

            if dedup_by_step:
                st = parsed["step"]
                if st in seen_steps:
                    continue
                seen_steps.add(st)

            rows.append(parsed)

    rows.sort(key=lambda r: r["step"])
    return rows


def parse_wandb_logs_to_df(path_log: Union[str, Path], dedup_by_step: bool = True) -> "pd.DataFrame":
    """
    상위 함수(저장 없음):
      - path_log에서 필요한 지표를 파싱
      - pandas DataFrame으로 반환
    """
    rows = extract_rows(path_log=path_log, dedup_by_step=dedup_by_step)
    return pd.DataFrame(rows, columns=["step", "total_step", "s_per_it", "accuracy", "ce"])


def save_df_to_csv(df: "pd.DataFrame", path_csv: Union[str, Path]) -> None:
    """
    DataFrame을 CSV로 저장. (parse_wandb_logs_to_df와 기능 분리)
    """
    out = Path(path_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")


# =============================================================================
# CLI entrypoint
# =============================================================================
if __name__ == "__main__":
    # 실행 예시:
    #   1) path_log에서 파싱해서 df 생성
    #   2) df를 path_csv로 저장
    path_log = r"/path/to/wandb_or_dir"  # .wandb 파일 경로 또는 .wandb들이 있는 디렉토리
    path_csv = r"/path/to/output.csv"    # 저장할 CSV 파일 경로
	"""
	# Examples 
    path_log = "mlp_scm_v008/wandb/offline-run-20260228_032324-vastai000/run-vastai000.wandb"
    path_csv = "result2.csv"
	"""

    df = parse_wandb_logs_to_df(path_log=path_log, dedup_by_step=True)
    save_df_to_csv(df, path_csv=path_csv)

    print(f"Saved: {path_csv}")
    print(f"Rows: {len(df)}")
    print(df.head(10).to_string(index=False))