"""설정 로딩. YAML + 환경변수 + CLI override를 한 곳에서 처리한다.

우선순위(데이터 경로): --processed-dir > 환경변수 EMMNET_DATA > YAML data.processed_dir
임의 키 override: --set train.lr=3e-4 train.seed=1 (값은 YAML 문법으로 해석)
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml

ENV_DATA = "EMMNET_DATA"


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/mri_encoder.yaml")
    parser.add_argument("--processed-dir", help=f"data.processed_dir 덮어쓰기 (환경변수 {ENV_DATA}도 가능)")
    parser.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="YAML 키 덮어쓰기. 예: --set train.lr=3e-4 train.micro_batch_size=4")


def _parse_value(raw: str) -> Any:
    """YAML 해석 + 과학표기('3e-4')는 YAML 1.1에서 문자열이 되므로 숫자로 보정."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
    return value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """'a.b.c=value' 목록을 config에 적용한 복사본을 반환. 존재하지 않는 키는 오타 방지를 위해 에러."""
    out = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override는 KEY=VALUE 형식이어야 한다: {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = out
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                raise KeyError(f"알 수 없는 config 경로: {key}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"알 수 없는 config 키: {key} (오타 방지를 위해 YAML에 이미 있는 키만 허용)")
        node[parts[-1]] = _parse_value(raw)
    return out


def resolve_data_dir(config: dict[str, Any], base: Path | None, processed_dir: str | None = None) -> None:
    """processed_dir를 절대경로로 확정(in-place)."""
    chosen = processed_dir or os.environ.get(ENV_DATA)
    if chosen:
        config["data"]["processed_dir"] = str(Path(chosen).expanduser().resolve())
    elif base is not None:
        config["data"]["processed_dir"] = str((base / config["data"]["processed_dir"]).resolve())


def load_config(config_path: Path, processed_dir: str | None = None,
                overrides: list[str] | None = None) -> dict[str, Any]:
    """YAML 로드 -> override 적용 -> 경로 절대화(YAML 폴더 기준, CLI/환경변수 경로는 CWD 기준)."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = apply_overrides(config, overrides)
    base = config_path.resolve().parent
    resolve_data_dir(config, base, processed_dir)
    pre = config["model"].get("pretrained_path")
    if pre:
        config["model"]["pretrained_path"] = str((base / pre).resolve())
    return config
