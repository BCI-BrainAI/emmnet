#!/usr/bin/env python3
"""MRI 전처리 CLI. 실제 처리는 src/data/mri_dataset.py에서 수행한다.

python scripts/preprocess_mri.py --config configs/mri_dataset.yaml
YAML 상대 경로는 YAML 폴더 기준, CLI 상대 경로는 현재 작업 폴더 기준.
패키지 설치 없이도 저장소의 src/를 사용한다. 의존성 설치는 필요하다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_settings(args: argparse.Namespace) -> dict:
    import yaml

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("YAML must contain a mapping of settings")
    allowed = {"raw_dir", "out_dir", "metadata_dir", "target_size", "canonical", "split_ratios", "seed"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown YAML settings: {sorted(unknown)}")
    settings = dict(target_size=256, canonical=True, split_ratios=[0.8, 0.1, 0.1], seed=42)
    settings.update(config)
    for key in ("raw_dir", "metadata_dir", "out_dir"):
        override = getattr(args, key)
        value = override if override is not None else settings.get(key)
        if value is None:
            if key == "metadata_dir":
                settings[key] = None
                continue
            raise ValueError(f"Set {key} in YAML or provide --{key.replace('_', '-')}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string")
        path = Path(value).expanduser()
        base = Path.cwd() if override is not None else config_path.parent
        settings[key] = str((base / path).resolve())
    if type(settings["target_size"]) is not int or settings["target_size"] <= 0:
        raise ValueError("target_size must be a positive integer")
    if type(settings["canonical"]) is not bool:
        raise ValueError("canonical must be YAML true or false")
    if type(settings["seed"]) is not int:
        raise ValueError("seed must be an integer")
    ratios = settings["split_ratios"]
    if (not isinstance(ratios, list) or len(ratios) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in ratios)
            or not math.isclose(sum(ratios), 1.0)):
        raise ValueError("split_ratios must be [train, val, test], nonnegative and sum to 1")
    raw = Path(settings["raw_dir"])
    meta = Path(settings["metadata_dir"]) if settings["metadata_dir"] else raw
    out = Path(settings["out_dir"])
    for path in (raw, meta):
        if not path.is_dir():
            raise ValueError(f"Input directory does not exist: {path}")
    if raw.is_relative_to(out) or meta.is_relative_to(out):
        raise ValueError("Output must not equal or contain an input directory")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {out}")
    return settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ADNI Screening/Scaled MRI 전처리")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/mri_dataset.yaml"))
    parser.add_argument("--raw-dir", help="YAML 영상 경로를 덮어씀; 상대 경로는 현재 폴더 기준")
    parser.add_argument("--metadata-dir", help="YAML XML 경로를 덮어씀; 상대 경로는 현재 폴더 기준")
    parser.add_argument("--out-dir", help="YAML 출력 경로를 덮어씀; 상대 경로는 현재 폴더 기준")
    parser.add_argument("--dry-run", action="store_true", help="설정과 경로만 검사; 영상 읽기/저장 없음")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args)
    except ImportError:
        parser.error("PyYAML is required: python -m pip install pyyaml")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except Exception as error:
        # YAML 파서 오류도 사용자에게 짧게 표시한다.
        import yaml
        if isinstance(error, yaml.YAMLError):
            parser.error(f"Invalid YAML: {error}")
        raise
    print(json.dumps(settings, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("Settings/path checks passed. MRI files and metadata contents were not validated.")
        return
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from data.mri_dataset import build_mri_dataset
        build_mri_dataset(**settings)
    except ImportError as error:
        parser.error(f"Missing dependency: {error}. Install project requirements.txt in this Python environment.")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
