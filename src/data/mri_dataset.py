"""ADNI MRI 전처리 및 PyTorch Dataset.

논문 기준: zero-padding 후 256^3 resize, 영상별 min-max [0, 1]
(EMMNet Sec 3.1, p.283). Med3D의 percentile truncation + z-score와
다르므로 이번 구현에서는 EMMNet의 정규화를 따른다.

이번 구현의 결정 사항
--------------------
- ADNI Screening / 마지막 처리 항목이 Scaled인 영상만 사용한다.
  Scaled_2는 제외하며 피험자별 복수 후보는 임의로 선택하지 않는다.

- 제공 영상은 이미 N3/Scaled 등의 처리를 거쳤다. 처리 이력을 보존하고
  추가 N4/MNI 정합은 하지 않는다. ADNI에서의 구현 검증이며 원 논문의
  CAUEMM 데이터/성능 재현과 구분한다.

- XML researchGroup을 잠정 라벨로 사용한다: CN=0, MCI=1, AD=1.
  진단 변경 대상자 제외 여부 및 Screening 진단과의 일치는 미확인이다.

- 구현 기본값: RAS 축 방향 통일(정합/등방성 재표본화 아님), 최대 축 길이의
  정육면체로 중앙 zero-padding, trilinear resize(align_corners=False),
  resize 후 min-max. 원래 상수 영상/비유한 값/비3D 영상은 실패 처리한다.
  .npy float32 [D,H,W] 저장. RAS 배열의 축 0/1/2를 D/H/W로 사용한다.

- researchGroup별 피험자 분할 기본값 80/10/10, seed=42. 증강은 하지 않는다.
  분할 목록을 저장하고 Dataset에서 분할 간 피험자 중복을 검사한다.
  향후 방문/증강 추가 시에도 동일 피험자는 같은 split을 유지해야 한다.
  
- 논문의 '4,000 training samples' 구성은 미확인이다. 이를 맞추려고
  임의 증강하지 않는다. 서버 영상 ID/파일 연결은 실행 시 검사한다.

기본 호출 build_mri_dataset(raw_dir, out_dir)는 raw_dir 내 XML을 찾는다.
XML이 별도 위치에 있다면 metadata_dir=...를 전달한다. 출력 폴더는 새 폴더
또는 빈 폴더여야 한다. manifest.csv와 run.json, excluded.csv를 함께 저장한다.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

LABELS = {"CN": 0, "MCI": 1, "AD": 1}
SPLITS = ("train", "val", "test")


def read_mri_metadata(metadata_dir: str | Path) -> list[dict[str, Any]]:
    """상세 XML만 읽어 경로용 보조 XML과의 이중 집계를 피한다."""
    root = Path(metadata_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.xml")):
        xml = ET.parse(path).getroot()
        for element in xml.iter():
            element.tag = element.tag.rsplit("}", 1)[-1]
        if xml.tag == "metadata":
            continue  # 동일 영상의 경로용 보조 XML
        product = xml.find(".//derivedProduct")
        if product is None:
            continue
        def text(tag: str) -> str:
            return (xml.findtext(".//" + tag) or "").strip()
        image = (product.findtext("imageUID") or "").strip()
        subject = text("subjectIdentifier")
        if not image or not subject:
            raise ValueError(f"Missing image/subject ID: {path}")
        image_id = "I" + image.removeprefix("I")
        row = {
            "dataset": text("projectIdentifier"),
            "subject_id": subject,
            "study_id": text("studyIdentifier"),
            "series_id": "S" + text("seriesIdentifier").removeprefix("S"),
            "image_id": image_id,
            "source_image_id": "I" + (product.findtext("relatedImage/imageUID") or "").removeprefix("I"),
            "visit_name": text("visitIdentifier"),
            "scan_date": text("dateAcquired"),
            "research_group": text("researchGroup"),
            "age_at_scan": text("subjectAge"),
            "sex": text("subjectSex"),
            "processing_description": product.findtext("processedDataLabel") or "",
            "registration": product.findtext("registration") or "",
            "metadata_relpath": path.relative_to(root).as_posix(),
            "source_processing": json.dumps([
                {e.tag: e.text for e in step} for step in product.findall("provenanceDetail")
            ], ensure_ascii=False),
        }
        if image_id in records:
            raise ValueError(f"Duplicate detailed XML image_id: {image_id}")
        records[image_id] = row
    if not records:
        raise ValueError(f"No detailed ADNI XML found: {root}")
    return list(records.values())


def select_mri_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Screening + 정확히 Scaled만 선택. 중복 피험자 후보는 모두 제외."""
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = []
    for original in records:
        row = dict(original)
        if row["dataset"] != "ADNI":
            reason = "not_adni"
        elif row["visit_name"] != "ADNI Screening":
            reason = "not_screening"
        elif row["processing_description"].split(";")[-1].strip() != "Scaled":
            reason = "not_scaled"
        elif row["research_group"] not in LABELS:
            reason = "unknown_research_group"
        else:
            candidates[row["subject_id"]].append(row)
            continue
        excluded.append({**row, "exclusion_reason": reason})
    selected = []
    for subject in sorted(candidates):
        rows = candidates[subject]
        if len(rows) != 1:
            excluded.extend({**r, "exclusion_reason": "multiple_screening_scaled_candidates"} for r in rows)
            continue
        row = rows[0]
        row.update(label=LABELS[row["research_group"]], label_source="xml_researchGroup", label_verified=False)
        selected.append(row)
    return selected, excluded


def assign_subject_splits(records: list[dict[str, Any]], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> list[dict[str, Any]]:
    """원본 연구군별 층화 분할. 소수 그룹에서는 일부 split이 비어 있을 수 있다."""
    if len(ratios) != 3 or any(not math.isfinite(r) or r < 0 for r in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("split_ratios must be three nonnegative finite values summing to 1")
    groups: dict[str, set[str]] = defaultdict(set)
    diagnoses = {}
    for row in records:
        subject, group = row["subject_id"], row["research_group"]
        if subject in diagnoses and diagnoses[subject] != group:
            raise ValueError(f"Conflicting subject groups: {subject}")
        diagnoses[subject] = group
        groups[group].add(subject)
    rng = random.Random(seed)
    mapping = {}
    for group in sorted(groups):
        subjects = sorted(groups[group])
        rng.shuffle(subjects)
        exact = [len(subjects) * r for r in ratios]
        counts = [math.floor(x) for x in exact]
        order = sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))
        for i in order[:len(subjects) - sum(counts)]:
            counts[i] += 1
        start = 0
        for split, count in zip(SPLITS, counts):
            for subject in subjects[start:start + count]:
                mapping[subject] = split
            start += count
    return [{**r, "split": mapping[r["subject_id"]]} for r in records]


def preprocess_mri(path: str | Path, target_size: int = 256, canonical: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """NIfTI를 텐서용 배열로 변환. affine은 sidecar에 보존하며 결과는 NIfTI가 아님."""
    import nibabel as nib
    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")
    source = nib.load(str(path))
    if len(source.shape) != 3 or min(source.shape) <= 0:
        raise ValueError(f"Expected nonempty 3D NIfTI: {source.shape}")
    info = {"original_shape": list(source.shape), "original_spacing": [float(x) for x in source.header.get_zooms()],
            "original_affine": source.affine.tolist(), "original_orientation": list(nib.aff2axcodes(source.affine))}
    img = nib.as_closest_canonical(source) if canonical else source
    volume = img.get_fdata(dtype=np.float32)
    if not np.isfinite(volume).all() or float(volume.max()) <= float(volume.min()):
        raise ValueError("Non-finite or constant MRI volume")
    side = max(volume.shape)
    padding = [((side - n) // 2, (side - n + 1) // 2) for n in volume.shape]
    padded = np.pad(volume, padding, mode="constant", constant_values=0)
    tensor = torch.from_numpy(np.ascontiguousarray(padded))[None, None]
    with torch.no_grad():
        tensor = F.interpolate(tensor, size=(target_size,) * 3, mode="trilinear", align_corners=False)
        low, high = tensor.amin(), tensor.amax()
        if not torch.isfinite(tensor).all() or high <= low:
            raise ValueError("Invalid/constant volume after resize")
        result = ((tensor - low) / (high - low))[0, 0].numpy().astype(np.float32)
    info.update(canonical=canonical, orientation=list(nib.aff2axcodes(img.affine)),
                oriented_shape=list(volume.shape), padding=padding, output_shape=list(result.shape),
                interpolation="trilinear", align_corners=False, normalization="minmax_after_resize",
                normalization_min=float(low), normalization_max=float(high), axis_mapping="array axes 0,1,2 -> D,H,W")
    return result, info


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["image_id", "exclusion_reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_mri_dataset(raw_dir: str | Path, out_dir: str | Path, *, metadata_dir: str | Path | None = None,
                      target_size: int = 256, split_ratios=(0.8, 0.1, 0.1), seed: int = 42,
                      canonical: bool = True) -> None:
    """NIfTI 파일명에 포함된 I숫자 ID로 연결. DICOM은 별도 변환이 필요하다.

    실패/누락/중복 파일은 excluded.csv에 기록한다. 실패 후 재분할하지 않는다.
    XML이 서버 영상과 별도 위치이면 metadata_dir를 반드시 지정한다.
    """
    raw, out = Path(raw_dir).resolve(), Path(out_dir).resolve()
    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")
    meta = Path(metadata_dir).resolve() if metadata_dir is not None else raw
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    if out == raw or out == meta or raw.is_relative_to(out) or meta.is_relative_to(out):
        raise ValueError("Output must not contain the input directories")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {out}")
    records, excluded = select_mri_records(read_mri_metadata(meta))
    records = assign_subject_splits(records, split_ratios, seed)
    if not records:
        raise ValueError("No eligible Screening / Scaled records")
    files: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(raw.rglob("*")):
        if out in path.parents or not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
            continue
        for image_id in set(re.findall(r"(?<![A-Za-z0-9])I\d+(?![A-Za-z0-9])", path.name)):
            files[image_id].append(path)
    # 의존성 누락은 모든 영상을 실패로 기록하지 않고 실행 전에 알린다.
    import nibabel  # noqa: F401
    out.mkdir(parents=True, exist_ok=True)
    (out / "volumes").mkdir()
    (out / "provenance").mkdir()
    completed = []
    settings = dict(raw_dir=str(raw), metadata_dir=str(meta), target_size=target_size, canonical=canonical,
                    split_ratios=list(split_ratios), seed=seed, labels=LABELS, visit="ADNI Screening",
                    processing="Scaled", status="running")
    (out / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    _write_csv(out / "selection.csv", records)
    for record in records:
        row = dict(record)
        matches = files[row["image_id"]]
        if len(matches) != 1:
            excluded.append({**row, "exclusion_reason": "image_missing" if not matches else "multiple_image_files"})
            continue
        path = matches[0]
        try:
            array, info = preprocess_mri(path, target_size, canonical)
        except (ValueError, OSError, RuntimeError, nibabel.filebasedimages.ImageFileError) as error:
            excluded.append({**row, "exclusion_reason": f"preprocessing_failed: {error}"})
            continue
        relative = f"volumes/{row['image_id']}.npy"
        np.save(out / relative, array, allow_pickle=False)
        info.update(image_id=row["image_id"], image_relpath=path.relative_to(raw).as_posix(),
                    source_processing=json.loads(row["source_processing"]))
        (out / "provenance" / f"{row['image_id']}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        row.update(image_relpath=path.relative_to(raw).as_posix(), processed_relpath=relative,
                   target_size=target_size, qc_status="numeric_checks_passed_visual_review_pending")
        completed.append(row)
    _write_csv(out / "manifest.csv", completed)
    _write_csv(out / "excluded.csv", excluded)
    settings.update(status="complete" if completed else "no_usable_images", selected=len(records), processed=len(completed),
                    excluded=len(excluded), split_counts={s: sum(r["split"] == s for r in completed) for s in SPLITS})
    (out / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if not completed:
        raise ValueError(f"No MRI processed; inspect {out / 'excluded.csv'}")
    print(f"Processed {len(completed)}/{len(records)} selected MRIs. Manifest: {out / 'manifest.csv'}")


NORMALIZATIONS = ("percentile_zscore", "minmax")


def med3d_normalize(volume: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Med3D 사전학습 방식 정규화 [Med3D Eq.2]: percentile truncation 후 z-score.

    - 전경(volume > 0) 복셀로 percentile/mean/std를 계산한다.
    - 전경은 [p_low, p_high]로 clip 후 (x-mean)/std, 배경(0)은 0으로 둔다.
      (Med3D 공식 코드는 배경을 N(0,1) 난수로 채우지만 재현성을 위해 0을 사용.)
    - 저장된 .npy는 min-max 값이며 min-max는 아핀 변환이라 percentile+z-score 결과가
      원본 강도에 직접 적용한 것과 동일하다. 따라서 전처리 산출물은 그대로 쓴다.
    """
    fg = volume > 0
    if not fg.any():
        raise ValueError("Empty foreground")
    vals = volume[fg]
    lo, hi = np.percentile(vals, [low_pct, high_pct])
    vals = np.clip(vals, lo, hi)
    std = float(vals.std())
    if std < 1e-8:
        raise ValueError("Zero-variance foreground")
    out = np.zeros_like(volume, dtype=np.float32)
    out[fg] = ((vals - vals.mean()) / std).astype(np.float32)
    return out


class MRIDataset(Dataset):
    """전체 manifest를 검사한 뒤 요청 split의 (float32 [1,D,H,W], int label) 반환.

    normalization: percentile_zscore(기본, Med3D 사전학습과 동일) | minmax(저장값 [0,1] 그대로).
    """
    def __init__(self, processed_dir: str | Path, split: str, manifest: Any = None,
                 normalization: str = "percentile_zscore"):
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"Unknown normalization: {normalization}")
        self.normalization = normalization
        self.root = Path(processed_dir).resolve()
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if manifest is None or isinstance(manifest, (str, Path)):
            path = self.root / "manifest.csv" if manifest is None else Path(manifest)
            if manifest is not None and not path.is_absolute():
                path = self.root / path
            with path.open(encoding="utf-8-sig", newline="") as f:
                records = list(csv.DictReader(f))
        else:
            records = [dict(row) for row in manifest]
        subjects, images = {}, set()
        for row in records:
            for key in ("subject_id", "image_id", "split", "label", "processed_relpath", "target_size"):
                if key not in row or str(row[key]).strip() == "":
                    raise ValueError(f"Missing manifest field: {key}")
            subject, assigned = row["subject_id"], row["split"]
            if assigned not in SPLITS:
                raise ValueError(f"Invalid manifest split: {assigned}")
            if subject in subjects and subjects[subject] != assigned:
                raise ValueError(f"Subject leakage across splits: {subject}")
            subjects[subject] = assigned
            if row["image_id"] in images:
                raise ValueError(f"Duplicate image: {row['image_id']}")
            images.add(row["image_id"])
            row["label"] = int(row["label"])
            row["target_size"] = int(row["target_size"])
            if row["label"] not in (0, 1) or row["target_size"] <= 0:
                raise ValueError("Invalid binary label or target_size")
            path = (self.root / row["processed_relpath"]).resolve()
            if Path(row["processed_relpath"]).is_absolute() or not path.is_relative_to(self.root):
                raise ValueError("processed_relpath must stay within processed_dir")
            if not path.is_file():
                raise FileNotFoundError(path)
        self.records = [row for row in records if row["split"] == split]
        if not self.records:
            raise ValueError(f"No samples for split={split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.records[idx]
        volume = np.load(self.root / row["processed_relpath"], allow_pickle=False)
        if volume.shape != (row["target_size"],) * 3 or volume.dtype != np.float32:
            raise ValueError(f"Invalid MRI shape/dtype: {row['image_id']}")
        if not np.isfinite(volume).all() or volume.min() < 0 or volume.max() > 1:
            raise ValueError(f"Invalid normalized MRI values: {row['image_id']}")
        if self.normalization == "percentile_zscore":
            volume = med3d_normalize(volume)
        return torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0), row["label"]
