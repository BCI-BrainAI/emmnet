"""합성 데이터 end-to-end 스모크 테스트: Dataset -> 정규화 -> 학습 -> 평가."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.mri_dataset import MRIDataset, med3d_normalize
from evaluation.evaluate import evaluate_split, load_checkpoint
from evaluation.metrics import confusion_metrics, roc_auc, summarize, youden_threshold
from models.mri.classifier import MRIClassifier
from training.losses import BinaryFocalLoss
from training.trainer import train

SIZE = 32


def make_synthetic(root: Path, n_per_split: dict[str, int]) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    idx = 0
    for split, n in n_per_split.items():
        for k in range(n):
            label = k % 2
            vol = np.zeros((SIZE,) * 3, np.float32)
            vol[6:26, 6:26, 6:26] = rng.random((20, 20, 20)) * 0.5 + 0.1 + 0.3 * label
            vol = (vol - vol.min()) / (vol.max() - vol.min())
            image_id = f"I{idx}"
            (root / "volumes").mkdir(exist_ok=True)
            np.save(root / "volumes" / f"{image_id}.npy", vol)
            rows.append(dict(subject_id=f"S{idx}", image_id=image_id, split=split, label=label,
                             processed_relpath=f"volumes/{image_id}.npy", target_size=SIZE))
            idx += 1
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root


def test_med3d_normalize_properties():
    v = np.zeros((16,) * 3, np.float32)
    v[4:12, 4:12, 4:12] = np.random.default_rng(1).random((8, 8, 8)) + 0.1
    out = med3d_normalize(v)
    fg = v > 0
    assert abs(out[fg].mean()) < 1e-4 and abs(out[fg].std() - 1) < 1e-3
    assert (out[~fg] == 0).all()
    # 아핀 불변: min-max 스케일을 바꿔도 결과 동일
    assert np.allclose(out, med3d_normalize(v * 3.0), atol=1e-4)


def test_dataset_split_and_leakage_check(tmp_path):
    root = make_synthetic(tmp_path, {"train": 4, "val": 2, "test": 2})
    ds = MRIDataset(root, "val")
    x, y = ds[0]
    assert x.shape == (1, SIZE, SIZE, SIZE) and x.dtype == torch.float32 and y in (0, 1)
    rows = list(csv.DictReader((root / "manifest.csv").open()))
    rows[0]["split"] = "val"  # 같은 subject를 다른 split에 중복
    dup = dict(rows[1]); dup["subject_id"] = rows[0]["subject_id"]; dup["image_id"] = "Idup"
    with pytest.raises(ValueError, match="leakage"):
        MRIDataset(root, "val", manifest=rows + [dup])


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert roc_auc(y, np.array([0.5] * 4)) == pytest.approx(0.5)
    m = confusion_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]), 0.5)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert 0.0 <= youden_threshold(y, np.array([0.1, 0.2, 0.7, 0.8])) <= 1.0
    assert np.isnan(roc_auc(np.zeros(3), np.random.rand(3)))
    assert summarize(y, np.array([0.1, 0.2, 0.7, 0.8]), 0.5)["auc"] == 1.0


def test_focal_loss_reduces_to_weighted_bce_when_gamma0():
    logits = torch.tensor([0.3, -1.2, 2.0]); t = torch.tensor([1.0, 0.0, 1.0])
    fl = BinaryFocalLoss(alpha=0.25, gamma=0.0, reduction="none")(logits, t)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, t, reduction="none")
    assert torch.allclose(fl, bce * torch.tensor([0.25, 0.75, 0.25]), atol=1e-6)


def test_train_and_evaluate_smoke(tmp_path):
    root = make_synthetic(tmp_path / "data", {"train": 8, "val": 4, "test": 4}) if (tmp_path / "data").mkdir() is None else None
    config = {
        "data": {"processed_dir": str(root), "normalization": "percentile_zscore", "num_workers": 0},
        "input": {"channels": 1}, "model": {"proj_dim": 16},
        "train": {"batch_size": 4, "micro_batch_size": 2, "epochs": 2, "lr": 1e-3, "amp": False,
                  "device": "cpu", "early_stopping_patience": 5, "seed": 0},
    }
    model = MRIClassifier(in_channels=1, proj_dim=16)
    out_dir = tmp_path / "ckpt"
    result = train(model, config, out_dir)
    assert (out_dir / "best.pth").exists() and (out_dir / "last.pth").exists()
    assert len(result["history"]) >= 1 and all(np.isfinite(h["train_loss"]) for h in result["history"])
    ckpt = load_checkpoint(out_dir / "best.pth")
    model2 = MRIClassifier(in_channels=1, proj_dim=16)
    model2.load_state_dict(ckpt["state_dict"])
    metrics, rows = evaluate_split(model2, config, ckpt, "test")
    assert len(rows) == 4
    assert metrics["n"] == 4 and 0.0 <= metrics["accuracy"] <= 1.0
    assert json.dumps(metrics)


def test_train_script_pretrained_keys_match_if_present():
    """실제 Med3D 가중치가 있으면 backbone 키가 완전히 일치해야 한다."""
    ckpt = Path(__file__).resolve().parents[1] / "pretrained" / "resnet_18.pth"
    if not ckpt.exists():
        pytest.skip("pretrained/resnet_18.pth 없음")
    from models.mri.mri_encoder import load_med3d_pretrained
    model = MRIClassifier()
    missing, unexpected = load_med3d_pretrained(model.encoder, ckpt)
    assert missing == [] and unexpected == []


def test_lr_factor_warmup_then_cosine():
    from training.trainer import lr_factor
    f = [lr_factor(e, 10, 2) for e in range(10)]
    assert f[0] < f[1] < f[2] <= 1.0 and f[2] == pytest.approx(1.0)
    assert all(f[i] >= f[i + 1] for i in range(2, 9)) and f[-1] < 0.1
