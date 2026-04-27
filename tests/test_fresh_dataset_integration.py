from pathlib import Path

import pandas as pd

from src.pipeline.fresh_dataset import load_and_merge_fresh_dataset


def test_fresh_dataset_merge_converts_to_kaggle_status_and_internal_label(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PHISH_DATA_DIR", str(data_root))
    monkeypatch.setenv("PHISH_OUTPUTS_DIR", str(outputs_root))

    kaggle_norm = tmp_path / "kaggle_norm.csv"
    pd.DataFrame(
        [
            {"url": "https://good.example/a", "label": 0, "source_file": "k.csv", "source_dataset": "kaggle"},
            {"url": "https://bad.example/b", "label": 1, "source_file": "k.csv", "source_dataset": "kaggle"},
        ]
    ).to_csv(kaggle_norm, index=False)

    fresh = tmp_path / "fresh.csv"
    pd.DataFrame(
        [
            {"url": "https://x-phish.example/login", "label": 1, "source": "phishstats", "brand_target": "microsoft"},
            {"url": "https://x-legit.example/home", "label": 0, "source": "brand_official", "brand_target": "microsoft"},
            {"url": "https://source-overrides-label.example", "label": 0, "source": "phishstats", "brand_target": "paypal"},
        ]
    ).to_csv(fresh, index=False)

    out_csv, sanity = load_and_merge_fresh_dataset(
        kaggle_normalized_csv=kaggle_norm,
        fresh_dataset_csv=fresh,
        fresh_recent_holdout_csv=None,
    )
    out = pd.read_csv(out_csv)

    assert Path(out_csv).is_file()
    assert sanity["combined_rows_out"] == len(out)
    assert set(pd.to_numeric(out["status"], errors="coerce").dropna().astype(int).unique().tolist()) <= {0, 1}

    fresh_rows = out[out["source_dataset"] == "fresh_phishstats_extension"].copy()
    # source-based mapping must hold (Kaggle convention status: phish=0 legit=1).
    src_map = dict(zip(fresh_rows["fresh_source"], pd.to_numeric(fresh_rows["status"]).astype(int)))
    assert src_map["phishstats"] == 0
    assert src_map["brand_official"] == 1


def test_recent_holdout_urls_are_excluded_from_training_merge(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PHISH_DATA_DIR", str(data_root))
    monkeypatch.setenv("PHISH_OUTPUTS_DIR", str(outputs_root))

    kaggle_norm = tmp_path / "kaggle_norm.csv"
    pd.DataFrame([{"url": "https://kaggle.example/1", "label": 1}]).to_csv(kaggle_norm, index=False)

    fresh = tmp_path / "fresh.csv"
    pd.DataFrame(
        [
            {"url": "https://fresh.example/keep", "label": 1, "source": "phishstats", "registered_domain": "fresh.example"},
            {"url": "https://fresh.example/holdout", "label": 1, "source": "phishstats", "registered_domain": "fresh.example"},
        ]
    ).to_csv(fresh, index=False)

    holdout = tmp_path / "recent.csv"
    pd.DataFrame([{"url": "https://fresh.example/holdout", "registered_domain": "fresh.example"}]).to_csv(
        holdout, index=False
    )

    out_csv, _ = load_and_merge_fresh_dataset(
        kaggle_normalized_csv=kaggle_norm,
        fresh_dataset_csv=fresh,
        fresh_recent_holdout_csv=holdout,
    )
    out = pd.read_csv(out_csv)
    assert "https://fresh.example/holdout" not in set(out["url"].astype(str))
