from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.pipeline.fresh_data import collect_tranco, deduplicate_urls, label_sanity_check


def _patch_retrain_deploy_models_dir(tmp_path: Path) -> object:
    deploy = tmp_path / "deploy_models"
    deploy.mkdir(parents=True, exist_ok=True)
    return patch("src.pipeline.retrain_with_fresh.models_dir", return_value=deploy)
from src.pipeline.merge_datasets import merge_datasets
from src.pipeline.retrain_with_fresh import ensure_enrich_compatible, ensure_split_compatible, retrain_with_fresh
from src.pipeline.split_leak_safe import stratified_group_train_test


def test_label_sanity_check_keeps_only_0_1() -> None:
    df = pd.DataFrame({"url": ["a", "b", "c"], "status": [0, 1, 7]})
    out = label_sanity_check(df)
    assert set(out["status"].tolist()) == {0, 1}


def test_deduplicate_urls_removes_exact_duplicates() -> None:
    df = pd.DataFrame({"url": ["https://a.com", "https://a.com", "https://b.com"], "status": [0, 0, 1]})
    out = deduplicate_urls(df)
    assert len(out) == 2


def test_merge_removes_domain_overlap() -> None:
    kag = pd.DataFrame(
        {
            "url": ["https://a.com/x", "https://b.com/y"],
            "status": [0, 1],
            "label": [1, 0],
            "registered_domain": ["a.com", "b.com"],
            "source": ["kaggle", "kaggle"],
        }
    )
    fresh = pd.DataFrame(
        {
            "url": ["https://a.com/new", "https://c.com/new"],
            "status": [0, 1],
            "label": [1, 0],
            "registered_domain": ["a.com", "c.com"],
            "source": ["fresh", "fresh"],
        }
    )
    merged = merge_datasets(kag, fresh, fresh_weight=1.0)
    assert "a.com" in set(merged["registered_domain"])
    assert len(merged[merged["registered_domain"] == "a.com"]) == 1
    assert "c.com" in set(merged["registered_domain"])


def test_kaggle_only_merge_stats_non_negative_fresh_used() -> None:
    kag = pd.DataFrame(
        {
            "url": ["https://a.com/x", "https://b.com/y"],
            "status": [0, 1],
            "label": [1, 0],
            "registered_domain": ["a.com", "b.com"],
            "source": ["kaggle", "kaggle"],
        }
    )
    fresh = pd.DataFrame(columns=kag.columns)
    _, stats = merge_datasets(kag, fresh, fresh_weight=0.3, return_stats=True)
    assert stats["kaggle_rows"] == 2
    assert stats["fresh_rows_available"] == 0
    assert stats["fresh_rows_used"] == 0
    assert stats["combined_rows"] >= 0


def test_split_by_domain_zero_leakage() -> None:
    df = pd.DataFrame(
        {
            "canonical_url": [
                "https://a.com/1",
                "https://a.com/2",
                "https://b.com/1",
                "https://b.com/2",
                "https://c.com/1",
                "https://c.com/2",
            ],
            "label": [1, 1, 0, 0, 1, 0],
        }
    )
    tr, te, _ = stratified_group_train_test(df, test_size=0.5, random_state=42)
    tr_domains = set(tr["canonical_url"].str.extract(r"https?://([^/]+)")[0].tolist())
    te_domains = set(te["canonical_url"].str.extract(r"https?://([^/]+)")[0].tolist())
    assert not (tr_domains & te_domains)


def test_fresh_labels_not_inverted() -> None:
    fresh = pd.DataFrame({"url": ["https://x.com", "https://y.com"], "status": [0, 1]})
    fresh["label"] = fresh["status"].map(lambda s: 1 if int(s) == 0 else 0)
    assert fresh.loc[0, "label"] == 1
    assert fresh.loc[1, "label"] == 0


def test_ensure_split_compatible_adds_canonical_url(tmp_path: Path) -> None:
    p = tmp_path / "enriched.csv"
    pd.DataFrame({"url": ["https://a.com"], "label": [1], "status": [0]}).to_csv(p, index=False)
    out = ensure_split_compatible(p)
    df = pd.read_csv(out, dtype=str, low_memory=False)
    assert "canonical_url" in df.columns
    assert df.loc[0, "canonical_url"] == "https://a.com"


def test_ensure_split_compatible_keeps_existing_canonical_url(tmp_path: Path) -> None:
    p = tmp_path / "enriched.csv"
    pd.DataFrame(
        {"canonical_url": ["https://canonical.com"], "url": ["https://a.com"], "label": [1], "status": [0]}
    ).to_csv(p, index=False)
    out = ensure_split_compatible(p)
    df = pd.read_csv(out, dtype=str, low_memory=False)
    assert df.loc[0, "canonical_url"] == "https://canonical.com"


def test_ensure_enrich_compatible_adds_canonical_url() -> None:
    df = pd.DataFrame({"url": ["https://a.com"], "status": [0], "label": [1]})
    out = ensure_enrich_compatible(df)
    assert "canonical_url" in out.columns
    assert out.loc[0, "canonical_url"] == "https://a.com"


def test_retrain_pipeline_end_to_end_mocked(tmp_path: Path) -> None:
    kag = pd.DataFrame(
        {
            "url": [
                "https://k1.com/a",
                "https://k2.com/a",
                "https://k3.com/a",
                "https://k4.com/a",
                "https://k5.com/a",
                "https://k6.com/a",
                "https://k7.com/a",
                "https://k8.com/a",
                "https://k9.com/a",
                "https://k10.com/a",
            ],
            "status": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "registered_domain": [
                "k1.com",
                "k2.com",
                "k3.com",
                "k4.com",
                "k5.com",
                "k6.com",
                "k7.com",
                "k8.com",
                "k9.com",
                "k10.com",
            ],
            "source": ["kaggle"] * 10,
        }
    )
    kag_path = tmp_path / "kaggle.csv"
    kag.to_csv(kag_path, index=False)

    fresh_train = pd.DataFrame(
        {
            "url": ["https://f1.com", "https://f2.com"],
            "status": [0, 1],
            "label": [1, 0],
            "registered_domain": ["f1.com", "f2.com"],
            "source": ["fresh", "fresh"],
            "collection_date": ["2026-01-01", "2026-01-01"],
        }
    )
    fresh_holdout = pd.DataFrame(columns=fresh_train.columns)

    def _fake_enrich(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        df["canonical_url"] = df["url"]
        df["url_length"] = df["url"].str.len()
        out_csv = kwargs.get("output_csv")
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)

    def _fake_train(train_csv, test_csv, **kwargs):  # type: ignore[no-untyped-def]
        from src.pipeline.paths import metrics_dir, models_dir, reports_dir

        metrics_dir().mkdir(parents=True, exist_ok=True)
        models_dir().mkdir(parents=True, exist_ok=True)
        reports_dir().mkdir(parents=True, exist_ok=True)
        (models_dir() / "logistic_regression.joblib").write_bytes(b"x")
        (metrics_dir() / "metrics.json").write_text(json.dumps([{"model": "logistic_regression", "f1": 0.8}]))

    with _patch_retrain_deploy_models_dir(tmp_path):
        with patch(
            "src.pipeline.retrain_with_fresh.build_fresh_dataset",
            return_value=(
                fresh_train,
                fresh_holdout,
                {
                    "phishstats_rows_collected": 2,
                    "phishstats_errors_count": 0,
                    "tranco_rows_collected": 2,
                    "tranco_download_failed": False,
                    "tranco_error": None,
                },
            ),
        ):
            with patch("src.pipeline.retrain_with_fresh.enrich", side_effect=_fake_enrich):
                with patch("src.pipeline.retrain_with_fresh.train", side_effect=_fake_train):
                    summary = retrain_with_fresh(
                        kaggle_path=kag_path,
                        use_fresh_data=True,
                        fresh_weight=0.5,
                        overwrite_model=False,
                    )
    assert "dataset_size" in summary
    assert summary["dataset_size"]["kaggle_rows"] == 10


def test_retrain_kaggle_only_fresh_used_zero(tmp_path: Path) -> None:
    kag = pd.DataFrame(
        {
            "url": ["https://k1.com/a", "https://k2.com/a", "https://k3.com/a", "https://k4.com/a"],
            "status": [0, 1, 0, 1],
            "label": [1, 0, 1, 0],
            "registered_domain": ["k1.com", "k2.com", "k3.com", "k4.com"],
            "source": ["kaggle"] * 4,
        }
    )
    kag_path = tmp_path / "kaggle.csv"
    kag.to_csv(kag_path, index=False)

    def _fake_enrich(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        if "canonical_url" not in df.columns:
            df["canonical_url"] = df["url"]
        df["url_length"] = df["url"].str.len()
        out_csv = kwargs.get("output_csv")
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)

    def _fake_split(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        tr = kwargs["train_out"]
        te = kwargs["test_out"]
        df.head(2).to_csv(tr, index=False)
        df.tail(2).to_csv(te, index=False)
        return tr, te

    def _fake_train(train_csv, test_csv, **kwargs):  # type: ignore[no-untyped-def]
        from src.pipeline.paths import metrics_dir, models_dir, reports_dir

        metrics_dir().mkdir(parents=True, exist_ok=True)
        models_dir().mkdir(parents=True, exist_ok=True)
        reports_dir().mkdir(parents=True, exist_ok=True)
        (models_dir() / "logistic_regression.joblib").write_bytes(b"x")
        (metrics_dir() / "metrics.json").write_text(json.dumps([{"model": "logistic_regression", "f1": 0.7}]))

    with _patch_retrain_deploy_models_dir(tmp_path):
        with patch("src.pipeline.retrain_with_fresh.enrich", side_effect=_fake_enrich):
            with patch("src.pipeline.retrain_with_fresh.split_leak_safe", side_effect=_fake_split):
                with patch("src.pipeline.retrain_with_fresh.train", side_effect=_fake_train):
                    summary = retrain_with_fresh(
                        kaggle_path=kag_path,
                        use_fresh_data=False,
                        overwrite_model=False,
                    )
    assert summary["dataset_size"]["fresh_rows_available"] == 0
    assert summary["dataset_size"]["fresh_rows_used"] == 0
    assert summary.get("witness_models_saved") == ["logistic_regression.joblib"]


def test_retrain_sample_rows_marks_debug_mode(tmp_path: Path) -> None:
    kag = pd.DataFrame(
        {
            "url": [f"https://k{i}.com/a" for i in range(1, 13)],
            "status": [0, 1] * 6,
            "label": [1, 0] * 6,
            "registered_domain": [f"k{i}.com" for i in range(1, 13)],
            "source": ["kaggle"] * 12,
        }
    )
    kag_path = tmp_path / "kaggle.csv"
    kag.to_csv(kag_path, index=False)

    def _fake_enrich(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        if "canonical_url" not in df.columns:
            df["canonical_url"] = df["url"]
        df["url_length"] = df["url"].str.len()
        out_csv = kwargs.get("output_csv")
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)

    def _fake_split(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        tr = kwargs["train_out"]
        te = kwargs["test_out"]
        df.head(max(1, len(df) // 2)).to_csv(tr, index=False)
        df.tail(max(1, len(df) // 2)).to_csv(te, index=False)
        return tr, te

    def _fake_train(train_csv, test_csv, **kwargs):  # type: ignore[no-untyped-def]
        from src.pipeline.paths import metrics_dir, models_dir, reports_dir

        metrics_dir().mkdir(parents=True, exist_ok=True)
        models_dir().mkdir(parents=True, exist_ok=True)
        reports_dir().mkdir(parents=True, exist_ok=True)
        (models_dir() / "logistic_regression.joblib").write_bytes(b"x")
        (metrics_dir() / "metrics.json").write_text(json.dumps([{"model": "logistic_regression", "f1": 0.7}]))

    with _patch_retrain_deploy_models_dir(tmp_path):
        with patch("src.pipeline.retrain_with_fresh.enrich", side_effect=_fake_enrich):
            with patch("src.pipeline.retrain_with_fresh.split_leak_safe", side_effect=_fake_split):
                with patch("src.pipeline.retrain_with_fresh.train", side_effect=_fake_train):
                    summary = retrain_with_fresh(
                        kaggle_path=kag_path,
                        use_fresh_data=False,
                        sample_rows=6,
                        overwrite_model=False,
                    )
    assert summary["debug_sample_mode"] is True
    assert summary["sample_rows_requested"] == 6
    assert summary["sample_random_state"] == 42
    assert summary["dataset_size"]["merged_rows"] == 6


def test_use_fresh_data_failed_collectors_sets_effective_false_and_warning(tmp_path: Path) -> None:
    kag = pd.DataFrame(
        {
            "url": [f"https://k{i}.com/a" for i in range(1, 7)],
            "status": [0, 1, 0, 1, 0, 1],
            "label": [1, 0, 1, 0, 1, 0],
            "registered_domain": [f"k{i}.com" for i in range(1, 7)],
            "source": ["kaggle"] * 6,
        }
    )
    kag_path = tmp_path / "kaggle.csv"
    kag.to_csv(kag_path, index=False)

    def _fake_enrich(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        if "canonical_url" not in df.columns:
            df["canonical_url"] = df["url"]
        df["url_length"] = df["url"].str.len()
        out_csv = kwargs.get("output_csv")
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)

    def _fake_split(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        tr = kwargs["train_out"]
        te = kwargs["test_out"]
        df.head(3).to_csv(tr, index=False)
        df.tail(3).to_csv(te, index=False)
        from src.pipeline.paths import reports_dir

        (reports_dir() / "split_leak_safe_stats.json").write_text(
            json.dumps({"registered_domain_overlap_count": 0, "canonical_url_overlap_count": 0})
        )
        return tr, te

    def _fake_train(train_csv, test_csv, **kwargs):  # type: ignore[no-untyped-def]
        from src.pipeline.paths import metrics_dir, models_dir, reports_dir

        metrics_dir().mkdir(parents=True, exist_ok=True)
        models_dir().mkdir(parents=True, exist_ok=True)
        reports_dir().mkdir(parents=True, exist_ok=True)
        (models_dir() / "logistic_regression.joblib").write_bytes(b"x")
        (metrics_dir() / "metrics.json").write_text(json.dumps([{"model": "logistic_regression", "f1": 0.7}]))

    with _patch_retrain_deploy_models_dir(tmp_path):
        with patch(
            "src.pipeline.retrain_with_fresh.build_fresh_dataset",
            return_value=(
                pd.DataFrame(columns=["url", "status", "label", "registered_domain", "source"]),
                pd.DataFrame(columns=["url", "status", "label", "registered_domain", "source"]),
                {
                    "phishstats_rows_collected": 0,
                    "phishstats_errors_count": 4,
                    "tranco_rows_collected": 0,
                    "tranco_download_failed": True,
                    "tranco_error": "http_status_404",
                },
            ),
        ):
            with patch("src.pipeline.retrain_with_fresh.enrich", side_effect=_fake_enrich):
                with patch("src.pipeline.retrain_with_fresh.split_leak_safe", side_effect=_fake_split):
                    with patch("src.pipeline.retrain_with_fresh.train", side_effect=_fake_train):
                        summary = retrain_with_fresh(
                            kaggle_path=kag_path,
                            use_fresh_data=True,
                            overwrite_model=False,
                        )
    assert summary["fresh_data_effective"] is False
    assert summary["tranco_download_failed"] is True
    assert summary["tranco_error"] == "http_status_404"
    assert any("no fresh rows were used" in w.lower() for w in summary["warnings"])


def test_train_test_domain_overlap_uses_split_stats(tmp_path: Path) -> None:
    kag = pd.DataFrame(
        {
            "url": [f"https://k{i}.com/a" for i in range(1, 7)],
            "status": [0, 1, 0, 1, 0, 1],
            "label": [1, 0, 1, 0, 1, 0],
            "registered_domain": [f"k{i}.com" for i in range(1, 7)],
            "source": ["kaggle"] * 6,
        }
    )
    kag_path = tmp_path / "kaggle.csv"
    kag.to_csv(kag_path, index=False)

    def _fake_enrich(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        if "canonical_url" not in df.columns:
            df["canonical_url"] = df["url"]
        df["url_length"] = df["url"].str.len()
        out_csv = kwargs.get("output_csv")
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)

    def _fake_split(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        tr = kwargs["train_out"]
        te = kwargs["test_out"]
        df.head(3).to_csv(tr, index=False)
        df.tail(3).to_csv(te, index=False)
        from src.pipeline.paths import reports_dir

        (reports_dir() / "split_leak_safe_stats.json").write_text(
            json.dumps({"registered_domain_overlap_count": 0, "canonical_url_overlap_count": 0})
        )
        return tr, te

    def _fake_train(train_csv, test_csv, **kwargs):  # type: ignore[no-untyped-def]
        from src.pipeline.paths import metrics_dir, models_dir, reports_dir

        metrics_dir().mkdir(parents=True, exist_ok=True)
        models_dir().mkdir(parents=True, exist_ok=True)
        reports_dir().mkdir(parents=True, exist_ok=True)
        (models_dir() / "logistic_regression.joblib").write_bytes(b"x")
        (metrics_dir() / "metrics.json").write_text(json.dumps([{"model": "logistic_regression", "f1": 0.7}]))

    with _patch_retrain_deploy_models_dir(tmp_path):
        with patch("src.pipeline.retrain_with_fresh.enrich", side_effect=_fake_enrich):
            with patch("src.pipeline.retrain_with_fresh.split_leak_safe", side_effect=_fake_split):
                with patch("src.pipeline.retrain_with_fresh.train", side_effect=_fake_train):
                    summary = retrain_with_fresh(
                        kaggle_path=kag_path,
                        use_fresh_data=False,
                        overwrite_model=False,
                    )
    assert summary["leakage_counts"]["train_test_domain_overlap"] == 0


def test_collect_tranco_records_404_failure() -> None:
    class _Resp:
        status_code = 404
        text = ""
        headers = {}

    with patch("src.pipeline.fresh_data.requests.get", return_value=_Resp()):
        df, meta = collect_tranco(n=10, return_meta=True)
    assert df.empty
    assert meta["tranco_download_failed"] is True
    assert "404" in str(meta["tranco_error"])


def test_sample_mode_fresh_preserve_includes_all_fresh_no_duplicates_and_labels(tmp_path: Path) -> None:
    kag = pd.DataFrame(
        {
            "url": [f"https://k{i}.com/a" for i in range(1, 101)],
            "status": [0, 1] * 50,
            "label": [1, 0] * 50,
            "registered_domain": [f"k{i}.com" for i in range(1, 101)],
            "source": ["kaggle"] * 100,
        }
    )
    fresh_train = pd.DataFrame(
        {
            "url": [f"https://fresh{i}.com/a" for i in range(1, 6)],
            "status": [0, 1, 0, 1, 0],
            "label": [1, 0, 1, 0, 1],
            "registered_domain": [f"fresh{i}.com" for i in range(1, 6)],
            "source": ["phishstats", "tranco", "phishstats", "tranco", "phishstats"],
            "collection_date": ["2026-01-01"] * 5,
        }
    )
    fresh_holdout = pd.DataFrame(columns=fresh_train.columns)
    kag_path = tmp_path / "kaggle.csv"
    kag.to_csv(kag_path, index=False)

    sampled_rows: dict = {}

    def _fake_enrich(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        sampled_rows["df"] = df.copy()
        if "canonical_url" not in df.columns:
            df["canonical_url"] = df["url"]
        df["url_length"] = df["url"].str.len()
        out_csv = kwargs.get("output_csv")
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)

    def _fake_split(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        tr = kwargs["train_out"]
        te = kwargs["test_out"]
        df.head(max(1, len(df) // 2)).to_csv(tr, index=False)
        df.tail(max(1, len(df) // 2)).to_csv(te, index=False)
        from src.pipeline.paths import reports_dir

        (reports_dir() / "split_leak_safe_stats.json").write_text(
            json.dumps({"registered_domain_overlap_count": 0, "canonical_url_overlap_count": 0})
        )
        return tr, te

    def _fake_train(train_csv, test_csv, **kwargs):  # type: ignore[no-untyped-def]
        from src.pipeline.paths import metrics_dir, models_dir, reports_dir

        metrics_dir().mkdir(parents=True, exist_ok=True)
        models_dir().mkdir(parents=True, exist_ok=True)
        reports_dir().mkdir(parents=True, exist_ok=True)
        (models_dir() / "logistic_regression.joblib").write_bytes(b"x")
        (metrics_dir() / "metrics.json").write_text(json.dumps([{"model": "logistic_regression", "f1": 0.7}]))

    with _patch_retrain_deploy_models_dir(tmp_path):
        with patch(
            "src.pipeline.retrain_with_fresh.build_fresh_dataset",
            return_value=(
                fresh_train,
                fresh_holdout,
                {
                    "phishstats_rows_collected": 3,
                    "phishstats_errors_count": 0,
                    "tranco_rows_collected": 2,
                    "tranco_download_failed": False,
                    "tranco_error": None,
                },
            ),
        ):
            with patch("src.pipeline.retrain_with_fresh.enrich", side_effect=_fake_enrich):
                with patch("src.pipeline.retrain_with_fresh.split_leak_safe", side_effect=_fake_split):
                    with patch("src.pipeline.retrain_with_fresh.train", side_effect=_fake_train):
                        summary = retrain_with_fresh(
                            kaggle_path=kag_path,
                            use_fresh_data=True,
                            sample_rows=50,
                            fresh_preserve_in_sample=True,
                            overwrite_model=False,
                        )

    sampled_df = sampled_rows["df"]
    fresh_in_sample = sampled_df[sampled_df["source"].astype(str).str.lower() != "kaggle"]
    assert len(fresh_in_sample) == 5
    assert sampled_df["url"].nunique() == len(sampled_df)
    st = pd.to_numeric(sampled_df["status"], errors="coerce")
    lb = pd.to_numeric(sampled_df["label"], errors="coerce")
    assert bool(((st == 0) == (lb == 1)).all())
    assert summary["fresh_rows_in_sample"] == 5
    assert summary["kaggle_rows_in_sample"] == len(sampled_df) - 5


def test_no_primary_overwrite_without_flag(tmp_path: Path) -> None:
    kag = pd.DataFrame(
        {
            "url": [f"https://k{i}.com/a" for i in range(1, 9)],
            "status": [0, 1] * 4,
            "label": [1, 0] * 4,
            "registered_domain": [f"k{i}.com" for i in range(1, 9)],
            "source": ["kaggle"] * 8,
        }
    )
    kag_path = tmp_path / "kaggle.csv"
    kag.to_csv(kag_path, index=False)
    custom_models_dir = tmp_path / "models_out"
    custom_models_dir.mkdir(parents=True, exist_ok=True)
    sentinel = custom_models_dir / "layer1_primary.joblib"
    sentinel.write_bytes(b"sentinel")

    def _fake_enrich(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        if "canonical_url" not in df.columns:
            df["canonical_url"] = df["url"]
        df["url_length"] = df["url"].str.len()
        out_csv = kwargs.get("output_csv")
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)

    def _fake_split(input_csv, **kwargs):  # type: ignore[no-untyped-def]
        df = pd.read_csv(input_csv, dtype=str, low_memory=False)
        tr = kwargs["train_out"]
        te = kwargs["test_out"]
        df.head(4).to_csv(tr, index=False)
        df.tail(4).to_csv(te, index=False)
        from src.pipeline.paths import reports_dir

        (reports_dir() / "split_leak_safe_stats.json").write_text(
            json.dumps({"registered_domain_overlap_count": 0, "canonical_url_overlap_count": 0})
        )
        return tr, te

    def _fake_train(train_csv, test_csv, **kwargs):  # type: ignore[no-untyped-def]
        from src.pipeline.paths import metrics_dir, models_dir, reports_dir

        metrics_dir().mkdir(parents=True, exist_ok=True)
        models_dir().mkdir(parents=True, exist_ok=True)
        reports_dir().mkdir(parents=True, exist_ok=True)
        (models_dir() / "logistic_regression.joblib").write_bytes(b"x")
        (metrics_dir() / "metrics.json").write_text(json.dumps([{"model": "logistic_regression", "f1": 0.7}]))

    with patch("src.pipeline.retrain_with_fresh.models_dir", return_value=custom_models_dir):
        with patch("src.pipeline.retrain_with_fresh.enrich", side_effect=_fake_enrich):
            with patch("src.pipeline.retrain_with_fresh.split_leak_safe", side_effect=_fake_split):
                with patch("src.pipeline.retrain_with_fresh.train", side_effect=_fake_train):
                    retrain_with_fresh(
                        kaggle_path=kag_path,
                        use_fresh_data=False,
                        overwrite_model=False,
                    )

    assert sentinel.read_bytes() == b"sentinel"
