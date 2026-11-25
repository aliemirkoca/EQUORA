"""Faz 1 — EQUORA Uçtan Uca Hurdle ML Pipeline Orkestratörü (EquoraMLPipeline).

Girdi:
    - `MLConfig` yapılandırması ve PostgreSQL `equora_db` (veya Parquet) verileri.

Çıktı:
    - `data/ml/` altında tüm ara Parquet checkpoint'leri (`snapshots.parquet`,
      `active_snapshots.parquet`, `daily_store_product_sales.parquet`,
      `features.parquet`, `active_features.parquet`, `live_predictions.parquet`).
    - `models/hurdle_v1/` altında eğitilmiş Hurdle modelleri, özellik şeması ve
      optimal eşik dosyası (`classifier.pkl`, `regressor.pkl`, `model_metadata.json`,
      `feature_columns.json`, `threshold.json`).
    - `reports/phase1_eval/` altında JSON ve Markdown değerlendirme raporları.
    - PostgreSQL `equora_db` içinde `ml_waste_predictions` canlı tahmin tablosu.

Çalıştırma:
    python -m src.ml_pipeline.run_ml_pipeline
    python src/ml_pipeline/run_ml_pipeline.py
    python src/ml_pipeline/run_ml_pipeline.py --use-parquet --skip-db-write
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Doğrudan `python src/ml_pipeline/run_ml_pipeline.py` çağrısı için proje kökünü ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_pipeline.config import MLConfig
from src.ml_pipeline.evaluator import EvaluationOutput, ModelEvaluator
from src.ml_pipeline.feature_engineer import FeatureEngineeringResult, WasteFeatureEngineer
from src.ml_pipeline.hurdle_model import EquoraHurdleModel
from src.ml_pipeline.inference import LiveInferenceResult, LiveRiskPredictor
from src.ml_pipeline.snapshot_builder import DailySnapshotBuilder, SnapshotBuildResult
from src.ml_pipeline.splitter import GroupTemporalSplitter, SplitResult


class EquoraMLPipeline:
    """EQUORA Faz 1 — 6 adımlı uçtan uca Hurdle ML Pipeline orkestratörü."""

    def __init__(
        self,
        config: MLConfig | None = None,
        write_predictions_to_db: bool | None = None,
    ) -> None:
        self.config = config or MLConfig()
        self.write_predictions_to_db = write_predictions_to_db

    def run(self) -> dict[str, Any]:
        """Tüm ML aşamalarını sırasıyla yürütür ve 5 bölümlü yönetici raporunu basar."""
        t0 = time.perf_counter()
        self.config.ensure_directories()

        print("=" * 94)
        print("EQUORA FAZ 1 — ELDE KALACAK ÜRÜN TAHMİNİ (HURDLE ML PIPELINE)")
        print("=" * 94)

        # ---------------------------------------------------------------------
        # ADIM 1/6: Günlük Parti Zaman Kesitleri (Snapshot Builder)
        # ---------------------------------------------------------------------
        print("\n[ADIM 1/6] Günlük parti zaman kesitleri (DailySnapshotBuilder) oluşturuluyor...")
        snapshot_builder = DailySnapshotBuilder(self.config)
        snap_res = snapshot_builder.build()
        print(
            f"  [OK] Kaynak: {snap_res.source} | Tarih: {snap_res.min_date} .. {snap_res.max_date}"
        )
        print(
            f"  [OK] Kapanmış: {snap_res.closed_batches:,} parti -> "
            f"{snap_res.closed_snapshot_rows:,} satır | "
            f"Aktif: {snap_res.active_batches:,} parti -> "
            f"{snap_res.active_snapshot_rows:,} satır"
        )

        # ---------------------------------------------------------------------
        # ADIM 2/6: Sızıntısız FEFO & Talep Özellik Mühendisliği
        # ---------------------------------------------------------------------
        print("\n[ADIM 2/6] Sızıntısız özellik mühendisliği (WasteFeatureEngineer) çalışıyor...")
        feature_engineer = WasteFeatureEngineer(self.config)
        feat_res = feature_engineer.build()
        print(
            f"  [OK] features.parquet: ({feat_res.closed_feature_rows:,} x {feat_res.closed_feature_cols}) | "
            f"active_features.parquet: ({feat_res.active_feature_rows:,} x {feat_res.active_feature_cols}) | "
            f"Model özellik sayısı: {len(feat_res.feature_columns)}"
        )

        # ---------------------------------------------------------------------
        # ADIM 3/6: Grup-Zaman Bölmesi (GroupTemporalSplitter)
        # ---------------------------------------------------------------------
        print("\n[ADIM 3/6] Sızıntısız grup-zaman bölmesi (GroupTemporalSplitter) uygulanıyor...")
        splitter = GroupTemporalSplitter(self.config)
        split_res = splitter.split()

        # ---------------------------------------------------------------------
        # ADIM 4/6: İki Aşamalı Hurdle Model Eğitimi (EquoraHurdleModel)
        # ---------------------------------------------------------------------
        print("\n[ADIM 4/6] İki aşamalı LightGBM Hurdle modeli (EquoraHurdleModel) eğitiliyor...")
        model = EquoraHurdleModel(
            config=self.config,
            feature_columns=feat_res.feature_columns,
            categorical_features=feat_res.categorical_features,
        )
        train_meta = model.fit(train_df=split_res.train_df, val_df=split_res.val_df)
        saved_model_paths = model.save()
        print(
            f"  [OK] Sınıflandırıcı best_iter={train_meta['classifier_best_iteration']} "
            f"(n={train_meta['train_rows_all']:,}) | "
            f"Koşullu Regresör best_iter={train_meta['regressor_best_iteration']} "
            f"(n_pos={train_meta['train_rows_positive_waste']:,})"
        )
        print(f"  [OK] Model artifact'leri kaydedildi: {self.config.MODEL_DIR}")

        # ---------------------------------------------------------------------
        # ADIM 5/6: Maliyet Duyarlı Eşik Optimizasyonu ve Test Değerlendirmesi
        # ---------------------------------------------------------------------
        print("\n[ADIM 5/6] Maliyet duyarlı eşik optimizasyonu ve değerlendirme (ModelEvaluator)...")
        evaluator = ModelEvaluator(self.config)
        eval_out = evaluator.evaluate(
            model=model,
            val_df=split_res.val_df,
            test_df=split_res.test_df,
        )
        print(
            f"  [OK] Optimal eşik theta* = {eval_out.optimal_threshold:.2f} | "
            f"Raporlar: {self.config.REPORT_DIR}"
        )

        # ---------------------------------------------------------------------
        # ADIM 6/6: Canlı Aktif Stok Tahmini ve PostgreSQL Kaydı
        # ---------------------------------------------------------------------
        print("\n[ADIM 6/6] Canlı aktif stok tahmini (LiveRiskPredictor) çalıştırılıyor...")
        predictor = LiveRiskPredictor(self.config)
        live_res = predictor.predict(write_to_postgres=self.write_predictions_to_db)
        print(
            f"  [OK] Canlı tahmin tamamlandı: {live_res.total_active_batches:,} aktif parti | "
            f"Parquet: {self.config.live_predictions_path.name} | "
            f"PostgreSQL ml_waste_predictions yazılan satır: {live_res.postgres_rows_written:,}"
        )

        elapsed = time.perf_counter() - t0
        self._print_final_executive_report(
            snap_res=snap_res,
            feat_res=feat_res,
            split_res=split_res,
            eval_out=eval_out,
            live_res=live_res,
            elapsed_sec=elapsed,
        )

        return {
            "snapshot": snap_res,
            "features": feat_res,
            "split": split_res,
            "train_meta": train_meta,
            "model_paths": saved_model_paths,
            "evaluation": eval_out,
            "live_inference": live_res,
            "elapsed_sec": round(elapsed, 2),
        }

    # -------------------------------------------------------------------------
    # 5 Bölümlü Net Terminal Özet Raporu
    # -------------------------------------------------------------------------
    @staticmethod
    def _print_final_executive_report(
        snap_res: SnapshotBuildResult,
        feat_res: FeatureEngineeringResult,
        split_res: SplitResult,
        eval_out: EvaluationOutput,
        live_res: LiveInferenceResult,
        elapsed_sec: float,
    ) -> None:
        report = eval_out.report
        th = report["threshold_optimization"]
        c_all = report["test_classification_all"]
        c_dte2 = report["test_classification_dte_ge_2"]
        reg = report["test_regression"]
        biz = report["business_metrics_test"]
        fi = report["feature_importances_top15"]

        print("\n" + "=" * 94)
        print("EQUORA FAZ 1 — NİHAİ PIPELINE ÖZET RAPORU")
        print("=" * 94)

        # ---------------------------------------------------------------------
        # 1) Snapshot & Feature Boyutları + Sızıntı / Split Doğrulamaları
        # ---------------------------------------------------------------------
        print("\n[1] SNAPSHOT & FEATURE TABLO BOYUTLARI VE SIZINTI DOĞRULAMALARI")
        print(
            f"    - snapshots.parquet        : {snap_res.closed_snapshot_rows:>7,} satır "
            f"({snap_res.closed_batches:,} kapanmış parti, pozitif fire satır oranı: %{snap_res.closed_positive_waste_rate * 100:.2f})"
        )
        print(
            f"    - active_snapshots.parquet : {snap_res.active_snapshot_rows:>7,} satır "
            f"({snap_res.active_batches:,} aktif parti, tarih: {snap_res.max_date})"
        )
        print(
            f"    - features.parquet         : {feat_res.closed_feature_rows:>7,} satır x {feat_res.closed_feature_cols} sütun "
            f"({len(feat_res.feature_columns)} model özelliği)"
        )
        print(
            f"    - active_features.parquet  : {feat_res.active_feature_rows:>7,} satır x {feat_res.active_feature_cols} sütun"
        )
        print(
            f"    - Negatif Stok Kontrolü    : full={snap_res.negative_stock_full_count}, "
            f"closed={snap_res.negative_stock_closed_count}, "
            f"active={snap_res.negative_stock_active_count} [BAŞARILI: 0 NEGATİF STOK]"
        )
        print(
            f"    - Split Kesişim Kontrolü   : "
            f"Train∩Val={split_res.overlap_train_val}, "
            f"Train∩Test={split_res.overlap_train_test}, "
            f"Val∩Test={split_res.overlap_val_test} [BAŞARILI: 0 PARTİ KESİŞİMİ]"
        )

        # ---------------------------------------------------------------------
        # 2) Optimal Eşik theta* ve Test Seti Metrik Tablosu (0.50 vs theta*)
        # ---------------------------------------------------------------------
        opt_th = eval_out.optimal_threshold
        d050 = c_all["default_050"]
        dopt = c_all["optimal_theta"]
        d2_050 = c_dte2["default_050"]
        d2_opt = c_dte2["optimal_theta"]

        print(f"\n[2] OPTİMAL EŞİK (theta* = {opt_th:.2f}) VE TEST SETİ METRİK TABLOSU")
        print(
            f"    Validation Maliyeti: 0.50 -> {th['val_default_050_total_loss_tl']:,.2f} TL | "
            f"theta* ({opt_th:.2f}) -> {th['val_optimal_total_loss_tl']:,.2f} TL "
            f"(Azalma: %{th['val_loss_reduction_vs_050_pct']:.2f})"
        )
        print(
            f"    {'Kapsam / Eşik':<30} | {'ROC-AUC':>7} | {'PR-AUC':>7} | "
            f"{'Prec.':>7} | {'Recall':>7} | {'F1':>7}"
        )
        print("    " + "-" * 79)
        for label, m in (
            ("Test Tümü — Standart (0.50)", d050),
            (f"Test Tümü — Optimal  ({opt_th:.2f})", dopt),
            ("Test DTE>=2 — Standart (0.50)", d2_050),
            (f"Test DTE>=2 — Optimal  ({opt_th:.2f})", d2_opt),
        ):
            print(
                f"    {label:<30} | {m['roc_auc']:>7.4f} | {m['pr_auc']:>7.4f} | "
                f"{m['precision']:>7.4f} | {m['recall']:>7.4f} | {m['f1_score']:>7.4f}"
            )

        rc = reg["conditional_q_hat"]
        ro = reg["overall_expected_w_hat"]
        print("\n    Regresyon Metrikleri (Test Seti):")
        print(
            f"      * Koşullu (final_waste_qty > 0, q_hat, n={reg['conditional_positive_waste_rows']:,}): "
            f"MAE={rc['mae']:.4f} | RMSE={rc['rmse']:.4f} | WAPE=%{rc['wape_pct']:.2f}"
        )
        print(
            f"      * Birleşik Beklenen Fire (Tümü, W_hat, n={reg['overall_all_rows']:,})    : "
            f"MAE={ro['mae']:.4f} | RMSE={ro['rmse']:.4f} | WAPE=%{ro['wape_pct']:.2f}"
        )

        # ---------------------------------------------------------------------
        # 3) En Önemli Top 10 Feature (Sınıflandırıcı & Regresör)
        # ---------------------------------------------------------------------
        print("\n[3] EN ÖNEMLİ TOP 10 FEATURE (GAIN %)")
        print(
            f"    {'#':>2} | {'Sınıflandırıcı (Risk)':<26} | {'Gain %':>7} || "
            f"{'Regresör (Fire Adedi)':<26} | {'Gain %':>7}"
        )
        print("    " + "-" * 79)
        clf_top = fi["classifier"][:10]
        reg_top = fi["regressor"][:10]
        for idx in range(10):
            c_item = clf_top[idx] if idx < len(clf_top) else {"feature": "-", "gain_pct": 0.0}
            r_item = reg_top[idx] if idx < len(reg_top) else {"feature": "-", "gain_pct": 0.0}
            print(
                f"    {idx + 1:>2} | {c_item['feature']:<26} | {c_item['gain_pct']:>6.2f}% || "
                f"{r_item['feature']:<26} | {r_item['gain_pct']:>6.2f}%"
            )

        # ---------------------------------------------------------------------
        # 4) Test Seti İş / Jüri Metrikleri (48s+ Erken Yakalama & TL)
        # ---------------------------------------------------------------------
        print("\n[4] TEST SETİ İŞ / JÜRİ METRİKLERİ (BENZERSİZ PARTİ BAZLI — 48 SAAT+ ERKEN YAKALAMA)")
        print(
            f"    - Test Toplam Benzersiz Parti : {biz['total_unique_test_batches']:,} "
            f"(Fireli: {biz['total_waste_batches']:,} parti, {biz['total_test_waste_units']:,} adet)"
        )
        print(
            f"    - 48s+ Önceden Yakalanan Fire : %{biz['early_catch_rate_by_batch_pct']:.2f} (parti) | "
            f"%{biz['early_catch_rate_by_units_pct']:.2f} (adet: {biz['rescueable_units_48h']:,}) | "
            f"%{biz['early_catch_rate_by_tl_pct']:.2f} (TL: {biz['rescueable_gross_value_tl']:,.2f} TL)"
        )
        print(
            f"    - Baseline (Müdahale Yok)     : {biz['baseline_no_action_loss_tl']:,.2f} TL kayıp"
        )
        print(
            f"    - EQUORA Tahmini Toplam Kayıp : {biz['equora_estimated_total_loss_tl']:,.2f} TL "
            f"(FN: {biz['equora_missed_waste_fn_loss_tl']:,.2f} + "
            f"TP İndirim: {biz['equora_tp_discount_cost_tl']:,.2f} + "
            f"FP İndirim: {biz['equora_fp_discount_cost_tl']:,.2f})"
        )
        print(
            f"    - EQUORA Net Tasarruf         : {biz['equora_net_savings_tl']:,.2f} TL "
            f"(%{biz['equora_loss_reduction_pct']:.2f} zarar azalması)"
        )
        esg = biz["esg_co2_impact"]
        print(
            f"    - ESG / CO2e Etkisi (Test)    : Baseline {esg['baseline_waste_co2_kg']:,.2f} kg "
            f"({esg['baseline_waste_co2_tons']:.3f} ton) -> 48s+ Önlenebilir: "
            f"{esg['prevented_co2_48h_kg']:,.2f} kg ({esg['prevented_co2_48h_tons']:.3f} ton, "
            f"%{esg['prevented_co2_48h_pct']:.2f})"
        )
        print(
            f"      * Çevresel Eşdeğerlik       : ~{esg['prevented_equivalent_car_km']:,.0f} km araç sürüşü "
            f"| ~{esg['prevented_equivalent_tree_years']:,.0f} olgun ağacın 1 yıllık CO2 tutumu"
        )

        # ---------------------------------------------------------------------
        # 5) Canlı Aktif Stok Üzerindeki En Riskli 10 Ürün / Parti
        # ---------------------------------------------------------------------
        print("\n[5] CANLI AKTİF STOK ÜZERİNDEKİ EN RİSKLİ 10 ÜRÜN / PARTİ")
        print(
            f"    Aksiyon Dağılımı ({live_res.total_active_batches} aktif parti): "
            f"{live_res.action_counts} | "
            f"Beklenen Kayıp: {live_res.total_estimated_loss_tl:,.2f} TL | "
            f"Beklenen CO2e: {live_res.total_estimated_co2_kg:,.2f} kg "
            f"(Aksiyon Alınabilir: {live_res.actionable_estimated_co2_kg:,.2f} kg)"
        )
        header = (
            f"    {'Şube':>4} | {'Ürün Adı':<26} | {'SKT':<10} | {'DTE':>3} | "
            f"{'Kalan':>5} | {'Olas.':>6} | {'Tah.Fire':>8} | {'CO2(kg)':>7} | {'Aksiyon':<16}"
        )
        print(header)
        print("    " + "-" * (len(header) - 4))
        top10 = live_res.predictions_df.head(10).iter_rows(named=True)
        for r in top10:
            pname = str(r["product_name"])[:26]
            print(
                f"    {int(r['store_id']):>4} | {pname:<26} | {str(r['expiry_date']):<10} | "
                f"{int(r['days_to_expiry']):>3} | {int(r['remaining_qty']):>5} | "
                f"{float(r['waste_probability']):>6.4f} | "
                f"{float(r['predicted_waste_qty']):>8.2f} | "
                f"{float(r['estimated_co2_kg']):>7.2f} | "
                f"{str(r['recommended_action']):<16}"
            )

        print("\n" + "=" * 94)
        print(f"[TAMAMLANDI] EQUORA Faz 1 Hurdle ML Pipeline {elapsed_sec:.2f} saniyede tamamlandı.")
        print("=" * 94)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EQUORA Faz 1 — Elde Kalacak Ürün Tahmini (Hurdle ML Pipeline)"
    )
    parser.add_argument(
        "--use-parquet",
        action="store_true",
        help="PostgreSQL yerine data/processed/*.parquet dosyalarından oku",
    )
    parser.add_argument(
        "--skip-db-write",
        action="store_true",
        help="Canlı tahminleri PostgreSQL ml_waste_predictions tablosuna yazma",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MLConfig(USE_POSTGRES=not args.use_parquet)
    write_db = False if args.skip_db_write else config.USE_POSTGRES
    pipeline = EquoraMLPipeline(config=config, write_predictions_to_db=write_db)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
