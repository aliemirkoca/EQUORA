"""Faz 2A — EQUORA RL Dinamik İndirim Uçtan Uca Orkestratörü (run_rl_pipeline.py).

Girdi:
    - Faz 1 çıktıları (`data/ml/features.parquet`, `active_features.parquet`,
      `live_predictions.parquet`, `models/hurdle_v1/*`) ve `RLPricingConfig`.

Çıktı:
    - Eğitilmiş RL modelleri (`models/rl_pricing_v1/maskable_ppo.zip`, `dqn_benchmark.zip`),
    - 6'lı politika yarışı & stres testi raporları (`reports/phase2a_rl_eval/*`),
    - Canlı indirim önerileri (`data/rl/live_discount_recommendations.parquet` +
      PostgreSQL `rl_discount_recommendations`).

Çalıştırma:
    python -m src.rl_pricing.run_rl_pipeline
    python -m src.rl_pricing.run_rl_pipeline --timesteps 200000 --stop-dte 2
    python -m src.rl_pricing.run_rl_pipeline --skip-db-write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gymnasium.utils.env_checker import check_env

from src.ml_pipeline.config import MLConfig
from src.ml_pipeline.hurdle_model import EquoraHurdleModel
from src.rl_pricing.config import RLPricingConfig
from src.rl_pricing.demand_model import CounterfactualDemandModel
from src.rl_pricing.discount_env import EquoraDiscountEnv
from src.rl_pricing.episode_builder import DiscountEpisodeBuilder
from src.rl_pricing.live_recommender import LiveDiscountRecommender
from src.rl_pricing.train_rl import RLPricingTrainer


class EquoraRLPipeline:
    """EQUORA Faz 2A — 4 adımlı RL dinamik indirim orkestratörü."""

    def __init__(
        self,
        rl_config: RLPricingConfig | None = None,
        ml_config: MLConfig | None = None,
        train_dqn: bool = True,
        write_to_db: bool | None = None,
    ) -> None:
        self.rl_config = rl_config or RLPricingConfig()
        self.ml_config = ml_config or MLConfig()
        self.train_dqn = train_dqn
        self.write_to_db = write_to_db

    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.rl_config.ensure_directories()

        print("=" * 96)
        print("EQUORA FAZ 2A — RL TABANLI DİNAMİK İNDİRİM MOTORU (MASKABLE PPO + BENCHMARKS)")
        print("=" * 96)

        # ---------------------------------------------------------------------
        # ADIM 1/4: Faz 1 Modelini Yükle & Kronolojik Epizot Havuzu Oluştur
        # ---------------------------------------------------------------------
        print("\n[ADIM 1/4] Faz 1 modeli yükleniyor ve kronolojik epizot havuzu üretiliyor...")
        hurdle_model = EquoraHurdleModel.load(self.ml_config)
        th_payload = json.loads(
            self.ml_config.threshold_path.read_text(encoding="utf-8")
        )
        phase1_theta = float(th_payload["optimal_threshold"])

        pool = DiscountEpisodeBuilder(self.rl_config, self.ml_config).build()
        demand_model = CounterfactualDemandModel(self.rl_config)
        print(
            f"  [OK] Epizot Havuzu: Train={len(pool.train_episodes):,} "
            f"({pool.train_date_range[0]}..{pool.train_date_range[1]}) | "
            f"Out-of-Time Test={len(pool.test_episodes):,} "
            f"({pool.test_date_range[0]}..{pool.test_date_range[1]}) | "
            f"Kesişim={pool.batch_overlap}"
        )
        print(
            f"  [OK] Faz 1 Eşik theta*={phase1_theta:.2f} | "
            f"Baz Esneklik={demand_model.profile.markdown_elasticity:.3f}"
        )

        # Gymnasium Ortam Doğrulaması (`check_env`)
        chk_env = EquoraDiscountEnv(
            episodes=pool.train_episodes[:50],
            config=self.rl_config,
            demand_model=demand_model,
            hurdle_model=hurdle_model,
            phase1_threshold=phase1_theta,
            seed=self.rl_config.SEED,
        )
        check_env(chk_env, skip_render_check=True)
        print("  [OK] Gymnasium `check_env` doğrulaması başarıyla geçti (0 NaN/Inf, bounds OK).")

        # ---------------------------------------------------------------------
        # ADIM 2/4: MaskablePPO (+ DQN) Eğitimi & Out-of-Time 6'lı Yarış + Stres Testi
        # ---------------------------------------------------------------------
        print("\n[ADIM 2/4] RL ajanları eğitiliyor ve Out-of-Time politika yarışı yürütülüyor...")
        trainer = RLPricingTrainer(self.rl_config, self.ml_config)
        eval_out = trainer.train_and_evaluate(
            pool=pool,
            hurdle_model=hurdle_model,
            phase1_threshold=phase1_theta,
            demand_model=demand_model,
            train_dqn=self.train_dqn,
        )
        print(
            f"  [OK] Değerlendirme raporları yazıldı: {self.rl_config.RL_REPORT_DIR} | "
            f"Monotonluk İhlali: {eval_out.report['total_monotonic_violations']}"
        )

        # ---------------------------------------------------------------------
        # ADIM 3/4: Canlı Aktif Partiler İçin Optimal İndirim Önerisi + PostgreSQL
        # ---------------------------------------------------------------------
        print("\n[ADIM 3/4] Canlı aktif stok için RL indirim önerileri üretiliyor...")
        recommender = LiveDiscountRecommender(self.rl_config, self.ml_config)
        live_rec = recommender.recommend(
            ppo_model=eval_out.ppo_model,
            hurdle_model=hurdle_model,
            demand_model=demand_model,
            write_to_postgres=self.write_to_db,
        )
        print(
            f"  [OK] Canlı öneri tamamlandı: {live_rec.total_active_batches:,} aktif parti "
            f"({live_rec.discounted_batches:,} partiye indirim önerildi) | "
            f"Parquet: {self.rl_config.live_recommendations_path.name} | "
            f"PostgreSQL rl_discount_recommendations yazılan satır: {live_rec.postgres_rows_written:,}"
        )

        # ---------------------------------------------------------------------
        # ADIM 4/4: Yönetici Özet Tablosu
        # ---------------------------------------------------------------------
        elapsed = time.perf_counter() - t0
        self._print_summary(eval_out.report, live_rec, elapsed)

        return {
            "pool": pool,
            "rl_eval": eval_out,
            "live_recommendations": live_rec,
            "elapsed_sec": round(elapsed, 2),
        }

    @staticmethod
    def _print_summary(
        report: dict[str, Any],
        live_rec: Any,
        elapsed_sec: float,
    ) -> None:
        print("\n" + "=" * 96)
        print("EQUORA FAZ 2A — RL DİNAMİK İNDİRİM ÖZET RAPORU")
        print("=" * 96)

        print("\n[1] OUT-OF-TIME TEST POLİTİKA YARIŞI (CRN — AYNI EPİZOTLAR)")
        hdr = (
            f"    {'Politika':<28} | {'Gelir (TL)':>11} | {'Artış':>7} | {'Komis.(TL)':>10} | "
            f"{'Net Eko.(TL)':>12} | {'Fire':>6} | {'Azalış':>7} | {'CO2(kg)':>8} | {'Ort.%':>6}"
        )
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for r in report["main_benchmark_test"]:
            print(
                f"    {r['policy_name']:<28} | {r['total_revenue_tl']:>11,.2f} | "
                f"{r['revenue_lift_vs_nodisc_pct']:>+6.2f}% | {r['total_commission_tl']:>10,.2f} | "
                f"{r['total_net_economic_tl']:>12,.2f} | {r['total_residual_waste_units']:>6,} | "
                f"{r['waste_reduction_vs_nodisc_pct']:>+6.2f}% | "
                f"{r['prevented_co2_vs_nodisc_kg']:>8,.1f} | {r['avg_discount_pct']:>5.1f}%"
            )

        sig = report.get("ppo_vs_nodiscount_significance", {})
        if sig:
            print("\n[2] İSTATİSTİKSEL ANLAMLILIK (MaskablePPO vs. NoDiscountPolicy)")
            for k, s in sig.items():
                print(
                    f"    - {k:<22}: ort. fark={s['mean_diff_per_episode']:+.3f} "
                    f"[{s['ci95_low']:+.3f}, {s['ci95_high']:+.3f}] | "
                    f"t={s['t_statistic']:.2f} | p={s['p_value']:.2e} "
                    f"({'ANLAMLI' if s['significant_at_001'] else 'ANLAMSIZ'})"
                )

        print("\n[3] HASSASİYET & STRES TESTLERİ (MaskablePPO vs. NoDiscountPolicy)")
        for sc in report["stress_tests"]:
            ppo_r = next(r for r in sc["results"] if r["policy_name"] == "MaskablePPO")
            print(
                f"    - {sc['title']}: "
                f"Gelir {ppo_r['revenue_lift_vs_nodisc_pct']:+.2f}% | "
                f"Net Eko {ppo_r['net_economic_lift_vs_nodisc_pct']:+.2f}% | "
                f"Fire {ppo_r['waste_reduction_vs_nodisc_pct']:+.2f}% | "
                f"Önlenen CO2 {ppo_r['prevented_co2_vs_nodisc_kg']:,.1f} kg | "
                f"Ort. İndirim %{ppo_r['avg_discount_pct']:.1f}"
            )

        print("\n[4] CANLI AKTİF STOK RL İNDİRİM ÖNERİLERİ (EN YÜKSEK ÖNCELİKLİ 10 PARTİ)")
        print(
            f"    İndirim Kademesi Dağılımı: {live_rec.discount_tier_counts} | "
            f"Beklenen Gelir: {live_rec.total_expected_revenue_tl:,.2f} TL | "
            f"EQUORA Komisyon: {live_rec.total_expected_commission_tl:,.2f} TL | "
            f"Faz 2B Artık Arz: {live_rec.total_expected_residual_qty:,.1f} adet | "
            f"Ek Önlenen CO2: {live_rec.total_expected_prevented_co2_kg:,.2f} kg"
        )
        lh = (
            f"    {'Şube':>4} | {'Ürün Adı':<26} | {'DTE':>3} | {'Kalan':>5} | "
            f"{'Olas.':>6} | {'İnd.%':>5} | {'Lift':>5} | {'Artık':>6} | {'Komis.TL':>8} | {'Devir':<11}"
        )
        print(lh)
        print("    " + "-" * (len(lh) - 4))
        for r in live_rec.recommendations_df.head(10).iter_rows(named=True):
            print(
                f"    {int(r['store_id']):>4} | {str(r['product_name'])[:26]:<26} | "
                f"{int(r['days_to_expiry']):>3} | {int(r['remaining_qty']):>5} | "
                f"{float(r['waste_probability']):>6.3f} | "
                f"{int(round(float(r['recommended_discount_rate']) * 100)):>4}% | "
                f"{float(r['expected_demand_lift']):>5.2f} | "
                f"{float(r['expected_residual_qty']):>6.1f} | "
                f"{float(r['expected_commission_tl']):>8.2f} | "
                f"{str(r['handoff_channel']):<11}"
            )

        print("\n" + "=" * 96)
        print(f"[TAMAMLANDI] EQUORA Faz 2A RL Pipeline {elapsed_sec:.2f} saniyede tamamlandı.")
        print("=" * 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EQUORA Faz 2A — RL Tabanlı Dinamik İndirim Motoru"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="MaskablePPO toplam eğitim adımı (varsayılan: 200000)",
    )
    parser.add_argument(
        "--dqn-timesteps",
        type=int,
        default=120_000,
        help="DQN benchmark toplam eğitim adımı (varsayılan: 120000)",
    )
    parser.add_argument(
        "--stop-dte",
        type=int,
        default=2,
        help="İndirim penceresinin bittiği DTE (2: Faz 2B devri, 0: SKT gün sonu)",
    )
    parser.add_argument(
        "--skip-dqn",
        action="store_true",
        help="DQN kıyaslama ajanını atla",
    )
    parser.add_argument(
        "--skip-db-write",
        action="store_true",
        help="PostgreSQL rl_discount_recommendations tablosuna yazma",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rl_cfg = RLPricingConfig(
        TOTAL_TIMESTEPS=args.timesteps,
        DQN_TIMESTEPS=args.dqn_timesteps,
        DISCOUNT_STOP_DTE=args.stop_dte,
    )
    ml_cfg = MLConfig(USE_POSTGRES=not args.skip_db_write)
    write_db = not args.skip_db_write
    pipeline = EquoraRLPipeline(
        rl_config=rl_cfg,
        ml_config=ml_cfg,
        train_dqn=not args.skip_dqn,
        write_to_db=write_db,
    )
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
