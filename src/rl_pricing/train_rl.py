r"""Faz 2A — RL Dinamik İndirim Eğitim, 6'lı Politika Yarışı ve Stres Testi (RLPricingTrainer).

Girdi:
    - `RLPricingConfig`, `MLConfig`, `EpisodePool` (kronolojik Train/Test),
      dondurulmuş Faz 1 `EquoraHurdleModel`, optimal eşik `theta*` ve
      `CounterfactualDemandModel`.

Çıktı:
    - `models/rl_pricing_v1/maskable_ppo.zip`: Birincil MaskablePPO ajanı.
    - `models/rl_pricing_v1/dqn_benchmark.zip`: Kıyaslama Double DQN ajanı.
    - `models/rl_pricing_v1/rl_metadata.json`: Eğitim hiperparametreleri, süre
      ve epizot havuzu meta verileri.
    - `reports/phase2a_rl_eval/rl_evaluation_report.json`: 6 politikanın Out-of-Time
      test sonuçları, eşleştirilmiş istatistiksel anlamlılık (t-test) ve stres testleri.
    - `reports/phase2a_rl_eval/summary.md`: Jüri sunumuna hazır Türkçe Markdown rapor.

Rol:
    - Aynı Out-of-Time test epizotlarında ve Ortak Rastgele Sayılarla (CRN)
      6 politikayı (NoDiscount, StaticRule, Phase1MLFixed, MyopicEV, MaskablePPO,
      DQN) gelir, EQUORA komisyonu, artık/fire adedi ve önlenen CO2e bazında
      adil şekilde yarıştırır; ayrıca `DISCOUNT_STOP_DTE` (2 vs 0) ve
      `CANNIBALIZATION_RATE` (0.00 vs 0.15) hassasiyet testlerini yürütür.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

# TensorFlow/Keras yan etkilerini bastır (SB3 tensorboard importu sırasında)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from scipy import stats
from stable_baselines3 import DQN

from src.ml_pipeline.config import MLConfig
from src.ml_pipeline.hurdle_model import EquoraHurdleModel
from src.rl_pricing.baselines import (
    BasePricingPolicy,
    MyopicExpectedValuePolicy,
    NoDiscountPolicy,
    Phase1MLFixedDiscountPolicy,
    StaticRulePolicy,
)
from src.rl_pricing.config import RLPricingConfig
from src.rl_pricing.demand_model import CounterfactualDemandModel
from src.rl_pricing.discount_env import EquoraDiscountEnv
from src.rl_pricing.episode_builder import BatchEpisode, EpisodePool


def _mask_fn(env: EquoraDiscountEnv) -> np.ndarray:
    """ActionMasker wrapper için aksiyon maskesi erişim fonksiyonu."""
    return env.action_masks()


class _MaskablePPOPolicyAdapter(BasePricingPolicy):
    """Eğitilmiş MaskablePPO ajanını ortak değerlendirme arayüzüne bağlar."""

    name = "MaskablePPO"

    def __init__(self, model: MaskablePPO) -> None:
        self.model = model

    def select_action(
        self,
        env: EquoraDiscountEnv,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


class _DQNPolicyAdapter(BasePricingPolicy):
    """Eğitilmiş DQN ajanını maskeli/projeksiyonlu değerlendirme arayüzüne bağlar."""

    name = "DQN_Benchmark"

    def __init__(self, model: DQN) -> None:
        self.model = model

    def select_action(
        self,
        env: EquoraDiscountEnv,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        # DQN Q-değerlerinden yalnızca geçerli maske içindeki en yüksek Q seçilir
        obs_t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=self.model.device)
        with torch.no_grad():
            q_vals = self.model.q_net(obs_t).cpu().numpy().reshape(-1)
        masked_q = np.where(mask, q_vals, -1e9)
        return int(np.argmax(masked_q))


@dataclass
class RLTrainingEvaluationOutput:
    """RLPricingTrainer tam eğitim, değerlendirme ve stres testi paketi."""

    ppo_model: MaskablePPO
    dqn_model: DQN | None
    report: dict[str, Any]
    report_json_path: str
    summary_md_path: str


class RLPricingTrainer:
    """MaskablePPO + DQN eğitim, 6'lı politika yarışı ve stres testi orkestratörü."""

    def __init__(
        self,
        rl_config: RLPricingConfig,
        ml_config: MLConfig | None = None,
    ) -> None:
        self.rl_config = rl_config
        self.ml_config = ml_config or MLConfig()
        self.rl_config.ensure_directories()

    # -------------------------------------------------------------------------
    # 1) Ortak Rastgele Sayılarla (CRN) Out-of-Time Test Değerlendirmesi
    # -------------------------------------------------------------------------
    @staticmethod
    def evaluate_policy_on_episodes(
        policy: BasePricingPolicy,
        env: EquoraDiscountEnv,
        episodes: list[BatchEpisode],
        base_seed: int,
    ) -> dict[str, Any]:
        """Verilen politikayı tüm test epizotlarında sabit CRN tohumlarıyla değerlendirir."""
        n_eps = len(episodes)
        ep_revenues = np.zeros(n_eps, dtype=np.float64)
        ep_commissions = np.zeros(n_eps, dtype=np.float64)
        ep_sold_units = np.zeros(n_eps, dtype=np.float64)
        ep_waste_units = np.zeros(n_eps, dtype=np.float64)
        ep_waste_tl = np.zeros(n_eps, dtype=np.float64)
        ep_co2_kg = np.zeros(n_eps, dtype=np.float64)
        ep_rewards = np.zeros(n_eps, dtype=np.float64)
        ep_avg_discounts = np.zeros(n_eps, dtype=np.float64)
        ep_changes = np.zeros(n_eps, dtype=np.float64)
        total_steps = 0
        total_violations = 0
        discounted_episodes = 0

        for idx in range(n_eps):
            obs, info = env.reset(
                seed=base_seed + idx,
                options={"episode_idx": idx},
            )
            done = False
            ep_rew = 0.0
            while not done:
                mask = env.action_masks()
                action = policy.select_action(env, obs, mask)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_rew += float(reward)
                done = terminated or truncated

            ep_revenues[idx] = float(info["revenue_tl"])
            ep_commissions[idx] = float(info["equora_commission_tl"])
            ep_sold_units[idx] = float(info["sold_units"])
            ep_waste_units[idx] = float(info["residual_waste_units"])
            ep_waste_tl[idx] = float(info["residual_waste_tl"])
            ep_co2_kg[idx] = float(info["residual_co2_kg"])
            ep_rewards[idx] = ep_rew
            ep_avg_discounts[idx] = float(info["avg_discount_applied"])
            ep_changes[idx] = float(info["discount_changes"])
            total_steps += int(info["discount_steps"])
            total_violations += int(info["monotonic_violations"])
            if float(info["avg_discount_applied"]) > 1e-6:
                discounted_episodes += 1

        # Net Ekonomik Değer = Satış Geliri + EQUORA Komisyonu - Artık Stok İmha/Devir Maliyeti
        net_economic_tl = (
            ep_revenues
            + ep_commissions
            - ep_waste_tl * env.config.RESIDUAL_WASTE_PENALTY_MULT
        )

        return {
            "policy_name": policy.name,
            "episodes": n_eps,
            "total_revenue_tl": round(float(np.sum(ep_revenues)), 2),
            "total_commission_tl": round(float(np.sum(ep_commissions)), 2),
            "total_net_economic_tl": round(float(np.sum(net_economic_tl)), 2),
            "total_sold_units": int(round(float(np.sum(ep_sold_units)))),
            "total_residual_waste_units": int(round(float(np.sum(ep_waste_units)))),
            "total_residual_waste_tl": round(float(np.sum(ep_waste_tl)), 2),
            "total_residual_co2_kg": round(float(np.sum(ep_co2_kg)), 2),
            "mean_episode_reward": round(float(np.mean(ep_rewards)), 5),
            "avg_discount_pct": round(float(np.mean(ep_avg_discounts)) * 100.0, 2),
            "discounted_episode_pct": round(100.0 * discounted_episodes / max(n_eps, 1), 2),
            "price_change_rate_per_episode": round(float(np.mean(ep_changes)), 3),
            "total_steps": int(total_steps),
            "monotonic_violations": int(total_violations),
            # İstatistiksel anlamlılık testi için ham vektörler (rapor JSON'a yazılmaz)
            "_ep_revenues": ep_revenues,
            "_ep_net_economic": net_economic_tl,
            "_ep_waste_units": ep_waste_units,
            "_ep_co2_kg": ep_co2_kg,
        }

    @staticmethod
    def _attach_relative_gains(
        policy_results: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """NoDiscountPolicy'ye göre göreli artış/azalışları ve eşleştirilmiş t-testini hesaplar."""
        base = next(r for r in policy_results if r["policy_name"] == "NoDiscountPolicy")
        base_rev = base["total_revenue_tl"]
        base_net = base["total_net_economic_tl"]
        base_waste = base["total_residual_waste_units"]
        base_co2 = base["total_residual_co2_kg"]

        cleaned: list[dict[str, Any]] = []
        ppo_raw: dict[str, Any] | None = None

        for r in policy_results:
            if r["policy_name"] == "MaskablePPO":
                ppo_raw = r
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            row["revenue_lift_vs_nodisc_pct"] = round(
                100.0 * (r["total_revenue_tl"] - base_rev) / max(base_rev, 1e-6), 2
            )
            row["net_economic_lift_vs_nodisc_pct"] = round(
                100.0 * (r["total_net_economic_tl"] - base_net) / max(base_net, 1e-6), 2
            )
            row["waste_reduction_vs_nodisc_pct"] = round(
                100.0 * (base_waste - r["total_residual_waste_units"]) / max(base_waste, 1),
                2,
            )
            prevented_co2 = base_co2 - r["total_residual_co2_kg"]
            row["prevented_co2_vs_nodisc_kg"] = round(prevented_co2, 2)
            row["prevented_co2_vs_nodisc_tons"] = round(prevented_co2 / 1000.0, 3)
            row["co2_reduction_vs_nodisc_pct"] = round(
                100.0 * prevented_co2 / max(base_co2, 1e-6), 2
            )
            cleaned.append(row)

        # Eşleştirilmiş t-test (MaskablePPO vs. NoDiscountPolicy — CRN çiftleri)
        sig_tests: dict[str, Any] = {}
        if ppo_raw is not None:
            for metric_key, label, sign in (
                ("_ep_revenues", "revenue_tl", 1.0),
                ("_ep_net_economic", "net_economic_tl", 1.0),
                ("_ep_waste_units", "waste_units_reduced", -1.0),
                ("_ep_co2_kg", "co2_kg_prevented", -1.0),
            ):
                diff = sign * (ppo_raw[metric_key] - base[metric_key])
                t_stat, p_val = stats.ttest_1samp(diff, popmean=0.0)
                mean_d = float(np.mean(diff))
                se_d = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
                sig_tests[label] = {
                    "mean_diff_per_episode": round(mean_d, 4),
                    "ci95_low": round(mean_d - 1.96 * se_d, 4),
                    "ci95_high": round(mean_d + 1.96 * se_d, 4),
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": float(f"{float(p_val):.3e}"),
                    "significant_at_001": bool(p_val < 0.01),
                }

        return cleaned, sig_tests

    # -------------------------------------------------------------------------
    # 2) Ana Eğitim + 6'lı Yarış + Stres Testi Akışı
    # -------------------------------------------------------------------------
    def train_and_evaluate(
        self,
        pool: EpisodePool,
        hurdle_model: EquoraHurdleModel,
        phase1_threshold: float,
        demand_model: CounterfactualDemandModel,
        train_dqn: bool = True,
    ) -> RLTrainingEvaluationOutput:
        """MaskablePPO (ve opsiyonel DQN) eğitir, test yarışı ve stres testlerini üretir."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # Küçük [128, 128] MLP rollout'larında OpenMP çekirdek senkronizasyon yükünü azalt
            torch.set_num_threads(1)
        print(f"  [BİLGİ] RL Eğitim Cihazı: {device.upper()} | Timesteps: {self.rl_config.TOTAL_TIMESTEPS:,}")

        # 1) Eğitim ortamı (ActionMasker sarmalayıcı ile)
        raw_train_env = EquoraDiscountEnv(
            episodes=pool.train_episodes,
            config=self.rl_config,
            demand_model=demand_model,
            hurdle_model=hurdle_model,
            phase1_threshold=phase1_threshold,
            seed=self.rl_config.SEED,
        )
        masked_train_env = ActionMasker(raw_train_env, _mask_fn)

        # 2) Birincil Ajan: MaskablePPO
        t_ppo = time.perf_counter()
        ppo_model = MaskablePPO(
            policy="MlpPolicy",
            env=masked_train_env,
            learning_rate=self.rl_config.PPO_LEARNING_RATE,
            n_steps=self.rl_config.PPO_N_STEPS,
            batch_size=self.rl_config.PPO_BATCH_SIZE,
            gamma=self.rl_config.PPO_GAMMA,
            ent_coef=self.rl_config.PPO_ENT_COEF,
            policy_kwargs={"net_arch": list(self.rl_config.PPO_NET_ARCH)},
            seed=self.rl_config.SEED,
            device=device,
            verbose=0,
        )
        ppo_model.learn(total_timesteps=self.rl_config.TOTAL_TIMESTEPS)
        ppo_train_sec = time.perf_counter() - t_ppo
        ppo_model.save(str(self.rl_config.ppo_model_path))
        print(
            f"  [OK] MaskablePPO eğitimi tamamlandı ({ppo_train_sec:.1f}s) -> "
            f"{self.rl_config.ppo_model_path.name}"
        )

        # 3) Kıyaslama Ajanı (Ablation): Double DQN
        dqn_model: DQN | None = None
        dqn_train_sec = 0.0
        if train_dqn:
            t_dqn = time.perf_counter()
            dqn_train_env = EquoraDiscountEnv(
                episodes=pool.train_episodes,
                config=self.rl_config,
                demand_model=demand_model,
                hurdle_model=hurdle_model,
                phase1_threshold=phase1_threshold,
                seed=self.rl_config.SEED + 101,
            )
            dqn_model = DQN(
                policy="MlpPolicy",
                env=dqn_train_env,
                learning_rate=self.rl_config.DQN_LEARNING_RATE,
                buffer_size=self.rl_config.DQN_BUFFER_SIZE,
                batch_size=self.rl_config.DQN_BATCH_SIZE,
                exploration_fraction=self.rl_config.DQN_EXPLORATION_FRACTION,
                gamma=self.rl_config.PPO_GAMMA,
                policy_kwargs={"net_arch": list(self.rl_config.DQN_NET_ARCH)},
                seed=self.rl_config.SEED,
                device=device,
                verbose=0,
            )
            dqn_model.learn(total_timesteps=self.rl_config.DQN_TIMESTEPS)
            dqn_train_sec = time.perf_counter() - t_dqn
            dqn_model.save(str(self.rl_config.dqn_model_path))
            print(
                f"  [OK] DQN Benchmark eğitimi tamamlandı ({dqn_train_sec:.1f}s) -> "
                f"{self.rl_config.dqn_model_path.name}"
            )

        # 4) Out-of-Time Test Yarışı (Ana Senaryo)
        policies: list[BasePricingPolicy] = [
            NoDiscountPolicy(),
            StaticRulePolicy(),
            Phase1MLFixedDiscountPolicy(),
            MyopicExpectedValuePolicy(),
            _MaskablePPOPolicyAdapter(ppo_model),
        ]
        if dqn_model is not None:
            policies.append(_DQNPolicyAdapter(dqn_model))

        test_env = EquoraDiscountEnv(
            episodes=pool.test_episodes,
            config=self.rl_config,
            demand_model=demand_model,
            hurdle_model=hurdle_model,
            phase1_threshold=phase1_threshold,
            seed=self.rl_config.SEED + 10_000,
        )

        raw_results = [
            self.evaluate_policy_on_episodes(
                policy=pol,
                env=test_env,
                episodes=pool.test_episodes,
                base_seed=self.rl_config.SEED + 10_000,
            )
            for pol in policies
        ]
        main_table, sig_tests = self._attach_relative_gains(raw_results)

        # Monoton indirim ihlali sıfır doğrulaması (tüm politikalar)
        total_mono_violations = sum(r["monotonic_violations"] for r in main_table)
        assert total_mono_violations == 0, (
            f"Monoton indirim kısıtı ihlali tespit edildi: {total_mono_violations}"
        )

        # 5) Hassasiyet / Stres Testleri:
        #    (a) DISCOUNT_STOP_DTE = 0 (SKT gün sonuna kadar tam 5 günlük indirim)
        #    (b) CANNIBALIZATION_RATE = 0.15 (Stratejik tüketici talep öne çekme şoku)
        stress_scenarios: list[dict[str, Any]] = []
        stress_specs = [
            (
                "stop_dte_0_full_window",
                "SKT Gün Sonuna Kadar İndirim (DISCOUNT_STOP_DTE=0)",
                replace(self.rl_config, DISCOUNT_STOP_DTE=0),
            ),
            (
                "cannibalization_015",
                "Stratejik Tüketici Yamyamlaşma Stresi (CANNIBALIZATION=0.15)",
                replace(self.rl_config, CANNIBALIZATION_RATE=0.15),
            ),
        ]

        eval_subset_policies: list[BasePricingPolicy] = [
            NoDiscountPolicy(),
            StaticRulePolicy(),
            Phase1MLFixedDiscountPolicy(),
            MyopicExpectedValuePolicy(),
            _MaskablePPOPolicyAdapter(ppo_model),
        ]
        for key, title, st_cfg in stress_specs:
            st_dm = CounterfactualDemandModel(st_cfg, profile=demand_model.profile)
            st_env = EquoraDiscountEnv(
                episodes=pool.test_episodes,
                config=st_cfg,
                demand_model=st_dm,
                hurdle_model=hurdle_model,
                phase1_threshold=phase1_threshold,
                seed=self.rl_config.SEED + 20_000,
            )
            st_raw = [
                self.evaluate_policy_on_episodes(
                    policy=pol,
                    env=st_env,
                    episodes=pool.test_episodes,
                    base_seed=self.rl_config.SEED + 20_000,
                )
                for pol in eval_subset_policies
            ]
            st_table, st_sig = self._attach_relative_gains(st_raw)
            stress_scenarios.append(
                {
                    "scenario_key": key,
                    "title": title,
                    "discount_stop_dte": st_cfg.DISCOUNT_STOP_DTE,
                    "cannibalization_rate": st_cfg.CANNIBALIZATION_RATE,
                    "results": st_table,
                    "ppo_vs_nodiscount_ttest": st_sig,
                }
            )

        # 6) Metadata ve Raporları Kaydet
        metadata: dict[str, Any] = {
            "policy_version": "maskable_ppo_v1",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "seed": self.rl_config.SEED,
            "phase1_threshold_used": phase1_threshold,
            "discount_grid": list(self.rl_config.DISCOUNT_GRID),
            "discount_start_max_dte": self.rl_config.DISCOUNT_START_MAX_DTE,
            "discount_stop_dte": self.rl_config.DISCOUNT_STOP_DTE,
            "enforce_monotonic_discount": self.rl_config.ENFORCE_MONOTONIC_DISCOUNT,
            "gate_with_phase1_threshold": self.rl_config.GATE_WITH_PHASE1_THRESHOLD,
            "equora_commission_rate": self.rl_config.EQUORA_COMMISSION_RATE,
            "residual_waste_penalty_mult": self.rl_config.RESIDUAL_WASTE_PENALTY_MULT,
            "co2_shadow_price_tl_per_kg": self.rl_config.CO2_SHADOW_PRICE_TL_PER_KG,
            "ppo_timesteps": self.rl_config.TOTAL_TIMESTEPS,
            "ppo_train_sec": round(ppo_train_sec, 2),
            "dqn_trained": train_dqn,
            "dqn_timesteps": self.rl_config.DQN_TIMESTEPS if train_dqn else 0,
            "dqn_train_sec": round(dqn_train_sec, 2),
            "train_episodes": len(pool.train_episodes),
            "test_episodes": len(pool.test_episodes),
            "train_date_range": [str(pool.train_date_range[0]), str(pool.train_date_range[1])],
            "test_date_range": [str(pool.test_date_range[0]), str(pool.test_date_range[1])],
            "batch_overlap": pool.batch_overlap,
        }
        self.rl_config.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report: dict[str, Any] = {
            "metadata": metadata,
            "main_benchmark_test": main_table,
            "ppo_vs_nodiscount_significance": sig_tests,
            "total_monotonic_violations": total_mono_violations,
            "stress_tests": stress_scenarios,
        }
        self.rl_config.eval_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_md = self._render_markdown_summary(report)
        self.rl_config.eval_summary_path.write_text(summary_md, encoding="utf-8")

        return RLTrainingEvaluationOutput(
            ppo_model=ppo_model,
            dqn_model=dqn_model,
            report=report,
            report_json_path=str(self.rl_config.eval_report_path),
            summary_md_path=str(self.rl_config.eval_summary_path),
        )

    @staticmethod
    def _render_markdown_summary(report: dict[str, Any]) -> str:
        """Jüri ve mühendislik sunumu için Faz 2A RL Markdown değerlendirme raporu."""
        meta = report["metadata"]
        main = report["main_benchmark_test"]
        sig = report["ppo_vs_nodiscount_significance"]

        lines: list[str] = [
            "# EQUORA Faz 2A — RL Dinamik İndirim Motoru Değerlendirme Raporu",
            "",
            "## 1. Eğitim ve Ortam Özeti",
            f"- **Birincil Ajan:** `MaskablePPO` (`{meta['ppo_timesteps']:,}` adım, `{meta['ppo_train_sec']:.1f} sn`, cihaz: `{meta['device']}`)",
            f"- **Kıyaslama Ajanı:** `DQN` (`{meta['dqn_timesteps']:,}` adım, `{meta['dqn_train_sec']:.1f} sn`)" if meta["dqn_trained"] else "- **Kıyaslama Ajanı:** Atlandı",
            f"- **Kronolojik Epizot Bölmesi:** Train `{meta['train_episodes']:,}` parti (`{meta['train_date_range'][0]}..{meta['train_date_range'][1]}`) | Out-of-Time Test `{meta['test_episodes']:,}` parti (`{meta['test_date_range'][0]}..{meta['test_date_range'][1]}`) | Kesişim: `{meta['batch_overlap']}`",
            f"- **İndirim Penceresi:** `DTE {meta['discount_start_max_dte']} -> {meta['discount_stop_dte']}` | Monotonluk Kısıtı: `{meta['enforce_monotonic_discount']}` (İhlal: **`{report['total_monotonic_violations']}`**) | Faz 1 Kapısı: `{meta['gate_with_phase1_threshold']}` (`theta* = {meta['phase1_threshold_used']:.2f}`)",
            f"- **Birim Ekonomi:** EQUORA Komisyonu `%{meta['equora_commission_rate'] * 100:.0f}` | Artık Stok Cezası `{meta['residual_waste_penalty_mult']:.2f}x` | CO2 Gölge Fiyatı `{meta['co2_shadow_price_tl_per_kg']:.2f} TL/kg`",
            "",
            "## 2. Out-of-Time Test Politika Yarışı (Ortak Rastgele Sayılar — CRN)",
            "| Politika | Brüt Gelir (TL) | Gelir Artışı | EQUORA Komisyon (TL) | Net Ekonomik Değer (TL) | Artık Fire (Adet) | Fire Azalışı | Önlenen CO2e (kg) | Ort. İndirim | Fiyat Değiş./Ep |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in main:
            bold = "**" if r["policy_name"] == "MaskablePPO" else ""
            lines.append(
                f"| {bold}{r['policy_name']}{bold} "
                f"| {r['total_revenue_tl']:,.2f} "
                f"| {r['revenue_lift_vs_nodisc_pct']:+.2f}% "
                f"| {r['total_commission_tl']:,.2f} "
                f"| {bold}{r['total_net_economic_tl']:,.2f}{bold} "
                f"| {r['total_residual_waste_units']:,} "
                f"| {r['waste_reduction_vs_nodisc_pct']:+.2f}% "
                f"| {r['prevented_co2_vs_nodisc_kg']:,.2f} "
                f"| %{r['avg_discount_pct']:.2f} "
                f"| {r['price_change_rate_per_episode']:.2f} |"
            )

        if sig:
            lines.extend(
                [
                    "",
                    "## 3. İstatistiksel Anlamlılık (MaskablePPO vs. NoDiscountPolicy — Eşleştirilmiş t-Test)",
                    "| Metrik (Epizot Başına Fark) | Ortalama Fark | %95 Güven Aralığı | t-İstatistiği | p-Değeri | Anlamlı (p<0.01) |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for label, s in sig.items():
                lines.append(
                    f"| `{label}` | {s['mean_diff_per_episode']:+.4f} "
                    f"| [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] "
                    f"| {s['t_statistic']:.2f} | `{s['p_value']:.3e}` "
                    f"| {'EVET' if s['significant_at_001'] else 'HAYIR'} |"
                )

        lines.extend(["", "## 4. Hassasiyet ve Stres Testleri (Out-of-Time Test)"])
        for sc in report["stress_tests"]:
            lines.extend(
                [
                    "",
                    f"### {sc['title']}",
                    "| Politika | Brüt Gelir (TL) | Gelir Artışı | EQUORA Komisyon (TL) | Net Ekonomik Değer (TL) | Artık Fire (Adet) | Fire Azalışı | Önlenen CO2e (kg) | Ort. İndirim |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for r in sc["results"]:
                bold = "**" if r["policy_name"] == "MaskablePPO" else ""
                lines.append(
                    f"| {bold}{r['policy_name']}{bold} "
                    f"| {r['total_revenue_tl']:,.2f} "
                    f"| {r['revenue_lift_vs_nodisc_pct']:+.2f}% "
                    f"| {r['total_commission_tl']:,.2f} "
                    f"| {bold}{r['total_net_economic_tl']:,.2f}{bold} "
                    f"| {r['total_residual_waste_units']:,} "
                    f"| {r['waste_reduction_vs_nodisc_pct']:+.2f}% "
                    f"| {r['prevented_co2_vs_nodisc_kg']:,.2f} "
                    f"| %{r['avg_discount_pct']:.2f} |"
                )

        lines.append("")
        return "\n".join(lines)
