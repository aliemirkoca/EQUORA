r"""Faz 2A — Dinamik İndirim Kıyaslama Politikaları (Baselines).

Girdi:
    - `EquoraDiscountEnv` durumu (`obs`, `action_masks()` ve ortam erişimcileri).

Çıktı:
    - Aynı Out-of-Time test epizotlarında RL ajanlarıyla yarıştırılacak 4
      deterministik kıyaslama politikası:
        1) `NoDiscountPolicy`: Hiç indirim yapmaz (0%).
        2) `StaticRulePolicy`: Mevcut simülatör kuralı (DTE=2->20%, DTE=1->30%, DTE=0->50%).
        3) `Phase1MLFixedDiscountPolicy`: Faz 1 `p_hat >= theta*` ise `risk_index`'e
           göre sabit indirim kademesi (10% / 20% / 30%).
        4) `MyopicExpectedValuePolicy`: Her adımda 6 indirim kademesi için analitik
           tek adımlı beklenen net değeri maksimize eden miyopik optimizasyon.

Rol:
    - Çok adımlı RL ajanının (`MaskablePPO` / `DQN`) yalnızca kural tabanlı
      yaklaşımlara değil, aynı zamanda tek adımlı analitik optimizasyona karşı da
      gerçek katma değerini ölçer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.rl_pricing.discount_env import EquoraDiscountEnv


class BasePricingPolicy(ABC):
    """Tüm fiyatlama politikaları için soyut taban sınıf."""

    name: str

    @abstractmethod
    def select_action(
        self,
        env: EquoraDiscountEnv,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        """Verilen durum ve geçerli aksiyon maskesi için aksiyon indeksi (0..5) döndürür."""

    @staticmethod
    def _snap_to_mask(desired_idx: int, mask: np.ndarray) -> int:
        """İstenen aksiyon maskelenmişse en yakın geçerli aksiyona hizalar."""
        if mask[desired_idx]:
            return int(desired_idx)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        # Monotonluk gereği istenen kademeden büyük veya en yakın geçerli kademe
        ge_valid = valid[valid >= desired_idx]
        if len(ge_valid) > 0:
            return int(ge_valid[0])
        return int(valid[np.argmin(np.abs(valid - desired_idx))])


class NoDiscountPolicy(BasePricingPolicy):
    """Politika 1 — Müdahalesiz Baseline: Her adımda %0 indirim uygular."""

    name = "NoDiscountPolicy"

    def select_action(
        self,
        env: EquoraDiscountEnv,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        return self._snap_to_mask(0, mask)


class StaticRulePolicy(BasePricingPolicy):
    """Politika 2 — Sabit Kural: DTE=2 -> %20 (15%'e en yakın), DTE=1 -> %30, DTE=0 -> %50."""

    name = "StaticRulePolicy"

    def select_action(
        self,
        env: EquoraDiscountEnv,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        dte = env.current_dte
        if dte <= 0:
            desired = 5  # %50
        elif dte == 1:
            desired = 3  # %30
        elif dte == 2:
            desired = 2  # %20 (mevcut simülatörün %15 kademesine en yakın grid)
        else:
            desired = 0  # %0
        return self._snap_to_mask(desired, mask)


class Phase1MLFixedDiscountPolicy(BasePricingPolicy):
    """Politika 3 — Faz 1 ML + Sabit Kademe: p_hat >= theta* ise risk_index'e göre indirim."""

    name = "Phase1MLFixedDiscountPolicy"

    def select_action(
        self,
        env: EquoraDiscountEnv,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        if env.current_p_hat < env.phase1_threshold:
            desired = 0
        else:
            rho = env.current_risk_index
            if rho < 1.3:
                desired = 1  # %10
            elif rho <= 2.0:
                desired = 2  # %20
            else:
                desired = 3  # %30
        return self._snap_to_mask(desired, mask)


class MyopicExpectedValuePolicy(BasePricingPolicy):
    """Politika 4 — Analitik Miyopik Optimizasyon: Tek adımlı beklenen net değeri maksimize eder.

    Her geçerli indirim kademesi d için:
      - Analitik beklenen günlük satış (FEFO ön kuyruk sonrası) hesaplanır,
      - Beklenen satış geliri + EQUORA komisyonu - etiket değişim cezası
        - kalan pencere başına beklenen artık stok & CO2 ceza payı kıyaslanır.
    """

    name = "MyopicExpectedValuePolicy"

    def select_action(
        self,
        env: EquoraDiscountEnv,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        ep = env.current_episode
        cfg = env.config
        dm = env.demand_model
        dte = env.current_dte
        rem = env.current_remaining_qty
        up = env.current_upstream_qty
        prev_d = env.current_prev_discount
        cur_date = env.current_date

        is_weekend = cur_date.isoweekday() >= 6
        is_payday = cur_date.day in (1, 2, 3, 15, 16, 17)
        steps_left = max(1, dte - cfg.DISCOUNT_STOP_DTE + 1)
        unit_co2 = ep.unit_weight_kg * ep.co2_factor_kg_per_kg
        v_init = max(float(ep.remaining_qty) * ep.unit_price, 1.0)

        best_action = int(np.flatnonzero(mask)[0])
        best_score = -float("inf")

        for a_idx in np.flatnonzero(mask):
            d = float(env.discount_grid[a_idx])
            exp_sold, _ = dm.expected_batch_sales(
                base_mu=env.current_base_mu,
                category=ep.category,
                discount_rate=d,
                days_to_expiry=dte,
                is_weekend=is_weekend,
                is_payday_window=is_payday,
                remaining_qty=rem,
                upstream_fefo_qty=up,
                prev_discount_rate=prev_d,
            )
            gross_rev = exp_sold * ep.unit_price * (1.0 - d)
            comm = gross_rev * cfg.EQUORA_COMMISSION_RATE if d > 1e-9 else 0.0
            churn = cfg.PRICE_CHURN_PENALTY * v_init if d > prev_d + 1e-9 else 0.0

            # Tek adımlı artık tahmini (kalan günlere orantılı beklenen imha/CO2 yükü)
            exp_rem_after = max(0.0, float(rem) - exp_sold)
            # Son günde tam terminal ceza; ara günlerde gelecek günlerdeki beklenen satış düşülür
            if steps_left == 1:
                est_terminal_rem = exp_rem_after
            else:
                future_exp_sold = max(0.0, exp_sold * float(steps_left - 1))
                est_terminal_rem = max(0.0, exp_rem_after - future_exp_sold)

            res_penalty = (
                est_terminal_rem * ep.unit_price * cfg.RESIDUAL_WASTE_PENALTY_MULT
                + est_terminal_rem * unit_co2 * cfg.CO2_SHADOW_PRICE_TL_PER_KG
            )
            score = gross_rev + comm - churn - res_penalty
            if score > best_score:
                best_score = score
                best_action = int(a_idx)

        return best_action
