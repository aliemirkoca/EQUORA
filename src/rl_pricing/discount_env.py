r"""Faz 2A — Gymnasium + Action Masking Uyumlu RL Fiyatlama Ortamı (EquoraDiscountEnv).

Girdi:
    - `RLPricingConfig`, `BatchEpisode` havuzu, `CounterfactualDemandModel`,
      dondurulmuş Faz 1 `EquoraHurdleModel` ve optimal karar eşiği `theta*`.

Çıktı:
    - `gymnasium.Env` uyumlu kapalı döngü dinamik indirim ortamı:
      * 18 boyutlu `[0.0, 1.0]` normalize durum vektörü (`Box(0, 1, (18,), float32)`),
      * 6 kademeli ayrık indirim aksiyon uzayı (`Discrete(6)` -> 0%..50%),
      * `action_masks()` metodu (monoton indirim `d_t >= d_{t-1}` + Faz 1 eşik kapısı),
      * Normalize gelir + EQUORA komisyonu - etiket değişim cezası - terminal
        artık stok & CO2 gölge fiyat cezası ödül fonksiyonu.

Rol:
    - MaskablePPO, Double DQN ve tüm kıyaslama politikalarının (No-Discount,
      Static Rule, Phase1-ML Rule, Myopic EV) birebir aynı fiziksel ve finansal
      kurallarla eğitilip yarıştırıldığı ortak kapalı döngü simülasyon ortamıdır.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.ml_pipeline.hurdle_model import EquoraHurdleModel
from src.rl_pricing.config import RLPricingConfig
from src.rl_pricing.demand_model import CounterfactualDemandModel
from src.rl_pricing.episode_builder import BatchEpisode


def _weekend_days_left_scalar(current_date: date, days_to_expiry: int) -> int:
    """(current_date, expiry_date] penceresindeki Cumartesi/Pazar gün sayısını O(1) hesaplar."""
    n = max(0, int(days_to_expiry))
    if n == 0:
        return 0
    dow = current_date.isoweekday()  # 1=Pzt .. 7=Paz
    full_weeks = n // 7
    rem = n % 7
    d_sat = ((6 - dow + 6) % 7) + 1
    d_sun = ((7 - dow + 6) % 7) + 1
    return 2 * full_weeks + int(rem >= d_sat) + int(rem >= d_sun)


class EquoraDiscountEnv(gym.Env[np.ndarray, int]):
    """EQUORA Faz 2A — Tek partili, FEFO kuyruklu dinamik indirim Gymnasium ortamı."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: list[BatchEpisode],
        config: RLPricingConfig,
        demand_model: CounterfactualDemandModel,
        hurdle_model: EquoraHurdleModel,
        phase1_threshold: float,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not episodes:
            raise ValueError("EquoraDiscountEnv için en az 1 BatchEpisode gereklidir.")

        self.episodes = episodes
        self.config = config
        self.demand_model = demand_model
        self.hurdle_model = hurdle_model
        self.phase1_threshold = float(phase1_threshold)
        self.discount_grid = np.asarray(config.DISCOUNT_GRID, dtype=np.float64)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(18,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(config.DISCOUNT_GRID))

        self._rng = np.random.default_rng(seed if seed is not None else config.SEED)
        self._feature_index = {
            name: idx for idx, name in enumerate(self.hurdle_model.feature_columns)
        }
        self._n_phase1_features = len(self.hurdle_model.feature_columns)

        # Aktif epizot durum değişkenleri
        self._ep: BatchEpisode = self.episodes[0]
        self._base_mu: float = 1.0
        self._current_date: date = self._ep.snapshot_date
        self._dte: int = self._ep.days_to_expiry
        self._remaining_qty: int = self._ep.remaining_qty
        self._upstream_qty: int = self._ep.upstream_fefo_qty
        self._prev_discount: float = 0.0
        self._prev_action_idx: int = 0
        self._daily_out_qty: int = 0
        self._vel_3d: float = 0.0
        self._vel_7d: float = 0.0
        self._vel_14d: float = 0.0
        self._vel_std_7d: float = 0.0
        self._recent_discounts: list[float] = [0.0, 0.0, 0.0]
        self._v_init: float = 1.0

        # Son Faz 1 tahmin sinyalleri (obs & action_masks için önbellekli)
        self._last_p_hat: float = 0.5
        self._last_q_hat: float = 0.0
        self._last_risk_index: float = 1.0

        # Epizot birikimli iş & ESG sayaçları
        self._ep_revenue_tl: float = 0.0
        self._ep_commission_tl: float = 0.0
        self._ep_sold_units: int = 0
        self._ep_discount_sum: float = 0.0
        self._ep_steps: int = 0
        self._ep_discount_changes: int = 0
        self._ep_monotonic_violations: int = 0

    # -------------------------------------------------------------------------
    # Yardımcı: Dondurulmuş Faz 1 Hurdle Modeli için 27 Boyutlu Özellik Vektörü
    # -------------------------------------------------------------------------
    def _build_phase1_feature_vector(self) -> np.ndarray:
        """Güncel ortam durumundan Faz 1 `EquoraHurdleModel` özellik vektörünü kurar."""
        ep = self._ep
        rem = float(self._remaining_qty)
        up = float(self._upstream_qty)
        eff_stock = rem + up
        dte = float(self._dte)
        init_in = float(max(ep.initial_in_qty, 1))
        shelf = float(max(ep.shelf_life_days, 1))

        days_since_in = float(max(0, (self._current_date - ep.first_in_date).days))
        remaining_ratio = rem / init_in
        life_ratio = dte / shelf
        demand_momentum = self._vel_3d / (self._vel_14d + 1e-5)
        days_of_cover = eff_stock / (self._vel_7d + 1e-5)
        risk_index = days_of_cover / max(dte, 0.5)

        dow = self._current_date.isoweekday()
        is_weekend = 1.0 if dow >= 6 else 0.0
        is_payday = 1.0 if self._current_date.day in (1, 2, 3, 15, 16, 17) else 0.0
        wknd_left = float(_weekend_days_left_scalar(self._current_date, self._dte))

        lag1_disc = self._recent_discounts[-1] if self._recent_discounts else 0.0
        roll3_disc = float(np.mean(self._recent_discounts[-3:])) if self._recent_discounts else 0.0

        self._last_risk_index = float(risk_index)

        feat_map = {
            "upstream_fefo_qty": up,
            "effective_stock_qty": eff_stock,
            "active_batch_count": float(max(1, ep.active_batch_count if up > 0 else 1)),
            "remaining_qty": rem,
            "initial_in_qty": init_in,
            "daily_out_qty": float(self._daily_out_qty),
            "days_to_expiry": dte,
            "remaining_ratio": remaining_ratio,
            "life_ratio": life_ratio,
            "days_since_in": days_since_in,
            "vel_3d": self._vel_3d,
            "vel_7d": self._vel_7d,
            "vel_14d": self._vel_14d,
            "vel_std_7d": self._vel_std_7d,
            "demand_momentum": demand_momentum,
            "days_of_cover": days_of_cover,
            "risk_index": risk_index,
            "lag1_discount_rate": lag1_disc,
            "rolling_3d_avg_discount": roll3_disc,
            "day_of_week": float(dow),
            "is_weekend": is_weekend,
            "is_payday_window": is_payday,
            "weekend_days_left": wknd_left,
            "category_code": float(ep.category_code),
            "store_id": float(ep.store_id),
            "unit_price": float(ep.unit_price),
            "shelf_life_days": shelf,
        }

        x = np.zeros(self._n_phase1_features, dtype=np.float64)
        for name, idx in self._feature_index.items():
            x[idx] = feat_map.get(name, 0.0)
        return x

    def _refresh_phase1_signals(self) -> None:
        """Dondurulmuş Faz 1 modelinden güncel (p_hat, q_hat) sinyallerini hesaplar."""
        if self._remaining_qty <= 0:
            self._last_p_hat = 0.0
            self._last_q_hat = 0.0
            self._last_risk_index = 0.0
            return
        x_vec = self._build_phase1_feature_vector()
        p_hat, q_hat, _ = self.hurdle_model.predict_single_fast(
            x_1d=x_vec,
            remaining_qty=float(self._remaining_qty),
        )
        self._last_p_hat = float(p_hat)
        self._last_q_hat = float(q_hat)

    def _get_obs(self) -> np.ndarray:
        """18 boyutlu [0.0, 1.0] aralığında float32 gözlem vektörü üretir."""
        ep = self._ep
        rem = float(self._remaining_qty)
        up = float(self._upstream_qty)
        init_in = float(max(ep.initial_in_qty, 1))
        shelf = float(max(ep.shelf_life_days, 1))

        remaining_ratio = np.clip(rem / init_in, 0.0, 1.0)
        effective_stock_ratio = np.clip((rem + up) / (2.0 * init_in), 0.0, 1.0)
        upstream_ratio = np.clip(up / max(rem + up, 1.0), 0.0, 1.0)
        dte_norm = np.clip(
            float(self._dte) / float(max(self.config.DISCOUNT_START_MAX_DTE, 1)),
            0.0,
            1.0,
        )
        life_ratio = np.clip(float(self._dte) / shelf, 0.0, 1.0)

        p_hat = np.clip(self._last_p_hat, 0.0, 1.0)
        q_hat_ratio = np.clip(self._last_q_hat / max(rem, 1.0), 0.0, 1.0)
        risk_index_norm = np.clip(self._last_risk_index / 4.0, 0.0, 1.0)

        vel_7d_norm = np.clip(self._vel_7d / 35.0, 0.0, 1.0)
        demand_momentum = self._vel_3d / (self._vel_14d + 1e-5)
        demand_momentum_norm = np.clip(demand_momentum / 2.5, 0.0, 1.0)
        unit_price_norm = np.clip(ep.unit_price / 850.0, 0.0, 1.0)

        cat_elast = self.demand_model.get_category_elasticity(ep.category)
        cat_elast_norm = np.clip((cat_elast - 0.5) / 1.5, 0.0, 1.0)
        prev_disc_norm = np.clip(self._prev_discount / 0.50, 0.0, 1.0)

        is_weekend = 1.0 if self._current_date.isoweekday() >= 6 else 0.0
        wknd_left = _weekend_days_left_scalar(self._current_date, self._dte)
        wknd_left_norm = np.clip(float(wknd_left) / 2.0, 0.0, 1.0)
        is_payday = 1.0 if self._current_date.day in (1, 2, 3, 15, 16, 17) else 0.0
        cat_code_norm = np.clip(float(ep.category_code) / 5.0, 0.0, 1.0)
        store_mult = self.demand_model.get_store_multiplier(ep.store_id)
        store_mult_norm = np.clip((store_mult - 0.70) / 0.60, 0.0, 1.0)

        obs = np.array(
            [
                remaining_ratio,
                effective_stock_ratio,
                upstream_ratio,
                dte_norm,
                life_ratio,
                p_hat,
                q_hat_ratio,
                risk_index_norm,
                vel_7d_norm,
                demand_momentum_norm,
                unit_price_norm,
                cat_elast_norm,
                prev_disc_norm,
                is_weekend,
                wknd_left_norm,
                is_payday,
                cat_code_norm,
                store_mult_norm,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

    # -------------------------------------------------------------------------
    # Aksiyon Maskeleme (MaskablePPO & Güvenli Kısıt Motoru)
    # -------------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """Geçerli aksiyonlar için boolean maske döndürür (shape=(6,))."""
        mask = np.ones(len(self.discount_grid), dtype=bool)

        # 1) Monoton indirim kısıtı: d_t >= d_{t-1} (dün yapılan indirim geri alınmaz)
        if self.config.ENFORCE_MONOTONIC_DISCOUNT:
            mask &= self.discount_grid >= (self._prev_discount - 1e-9)

        # 2) Faz 1 Eşik Kapısı: Güvenli (HOLD) ve henüz indirime girmemiş partide
        #    gereksiz erken indirimi engelle (yalnızca %0 açık)
        if (
            self.config.GATE_WITH_PHASE1_THRESHOLD
            and self._last_p_hat < self.phase1_threshold
            and self._prev_discount <= 1e-9
        ):
            mask[:] = False
            mask[0] = True

        # Güvenlik garantisi: Her durumda en az 1 aksiyon geçerli olmalıdır
        if not np.any(mask):
            mask[self._prev_action_idx] = True
        return mask

    def _project_to_valid_action(self, action: int) -> int:
        """Maske kullanmayan politikalar (örn. DQN) için aksiyonu en yakın geçerli kademeye projekte eder."""
        mask = self.action_masks()
        a_idx = int(np.clip(action, 0, len(self.discount_grid) - 1))
        if mask[a_idx]:
            return a_idx
        valid_indices = np.flatnonzero(mask)
        nearest = valid_indices[np.argmin(np.abs(valid_indices - a_idx))]
        return int(nearest)

    # -------------------------------------------------------------------------
    # Gymnasium reset() & step()
    # -------------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Deterministik test değerlendirmesi için `options={"episode_idx": k}` desteklenir
        if options is not None and "episode_idx" in options:
            ep_idx = int(options["episode_idx"]) % len(self.episodes)
        else:
            ep_idx = int(self._rng.integers(0, len(self.episodes)))

        ep = self.episodes[ep_idx]
        self._ep = ep
        self._base_mu = self.demand_model.estimate_base_daily_demand(
            category=ep.category,
            store_id=ep.store_id,
            vel_14d=ep.vel_14d,
            vel_7d=ep.vel_7d,
        )
        self._current_date = ep.snapshot_date
        self._dte = ep.days_to_expiry
        self._remaining_qty = ep.remaining_qty
        self._upstream_qty = ep.upstream_fefo_qty
        self._prev_discount = 0.0
        self._prev_action_idx = 0
        self._daily_out_qty = ep.daily_out_qty
        self._vel_3d = ep.vel_3d
        self._vel_7d = ep.vel_7d
        self._vel_14d = ep.vel_14d
        self._vel_std_7d = ep.vel_std_7d
        self._recent_discounts = [
            ep.rolling_3d_avg_discount,
            ep.rolling_3d_avg_discount,
            ep.lag1_discount_rate,
        ]
        self._v_init = max(float(ep.remaining_qty) * float(ep.unit_price), 1.0)

        # Epizot iş sayaçlarını sıfırla
        self._ep_revenue_tl = 0.0
        self._ep_commission_tl = 0.0
        self._ep_sold_units = 0
        self._ep_discount_sum = 0.0
        self._ep_steps = 0
        self._ep_discount_changes = 0
        self._ep_monotonic_violations = 0

        self._refresh_phase1_signals()
        obs = self._get_obs()
        info = self._build_step_info(terminated=False)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Bir günlük fiyatlama kararını uygular, FEFO satışını simüle eder ve ödül döndürür."""
        valid_action = self._project_to_valid_action(int(action))
        discount_rate = float(self.discount_grid[valid_action])

        # Monotonluk ihlali kontrolü (maske/projeksiyon ile 0 olmalıdır)
        if self.config.ENFORCE_MONOTONIC_DISCOUNT and discount_rate < self._prev_discount - 1e-9:
            self._ep_monotonic_violations += 1

        is_weekend = self._current_date.isoweekday() >= 6
        is_payday = self._current_date.day in (1, 2, 3, 15, 16, 17)

        # 1) Karşı-olgusal FEFO günlük satış simülasyonu
        outcome = self.demand_model.simulate_day(
            rng=self._rng,
            base_mu=self._base_mu,
            category=self._ep.category,
            discount_rate=discount_rate,
            days_to_expiry=self._dte,
            is_weekend=is_weekend,
            is_payday_window=is_payday,
            remaining_qty=self._remaining_qty,
            upstream_fefo_qty=self._upstream_qty,
            prev_discount_rate=self._prev_discount,
        )

        # 2) Günlük gelir, EQUORA komisyonu ve ara adım ödülü
        gross_sale_tl = float(outcome.batch_sold) * self._ep.unit_price * (1.0 - discount_rate)
        commission_tl = (
            gross_sale_tl * self.config.EQUORA_COMMISSION_RATE
            if discount_rate > 1e-9
            else 0.0
        )
        churn_flag = 1.0 if discount_rate > self._prev_discount + 1e-9 else 0.0

        # Ara adım ödülü: (Gelir + Komisyon) / V_init - etiket değişim cezası
        step_reward = (gross_sale_tl + commission_tl) / self._v_init - (
            self.config.PRICE_CHURN_PENALTY * churn_flag
        )

        # Sayaçları güncelle
        self._ep_revenue_tl += gross_sale_tl
        self._ep_commission_tl += commission_tl
        self._ep_sold_units += int(outcome.batch_sold)
        self._ep_discount_sum += discount_rate
        self._ep_steps += 1
        if churn_flag > 0.0:
            self._ep_discount_changes += 1

        # 3) Stok, kuyruk ve hareketli satış hızını bir sonraki güne ilerlet
        self._remaining_qty = outcome.new_remaining_qty
        self._upstream_qty = outcome.new_upstream_qty
        self._daily_out_qty = outcome.realized_total_demand
        self._vel_3d = (2.0 * self._vel_3d + float(outcome.realized_total_demand)) / 3.0
        self._vel_7d = (6.0 * self._vel_7d + float(outcome.realized_total_demand)) / 7.0
        self._vel_14d = (13.0 * self._vel_14d + float(outcome.realized_total_demand)) / 14.0
        self._recent_discounts.append(discount_rate)
        if len(self._recent_discounts) > 3:
            self._recent_discounts.pop(0)
        self._prev_discount = discount_rate
        self._prev_action_idx = valid_action

        # 4) Terminal koşul kontrolü: Stok sıfırlandıysa VEYA indirim penceresi bittiyse
        terminated = (self._remaining_qty <= 0) or (self._dte <= self.config.DISCOUNT_STOP_DTE)
        truncated = False

        if terminated:
            # Terminal adım ödülü: Pencere sonunda satılamayan artık stok (W_res)
            # için hem finansal imha/devir cezası hem CO2 gölge fiyat cezası yazılır.
            w_res = float(self._remaining_qty)
            if w_res > 0.0:
                res_value_tl = w_res * self._ep.unit_price
                res_co2_kg = (
                    w_res * self._ep.unit_weight_kg * self._ep.co2_factor_kg_per_kg
                )
                terminal_penalty = (
                    res_value_tl * self.config.RESIDUAL_WASTE_PENALTY_MULT
                    + res_co2_kg * self.config.CO2_SHADOW_PRICE_TL_PER_KG
                ) / self._v_init
                step_reward -= terminal_penalty
        else:
            # Bir sonraki güne ilerle ve Faz 1 sinyallerini yenile
            self._current_date = self._current_date + timedelta(days=1)
            self._dte -= 1
            self._refresh_phase1_signals()

        obs = self._get_obs()
        info = self._build_step_info(terminated=terminated)
        return obs, float(step_reward), terminated, truncated, info

    def _build_step_info(self, terminated: bool) -> dict[str, Any]:
        """Ham iş, ESG ve kısıt metriklerini içeren `info` sözlüğünü üretir."""
        w_res = float(self._remaining_qty) if terminated else 0.0
        res_tl = w_res * self._ep.unit_price
        res_co2 = w_res * self._ep.unit_weight_kg * self._ep.co2_factor_kg_per_kg
        avg_disc = self._ep_discount_sum / max(self._ep_steps, 1)

        return {
            "batch_id": self._ep.batch_id,
            "category": self._ep.category,
            "store_id": self._ep.store_id,
            "days_to_expiry": self._dte,
            "remaining_qty": self._remaining_qty,
            "upstream_fefo_qty": self._upstream_qty,
            "unit_price": self._ep.unit_price,
            "p_hat": round(self._last_p_hat, 4),
            "q_hat": round(self._last_q_hat, 2),
            "risk_index": round(self._last_risk_index, 4),
            "prev_discount": self._prev_discount,
            "revenue_tl": round(self._ep_revenue_tl, 2),
            "equora_commission_tl": round(self._ep_commission_tl, 2),
            "sold_units": int(self._ep_sold_units),
            "residual_waste_units": int(round(w_res)),
            "residual_waste_tl": round(res_tl, 2),
            "residual_co2_kg": round(res_co2, 3),
            "avg_discount_applied": round(avg_disc, 4),
            "discount_steps": int(self._ep_steps),
            "discount_changes": int(self._ep_discount_changes),
            "monotonic_violations": int(self._ep_monotonic_violations),
        }

    # -------------------------------------------------------------------------
    # Yardımcı Salt-Okunur Erişimciler (Baselines / Analitik Politikalar İçin)
    # -------------------------------------------------------------------------
    @property
    def current_episode(self) -> BatchEpisode:
        return self._ep

    @property
    def current_base_mu(self) -> float:
        return self._base_mu

    @property
    def current_date(self) -> date:
        return self._current_date

    @property
    def current_dte(self) -> int:
        return self._dte

    @property
    def current_remaining_qty(self) -> int:
        return self._remaining_qty

    @property
    def current_upstream_qty(self) -> int:
        return self._upstream_qty

    @property
    def current_prev_discount(self) -> float:
        return self._prev_discount

    @property
    def current_p_hat(self) -> float:
        return self._last_p_hat

    @property
    def current_risk_index(self) -> float:
        return self._last_risk_index
