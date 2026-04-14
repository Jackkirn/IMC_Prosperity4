"""
OU Mean-Reversion Pairs Trading Strategy — Prosperity 4
========================================================
dynamic bands · dynamic sizing · spread filter · dual-regime vol filter

Changes vs v3:
  [4] DUAL-REGIME VOLATILITY FILTER  (replaces the flawed single-sided filter)
      Compares σ_short (fast EWMA) against σ_long (slow EWMA reference):

        vol_collapsed  →  σ_short < SIGMA_COLLAPSE_RATIO * σ_long
                          Market is unusually quiet / thin.
                          NOTE: already partially handled by dynamic bands,
                          but this kills entries before even computing z.

        vol_spike      →  σ_short > SIGMA_SPIKE_RATIO * σ_long
                          Volatility has exploded → possible regime change
                          or cointegration breakdown. This is the dangerous
                          case for mean-reversion: the spread may NOT revert.

      signal_ok = not vol_collapsed and not vol_spike
      Exits are NEVER gated by this filter.

  The two mechanisms are now complementary:
    - Dynamic bands  →  handles micro-structure noise (σ small but not anomalous)
    - Dual-regime    →  handles truly anomalous regimes in BOTH directions
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math


class Trader:

    # ═══════════════════════════════════════════════════════════
    # PRODUCTS
    # ═══════════════════════════════════════════════════════════
    PRODUCT_A = "EMERALDS"
    PRODUCT_B = "TOMATOES"

    POSITION_LIMITS: Dict[str, int] = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    # ═══════════════════════════════════════════════════════════
    # OU BAND PARAMETERS  (in σ units)
    # ═══════════════════════════════════════════════════════════
    ENTRY_D: float =  0.922   # base entry band
    EXIT_U:  float =  0.15    # exit when |z| < EXIT_U (back near mean)
    STOP_L:  float = -1.20    # stop loss

    # ═══════════════════════════════════════════════════════════
    # [1] DYNAMIC BANDS
    # ═══════════════════════════════════════════════════════════
    SIGMA_MIN_THRESHOLD: float = 0.001   # below this σ, entry band widens
    SIGMA_BAND_SCALE:    float = 0.5     # exponent — 0.5 = square-root dampening

    # ═══════════════════════════════════════════════════════════
    # [2] DYNAMIC POSITION SIZING
    # ═══════════════════════════════════════════════════════════
    TRADE_UNITS:     int   = 40     # max units per leg
    MIN_TRADE_UNITS: int   = 10     # floor — never trade less than this
    SIZE_SCALE:      float = 0.50   # extra σ above entry band for full size
    #   full size when |z| ≥ ENTRY_D + SIZE_SCALE  (e.g. 0.922 + 0.50 = 1.422)

    # ═══════════════════════════════════════════════════════════
    # [3] SPREAD FILTER  (entry only)
    # ═══════════════════════════════════════════════════════════
    MAX_BOOK_SPREAD: int = 4     # set to 9999 to disable

    # ═══════════════════════════════════════════════════════════
    # [4] DUAL-REGIME VOLATILITY FILTER  (entry only)
    # ═══════════════════════════════════════════════════════════
    EWMA_ALPHA:           float = 0.00664   # fast EWMA  ≈ N=300
    EWMA_SLOW_ALPHA:      float = 0.0015    # slow EWMA  ≈ N=1333  (long-run reference)
    SIGMA_COLLAPSE_RATIO: float = 0.50      # block if σ_short < 0.50 * σ_long
    SIGMA_SPIKE_RATIO:    float = 2.00      # block if σ_short > 2.00 * σ_long          THIS NEED TO BE CALIBRATED (pairs_check)
    MIN_HISTORY:          int   = 60        # ticks before trading

    # ═══════════════════════════════════════════════════════════
    # STATE LABELS
    # ═══════════════════════════════════════════════════════════
    FLAT:  str = "FLAT"
    LONG:  str = "LONG"
    SHORT: str = "SHORT"

    # ───────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result:      Dict[str, List[Order]] = {}
        conversions: int = 0

        ctx = self._load(state.traderData)

        # ── Mid prices ─────────────────────────────────────────
        mid_a = self._mid(state.order_depths.get(self.PRODUCT_A))
        mid_b = self._mid(state.order_depths.get(self.PRODUCT_B))

        if mid_a is None or mid_b is None:
            return result, conversions, json.dumps(ctx)

        # ── Update both EWMAs ──────────────────────────────────
        log_spread = math.log(mid_a) - math.log(mid_b)
        ctx, mu, sigma_short = self._update_ewma(ctx, log_spread)
        ctx, sigma_long      = self._update_slow_ewma(ctx, log_spread)

        if ctx["ticks"] < self.MIN_HISTORY:
            return result, conversions, json.dumps(ctx)

        if sigma_short < 1e-9:
            return result, conversions, json.dumps(ctx)

        # ── Z-score ────────────────────────────────────────────
        z = (log_spread - mu) / sigma_short
        ctx["last_z"]      = round(z, 5)
        ctx["sigma_short"] = round(sigma_short, 6)
        ctx["sigma_long"]  = round(sigma_long,  6)

        # ── [1] Effective entry band ───────────────────────────
        effective_entry = self._effective_entry_band(sigma_short)
        ctx["eff_entry"] = round(effective_entry, 5)

        # ── [4] Dual-regime volatility filter ──────────────────
        signal_ok = self._vol_regime_ok(sigma_short, sigma_long)
        ctx["signal_ok"] = int(signal_ok)

        # ── References ─────────────────────────────────────────
        pos_a        = state.position.get(self.PRODUCT_A, 0)
        pos_b        = state.position.get(self.PRODUCT_B, 0)
        spread_state = ctx["state"]

        oda = state.order_depths.get(self.PRODUCT_A)
        odb = state.order_depths.get(self.PRODUCT_B)

        orders_a: List[Order] = []
        orders_b: List[Order] = []

        # ══════════════════════════════════════════════════════
        # STATE MACHINE
        # ══════════════════════════════════════════════════════

        if spread_state == self.FLAT:

            # [3] + [4]: both filters must pass before any entry
            can_enter = self._books_acceptable(oda, odb) and signal_ok

            # ── Enter LONG spread (buy A, sell B) ──────────────
            if z < -effective_entry and can_enter:
                qty = self._dynamic_qty(pos_a, pos_b, +1, z, effective_entry)
                if qty > 0 and oda and odb:
                    ask_a = self._best_ask(oda)
                    bid_b = self._best_bid(odb)
                    if ask_a is not None and bid_b is not None:
                        orders_a.append(Order(self.PRODUCT_A,  ask_a,  qty))
                        orders_b.append(Order(self.PRODUCT_B,  bid_b, -qty))
                        ctx["state"]   = self.LONG
                        ctx["entry_z"] = round(z, 5)

            # ── Enter SHORT spread (sell A, buy B) ─────────────
            elif z > effective_entry and can_enter:
                qty = self._dynamic_qty(pos_a, pos_b, -1, z, effective_entry)
                if qty > 0 and oda and odb:
                    bid_a = self._best_bid(oda)
                    ask_b = self._best_ask(odb)
                    if bid_a is not None and ask_b is not None:
                        orders_a.append(Order(self.PRODUCT_A, bid_a, -qty))
                        orders_b.append(Order(self.PRODUCT_B, ask_b,  qty))
                        ctx["state"]   = self.SHORT
                        ctx["entry_z"] = round(z, 5)

        elif spread_state == self.LONG:
            # Exits are NEVER filtered
            close = (z >= -self.EXIT_U) or (z <= self.STOP_L)
            if close:
                qty_a = abs(pos_a)
                qty_b = abs(pos_b)
                if (qty_a > 0 or qty_b > 0) and oda and odb:
                    bid_a = self._best_bid(oda)
                    ask_b = self._best_ask(odb)
                    if bid_a is not None and ask_b is not None:
                        if qty_a > 0:
                            orders_a.append(Order(self.PRODUCT_A, bid_a, -qty_a))
                        if qty_b > 0:
                            orders_b.append(Order(self.PRODUCT_B, ask_b,  qty_b))
                        ctx["state"]  = self.FLAT
                        ctx["exit_z"] = round(z, 5)

        elif spread_state == self.SHORT:
            # Exits are NEVER filtered
            close = (z <= self.EXIT_U) or (z >= -self.STOP_L)
            if close:
                qty_a = abs(pos_a)
                qty_b = abs(pos_b)
                if (qty_a > 0 or qty_b > 0) and oda and odb:
                    ask_a = self._best_ask(oda)
                    bid_b = self._best_bid(odb)
                    if ask_a is not None and bid_b is not None:
                        if qty_a > 0:
                            orders_a.append(Order(self.PRODUCT_A, ask_a,  qty_a))
                        if qty_b > 0:
                            orders_b.append(Order(self.PRODUCT_B, bid_b, -qty_b))
                        ctx["state"]  = self.FLAT
                        ctx["exit_z"] = round(z, 5)

        # ── Pack results ────────────────────────────────────────
        if orders_a:
            result[self.PRODUCT_A] = orders_a
        if orders_b:
            result[self.PRODUCT_B] = orders_b

        return result, conversions, json.dumps(ctx)

    # ═══════════════════════════════════════════════════════════
    # [1] DYNAMIC BAND
    # ═══════════════════════════════════════════════════════════
    def _effective_entry_band(self, sigma: float) -> float:
        if self.SIGMA_MIN_THRESHOLD <= 0 or sigma >= self.SIGMA_MIN_THRESHOLD:
            return self.ENTRY_D
        ratio = self.SIGMA_MIN_THRESHOLD / sigma
        return self.ENTRY_D * (ratio ** self.SIGMA_BAND_SCALE)

    # ═══════════════════════════════════════════════════════════
    # [2] DYNAMIC POSITION SIZING
    # ═══════════════════════════════════════════════════════════
    def _dynamic_qty(self, pos_a: int, pos_b: int, direction: int, z: float, entry_band: float) -> int:
        excess = abs(z) - entry_band
        if excess <= 0:
            return 0
        scale = min(1.0, excess / self.SIZE_SCALE)
        base  = max(self.MIN_TRADE_UNITS, round(self.TRADE_UNITS * scale))
        lim   = self.POSITION_LIMITS
        if direction == +1:
            cap = min(lim[self.PRODUCT_A] - pos_a, lim[self.PRODUCT_B] + pos_b)
        else:
            cap = min(lim[self.PRODUCT_A] + pos_a, lim[self.PRODUCT_B] - pos_b)
        return max(0, min(base, cap))

    # ═══════════════════════════════════════════════════════════
    # [3] SPREAD FILTER
    # ═══════════════════════════════════════════════════════════
    def _books_acceptable(self, oda: Optional[OrderDepth], odb: Optional[OrderDepth]) -> bool:
        if oda is None or odb is None:
            return False
        sa = self._book_spread(oda)
        sb = self._book_spread(odb)
        if sa is None or sb is None:
            return False
        return sa <= self.MAX_BOOK_SPREAD and sb <= self.MAX_BOOK_SPREAD

    @staticmethod
    def _book_spread(od: OrderDepth) -> Optional[int]:
        bb = max(od.buy_orders.keys(),  default=None)
        ba = min(od.sell_orders.keys(), default=None)
        if bb is None or ba is None:
            return None
        return ba - bb

    # ═══════════════════════════════════════════════════════════
    # [4] DUAL-REGIME VOLATILITY FILTER
    # ═══════════════════════════════════════════════════════════
    def _vol_regime_ok(self, sigma_short: float, sigma_long: float) -> bool:
        """
        Returns True only when volatility is in a 'normal' regime:
          - not collapsed (σ_short too far below long-run level)
          - not spiking  (σ_short too far above long-run level)
        When sigma_long is near zero (early ticks) we skip the check.
        """
        if sigma_long < 1e-9:
            return True   # not enough history yet, don't block
        vol_collapsed = sigma_short < self.SIGMA_COLLAPSE_RATIO * sigma_long
        vol_spike     = sigma_short > self.SIGMA_SPIKE_RATIO    * sigma_long
        return not vol_collapsed and not vol_spike

    # ═══════════════════════════════════════════════════════════
    # EWMA  —  fast (mean + variance)
    # ═══════════════════════════════════════════════════════════
    def _update_ewma(self, ctx: dict, x: float):
        alpha = self.EWMA_ALPHA
        ctx["ticks"] += 1
        if ctx["ticks"] == 1:
            ctx["ewm_mean"] = x
            ctx["ewm_var"]  = 0.0
        else:
            delta           = x - ctx["ewm_mean"]
            ctx["ewm_mean"] += alpha * delta
            ctx["ewm_var"]   = (1 - alpha) * (ctx["ewm_var"] + alpha * delta ** 2)
        return ctx, ctx["ewm_mean"], math.sqrt(max(ctx["ewm_var"], 0.0))

    # ═══════════════════════════════════════════════════════════
    # EWMA  —  slow (variance only, used as long-run σ reference)
    # ═══════════════════════════════════════════════════════════
    def _update_slow_ewma(self, ctx: dict, x: float) -> tuple:
        alpha = self.EWMA_SLOW_ALPHA
        if ctx["ticks"] == 1:
            ctx["slow_mean"] = x
            ctx["slow_var"]  = 0.0
        else:
            delta             = x - ctx["slow_mean"]
            ctx["slow_mean"] += alpha * delta
            ctx["slow_var"]   = (1 - alpha) * (ctx["slow_var"] + alpha * delta ** 2)
        return ctx, math.sqrt(max(ctx["slow_var"], 0.0))

    # ═══════════════════════════════════════════════════════════
    # ORDER BOOK HELPERS
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _mid(od: Optional[OrderDepth]) -> Optional[float]:
        if od is None:
            return None
        bb = max(od.buy_orders.keys(),  default=None)
        ba = min(od.sell_orders.keys(), default=None)
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @staticmethod
    def _best_bid(od: OrderDepth) -> Optional[int]:
        return max(od.buy_orders.keys(), default=None)

    @staticmethod
    def _best_ask(od: OrderDepth) -> Optional[int]:
        return min(od.sell_orders.keys(), default=None)

    # ═══════════════════════════════════════════════════════════
    # STATE PERSISTENCE
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _load(raw: str) -> dict:
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return {
            "ticks":       0,
            "ewm_mean":    0.0,
            "ewm_var":     0.0,
            "slow_mean":   0.0,
            "slow_var":    0.0,
            "state":       "FLAT",
            "entry_z":     0.0,
            "exit_z":      0.0,
            "last_z":      0.0,
            "eff_entry":   0.0,
            "sigma_short": 0.0,
            "sigma_long":  0.0,
            "signal_ok":   1,
        }