"""
OU Mean-Reversion Pairs Trading Strategy — Prosperity 4
========================================================
Based on Baviera & Santagostino (2019) stop-loss / optimal-band framework.

Trading logic (per-product):
  - Compute log-spread  S_t = log(P_A) - log(P_B)
  - Estimate online μ, σ  over the last LOOKBACK ticks
  - Normalize:  z = (S_t - μ) / σ

  FLAT → LONG  (buy A, sell B) when z  < −ENTRY_D   [spread below mean]
  FLAT → SHORT (sell A, buy B) when z  >  ENTRY_D   [spread above mean]
  LONG  → FLAT (take profit)   when z  ≥  EXIT_U    [spread recovered above mean]
  LONG  → FLAT (stop loss)     when z  ≤  STOP_L    [spread fell further]
  SHORT → FLAT (take profit)   when z  ≤ −EXIT_U    [spread recovered below mean]
  SHORT → FLAT (stop loss)     when z  ≥ −STOP_L    [spread rose further]

Parameters from project calibration (l = −1.645, f = 1):
  d* = 0.922 σ   (entry band)
  u* = 0.597 σ   (exit  band)
  l  = −1.645 σ  (stop-loss)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math


class Trader:

    # ═══════════════════════════════════════════════════════════
    # PRODUCTS  —  change to match actual Prosperity 4 products
    # ═══════════════════════════════════════════════════════════
    PRODUCT_A = "EMERALDS"   # leg 1
    PRODUCT_B = "TOMATOES"   # leg 2

    POSITION_LIMITS: Dict[str, int] = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    # ═══════════════════════════════════════════════════════════
    # OU BAND PARAMETERS  (all in σ units, from report Q(c))
    # ═══════════════════════════════════════════════════════════
    ENTRY_D: float =  0.922   # |d*|  enter when |z| > ENTRY_D
    EXIT_U:  float =  0.597   # u*    take-profit when |z| crosses EXIT_U
    STOP_L:  float = -1.645   # l     stop-loss threshold (negative)

    # ═══════════════════════════════════════════════════════════
    # EXECUTION PARAMETERS
    # ═══════════════════════════════════════════════════════════
    LOOKBACK:     int =  300   # rolling window for μ / σ estimation
    MIN_HISTORY:  int =   60   # minimum ticks before trading
    TRADE_UNITS:  int =   40   # units per leg (≤ POSITION_LIMIT / 2)

    # ═══════════════════════════════════════════════════════════
    # STATE LABELS
    # ═══════════════════════════════════════════════════════════
    FLAT:  str = "FLAT"
    LONG:  str = "LONG"    # long  A / short B
    SHORT: str = "SHORT"   # short A / long  B

    # ───────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result:     Dict[str, List[Order]] = {}
        conversions: int = 0

        ctx = self._load(state.traderData)

        # ── Mid prices ─────────────────────────────────────────
        mid_a = self._mid(state.order_depths.get(self.PRODUCT_A))
        mid_b = self._mid(state.order_depths.get(self.PRODUCT_B))

        if mid_a is None or mid_b is None:
            return result, conversions, json.dumps(ctx)

        # ── Update rolling spread history ───────────────────────
        log_spread = math.log(mid_a) - math.log(mid_b)
        ctx["hist"].append(log_spread)
        if len(ctx["hist"]) > self.LOOKBACK:
            ctx["hist"] = ctx["hist"][-self.LOOKBACK:]

        if len(ctx["hist"]) < self.MIN_HISTORY:
            return result, conversions, json.dumps(ctx)

        # ── Compute z-score ─────────────────────────────────────
        mu, sigma = self._stats(ctx["hist"])
        if sigma < 1e-9:
            return result, conversions, json.dumps(ctx)

        z = (log_spread - mu) / sigma
        ctx["last_z"] = z

        # ── Positions and order-book references ─────────────────
        pos_a  = state.position.get(self.PRODUCT_A, 0)
        pos_b  = state.position.get(self.PRODUCT_B, 0)
        spread_state = ctx["state"]

        oda = state.order_depths.get(self.PRODUCT_A)
        odb = state.order_depths.get(self.PRODUCT_B)

        orders_a: List[Order] = []
        orders_b: List[Order] = []

        # ══════════════════════════════════════════════════════
        # STATE MACHINE
        # ══════════════════════════════════════════════════════

        if spread_state == self.FLAT:

            # ── Enter LONG spread ──────────────────────────────
            # Spread is z < −d* below mean  →  expect reversion up
            if z < -self.ENTRY_D:
                qty = self._open_qty(pos_a, pos_b, +1)
                if qty > 0 and oda and odb:
                    ask_a = self._best_ask(oda)
                    bid_b = self._best_bid(odb)
                    if ask_a is not None and bid_b is not None:
                        orders_a.append(Order(self.PRODUCT_A, ask_a,  qty))
                        orders_b.append(Order(self.PRODUCT_B, bid_b, -qty))
                        ctx["state"]   = self.LONG
                        ctx["entry_z"] = z

            # ── Enter SHORT spread ─────────────────────────────
            # Spread is z > +d* above mean  →  expect reversion down
            elif z > self.ENTRY_D:
                qty = self._open_qty(pos_a, pos_b, -1)
                if qty > 0 and oda and odb:
                    bid_a = self._best_bid(oda)
                    ask_b = self._best_ask(odb)
                    if bid_a is not None and ask_b is not None:
                        orders_a.append(Order(self.PRODUCT_A, bid_a, -qty))
                        orders_b.append(Order(self.PRODUCT_B, ask_b,  qty))
                        ctx["state"]   = self.SHORT
                        ctx["entry_z"] = z

        elif spread_state == self.LONG:
            # Entered LONG when z was below −d*
            # Take profit: z has reverted past +u*
            # Stop loss  : z fell further below l (even more negative)
            close = (z >= self.EXIT_U) or (z <= self.STOP_L)

            if close:
                qty = abs(pos_a)          # unwind whatever we hold
                if qty > 0 and oda and odb:
                    bid_a = self._best_bid(oda)
                    ask_b = self._best_ask(odb)
                    if bid_a is not None and ask_b is not None:
                        orders_a.append(Order(self.PRODUCT_A, bid_a, -qty))
                        orders_b.append(Order(self.PRODUCT_B, ask_b,  qty))
                        ctx["state"] = self.FLAT
                        ctx["exit_z"] = z

        elif spread_state == self.SHORT:
            # Entered SHORT when z was above +d*
            # Take profit: z has reverted back below −u*
            # Stop loss  : z rose further above −l (positive)
            close = (z <= -self.EXIT_U) or (z >= -self.STOP_L)

            if close:
                qty = abs(pos_a)
                if qty > 0 and oda and odb:
                    ask_a = self._best_ask(oda)
                    bid_b = self._best_bid(odb)
                    if ask_a is not None and bid_b is not None:
                        orders_a.append(Order(self.PRODUCT_A, ask_a,  qty))
                        orders_b.append(Order(self.PRODUCT_B, bid_b, -qty))
                        ctx["state"] = self.FLAT
                        ctx["exit_z"] = z

        # ── Pack results ────────────────────────────────────────
        if orders_a:
            result[self.PRODUCT_A] = orders_a
        if orders_b:
            result[self.PRODUCT_B] = orders_b

        return result, conversions, json.dumps(ctx)

    # ═══════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════

    def _open_qty(self, pos_a: int, pos_b: int, direction: int) -> int:
        """
        Max tradeable units for a given direction (+1 long, −1 short),
        clipped to TRADE_UNITS and position limits.
        """
        lim = self.POSITION_LIMITS
        if direction == +1:          # buy A, sell B
            cap_a = lim[self.PRODUCT_A] - pos_a
            cap_b = lim[self.PRODUCT_B] + pos_b
        else:                        # sell A, buy B
            cap_a = lim[self.PRODUCT_A] + pos_a
            cap_b = lim[self.PRODUCT_B] - pos_b
        return max(0, min(cap_a, cap_b, self.TRADE_UNITS))

    @staticmethod
    def _stats(hist: list):
        """Return (mean, std) of the history list."""
        n  = len(hist)
        mu = sum(hist) / n
        var = sum((x - mu) ** 2 for x in hist) / n
        return mu, var ** 0.5

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

    @staticmethod
    def _load(raw: str) -> dict:
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return {
            "hist":    [],
            "state":   "FLAT",
            "entry_z": 0.0,
            "exit_z":  0.0,
            "last_z":  0.0,
        }