"""
Backtest: OU Mean-Reversion Pairs Trading
==========================================
Simulates two correlated price paths:
  - EMERALDS : random walk with slight drift  (the "driver")
  - TOMATOES : EMERALDS * exp(−S_t)          (tracks EMERALDS via an OU spread)

The spread S_t = log(EMERALDS) − log(TOMATOES) follows an OU process by construction.
Then runs the Trader tick by tick and reports mark-to-market P&L.

Usage:
    python backtest_ou_pairs.py
"""

# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — inject mock 'datamodel' so ou_pairs_trader.py can import
#           it without the Prosperity runtime being present locally
# ═══════════════════════════════════════════════════════════════════
import sys, types

_dm = types.ModuleType("datamodel")

class _Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol   = symbol
        self.price    = price
        self.quantity = quantity          # + buy  / − sell
    def __repr__(self):
        side = "BUY" if self.quantity > 0 else "SELL"
        return f"Order({self.symbol} {side} {abs(self.quantity)} @ {self.price})"

class _OrderDepth:
    def __init__(self):
        self.buy_orders:  dict = {}   # price → +vol
        self.sell_orders: dict = {}   # price → −vol  (Prosperity convention)

class _TradingState:
    def __init__(self, timestamp, order_depths, position, traderData=""):
        self.timestamp    = timestamp
        self.order_depths = order_depths
        self.position     = position
        self.traderData   = traderData

_dm.Order        = _Order
_dm.OrderDepth   = _OrderDepth
_dm.TradingState = _TradingState
sys.modules["datamodel"] = _dm

# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — normal imports  (ou_pairs_trader lives in the same dir)
# ═══════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from ou_pairs_trader import Trader


# ═══════════════════════════════════════════════════════════════════
#  PRICE GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_prices(T: int = 6_000, seed: int = 42) -> tuple:
    """
    Returns (prices_A, prices_B, true_spread) all of length T.

    prices_A  : EMERALDS — random walk with slight upward drift
    prices_B  : TOMATOES = prices_A * exp(−spread_t)
    true_spread : OU process S_t  s.t.  log(A/B) = S_t
    """
    rng = np.random.default_rng(seed)

    # ── EMERALDS: linear drift + Gaussian noise ─────────────
    P0_A    = 10_000.0
    drift_A = 0.03           # price units per tick (slow linear trend)
    vol_A   = 4.0            # price std per tick

    prices_A = np.empty(T)
    prices_A[0] = P0_A
    for t in range(1, T):
        prices_A[t] = prices_A[t - 1] + drift_A + vol_A * rng.standard_normal()

    # ── OU spread: exact one-step simulation ────────────────
    # Parameters per tick (tick ≈ 30 min  →  dt ≈ 1/17520 year)
    # Chosen so that:
    #   Σ_stat = σ / sqrt(2κ) ≈ 0.015   (≈ 1.5 % of price)
    #   half-life ≈ 40 ticks
    k_step   = 0.017          # mean-reversion speed per tick
    eta_step = 0.0            # long-run mean of spread
    sig_step = 0.005          # diffusion per tick

    a  = np.exp(-k_step)
    sd = sig_step * np.sqrt((1.0 - a ** 2) / (2.0 * k_step))

    spread = np.empty(T)
    spread[0] = 0.0
    for t in range(1, T):
        spread[t] = (a * spread[t - 1]
                     + eta_step * (1.0 - a)
                     + sd * rng.standard_normal())

    # ── TOMATOES derived from EMERALDS + spread ─────────────
    prices_B = prices_A * np.exp(-spread)

    sigma_stat = sig_step / np.sqrt(2.0 * k_step)
    print(f"[SIM] OU params:  k={k_step:.4f}/tick  σ={sig_step:.4f}"
          f"  Σ_stat={sigma_stat:.5f}  half-life≈{np.log(2)/k_step:.0f} ticks")
    print(f"[SIM] Entry band (price units): d* = {Trader.ENTRY_D * sigma_stat:.5f}")
    print(f"[SIM] Exit  band (price units): u* = {Trader.EXIT_U  * sigma_stat:.5f}")

    return prices_A, prices_B, spread


# ═══════════════════════════════════════════════════════════════════
#  MARKET BUILDER  —  synthetic order-book from a mid price
# ═══════════════════════════════════════════════════════════════════

def make_book(mid: float, depth: int = 50) -> _OrderDepth:
    """One-tick bid-ask spread; 'depth' units on each side."""
    od = _OrderDepth()
    bid = int(mid - 0.5)
    ask = int(mid + 0.5)
    if ask == bid:
        ask += 1
    od.buy_orders[bid]  =  depth
    od.sell_orders[ask] = -depth        # Prosperity: negative = sell volume
    return od


# ═══════════════════════════════════════════════════════════════════
#  EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════

def execute(orders: dict, books: dict, positions: dict, cash: float) -> float:
    """
    Greedily match orders against the simulated book.
    Returns updated cash balance.
    """
    for symbol, order_list in orders.items():
        od = books.get(symbol)
        if od is None:
            continue
        for o in order_list:
            if o.quantity > 0:           # BUY — hit best ask
                best_ask = min(od.sell_orders.keys(), default=None)
                if best_ask is not None and best_ask <= o.price:
                    avail = abs(od.sell_orders[best_ask])
                    qty   = min(o.quantity, avail)
                    cash -= qty * best_ask
                    positions[symbol] = positions.get(symbol, 0) + qty

            elif o.quantity < 0:         # SELL — hit best bid
                best_bid = max(od.buy_orders.keys(), default=None)
                if best_bid is not None and best_bid >= o.price:
                    avail = od.buy_orders[best_bid]
                    qty   = min(abs(o.quantity), avail)
                    cash += qty * best_bid
                    positions[symbol] = positions.get(symbol, 0) - qty
    return cash


# ═══════════════════════════════════════════════════════════════════
#  BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_backtest(T: int = 6_000, seed: int = 42):
    prices_A, prices_B, true_spread = generate_prices(T=T, seed=seed)

    PROD_A = Trader.PRODUCT_A
    PROD_B = Trader.PRODUCT_B

    trader      = Trader()
    positions   = {PROD_A: 0, PROD_B: 0}
    cash        = 0.0
    trader_data = ""

    pnl_hist    = np.zeros(T)
    z_hist      = np.zeros(T)
    states      = []
    trade_log   = []     # (tick, side, z, pnl_at_exit)

    for t in range(T):
        book_a = make_book(prices_A[t])
        book_b = make_book(prices_B[t])
        books  = {PROD_A: book_a, PROD_B: book_b}

        state = _TradingState(
            timestamp    = t,
            order_depths = books,
            position     = dict(positions),
            traderData   = trader_data,
        )

        prev_state = json.loads(trader_data).get("state", "FLAT") if trader_data else "FLAT"

        orders, _, trader_data = trader.run(state)
        cash = execute(orders, books, positions, cash)

        # Mark-to-market
        mtm = (cash
               + positions[PROD_A] * prices_A[t]
               + positions[PROD_B] * prices_B[t])
        pnl_hist[t] = mtm

        ctx = json.loads(trader_data) if trader_data else {}
        z   = ctx.get("last_z", 0.0)
        z_hist[t] = z
        states.append(ctx.get("state", "FLAT"))

        # Log trade events
        cur_state = ctx.get("state", "FLAT")
        if prev_state == "FLAT" and cur_state != "FLAT":
            trade_log.append({"tick": t, "action": f"OPEN {cur_state}", "z": z, "pnl": mtm})
        elif prev_state != "FLAT" and cur_state == "FLAT":
            trade_log.append({"tick": t, "action": f"CLOSE {prev_state}", "z": z, "pnl": mtm})

    # ── Summary ────────────────────────────────────────────────
    states_arr   = np.array(states)
    n_long_ticks = np.sum(states_arr == "LONG")
    n_short_ticks= np.sum(states_arr == "SHORT")
    n_opens      = sum(1 for e in trade_log if "OPEN"  in e["action"])
    n_closes     = sum(1 for e in trade_log if "CLOSE" in e["action"])
    final_pnl    = pnl_hist[-1]

    print("\n" + "═" * 50)
    print("  OU PAIRS TRADING — BACKTEST SUMMARY")
    print("═" * 50)
    print(f"  Ticks simulated    : {T}")
    print(f"  Trades opened      : {n_opens}")
    print(f"  Trades closed      : {n_closes}")
    print(f"  Ticks LONG         : {n_long_ticks}")
    print(f"  Ticks SHORT        : {n_short_ticks}")
    print(f"  Ticks FLAT         : {T - n_long_ticks - n_short_ticks}")
    print(f"  Final P&L          : {final_pnl:+.2f}")
    print(f"  Max drawdown       : {_max_drawdown(pnl_hist):.2f}")
    print("═" * 50)

    _plot(prices_A, prices_B, true_spread, z_hist, pnl_hist, states_arr, trade_log, T)
    return pnl_hist, z_hist, states


# ═══════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════

def _plot(prices_A, prices_B, true_spread, z_hist, pnl_hist, states, trade_log, T):
    t_arr = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1.4, 1.8, 1.8]})
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#aaa")
        ax.spines[:].set_color("#333")

    def label(ax, text):
        ax.set_title(text, color="#ddd", fontsize=10, loc="left", pad=4)

    # ── Panel 1: Price paths ───────────────────────────────────
    ax = axes[0]
    ax.plot(t_arr, prices_A, color="#4da6ff", lw=0.8, label="EMERALDS (A)")
    ax2 = ax.twinx()
    ax2.plot(t_arr, prices_B, color="#ffa64d", lw=0.8, label="TOMATOES (B)")
    ax2.tick_params(colors="#aaa"); ax2.spines[:].set_color("#333")
    ax2.set_facecolor("#111111")
    ax.set_ylabel("EMERALDS", color="#4da6ff", fontsize=9)
    ax2.set_ylabel("TOMATOES", color="#ffa64d", fontsize=9)
    ax.legend(loc="upper left",  fontsize=8, facecolor="#222", labelcolor="#ccc")
    ax2.legend(loc="upper right", fontsize=8, facecolor="#222", labelcolor="#ccc")
    label(ax, "Synthetic Price Paths")

    # ── Panel 2: True OU spread ────────────────────────────────
    ax = axes[1]
    ax.plot(t_arr, true_spread, color="#b48ead", lw=0.8)
    ax.axhline(0, color="#555", lw=0.5)
    ax.set_ylabel("log(A/B)", color="#b48ead", fontsize=9)
    label(ax, "True OU Spread  S_t = log(EMERALDS / TOMATOES)")

    # ── Panel 3: Z-score + bands ───────────────────────────────
    ax = axes[2]
    ax.plot(t_arr, z_hist, color="#88c0d0", lw=0.7, alpha=0.9, label="z-score")

    trader = Trader()
    for level, col, ls, lbl in [
        ( trader.ENTRY_D,  "#e06c75", "--", f"+d  {+trader.ENTRY_D:.3f}"),
        (-trader.ENTRY_D,  "#e06c75", "--", f"−d  {-trader.ENTRY_D:.3f}"),
        ( trader.EXIT_U,   "#a3be8c", ":" , f"+u  {+trader.EXIT_U:.3f}"),
        (-trader.EXIT_U,   "#a3be8c", ":" , f"−u  {-trader.EXIT_U:.3f}"),
        (-trader.STOP_L,   "#ebcb8b", "-.", f"−l  {-trader.STOP_L:.3f}"),
        ( trader.STOP_L,   "#ebcb8b", "-.", f"+l  {+trader.STOP_L:.3f}"),
    ]:
        ax.axhline(level, color=col, ls=ls, lw=0.9, label=lbl)

    # Shade LONG / SHORT periods
    for lo, hi in _intervals(states == "LONG"):
        ax.axvspan(lo, hi, color="#4da6ff", alpha=0.08)
    for lo, hi in _intervals(states == "SHORT"):
        ax.axvspan(lo, hi, color="#ffa64d", alpha=0.08)

    ax.set_ylabel("z-score", color="#88c0d0", fontsize=9)
    ax.legend(fontsize=7, ncol=6, facecolor="#222", labelcolor="#ccc",
              loc="upper right")
    long_p  = mpatches.Patch(color="#4da6ff", alpha=0.4, label="LONG  spread")
    short_p = mpatches.Patch(color="#ffa64d", alpha=0.4, label="SHORT spread")
    ax.legend(handles=[long_p, short_p], fontsize=8, facecolor="#222",
              labelcolor="#ccc", loc="lower right")
    label(ax, "Normalised Spread z-score  +  OU Trading Bands")

    # ── Panel 4: P&L ──────────────────────────────────────────
    ax = axes[3]
    pnl = pnl_hist
    ax.plot(t_arr, pnl, color="#a3be8c", lw=1.2)
    ax.axhline(0, color="#555", lw=0.5)
    ax.fill_between(t_arr, pnl, 0, where=pnl >= 0, color="#a3be8c", alpha=0.25)
    ax.fill_between(t_arr, pnl, 0, where=pnl <  0, color="#e06c75", alpha=0.25)

    # Mark open / close events
    for ev in trade_log:
        color = "#4da6ff" if "LONG" in ev["action"] else "#ffa64d"
        marker = "^" if "OPEN" in ev["action"] else "v"
        ax.axvline(ev["tick"], color=color, lw=0.4, alpha=0.5)
        ax.plot(ev["tick"], ev["pnl"], marker, color=color, ms=5, alpha=0.8)

    ax.set_ylabel("P&L (mark-to-market)", color="#a3be8c", fontsize=9)
    ax.set_xlabel("Tick", color="#aaa", fontsize=9)
    label(ax, f"Portfolio P&L  |  Final: {pnl[-1]:+.1f}  "
              f"|  Max drawdown: {_max_drawdown(pnl):.1f}")

    fig.suptitle("OU Mean-Reversion Pairs Trading — Backtest",
                 color="white", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(os.path.dirname(__file__), "backtest_ou_pairs.png")
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    print(f"[PLOT] saved → {out}")


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _max_drawdown(pnl: np.ndarray) -> float:
    peak = pnl[0]
    max_dd = 0.0
    for v in pnl:
        peak = max(peak, v)
        max_dd = min(max_dd, v - peak)
    return max_dd          # negative number


def _intervals(mask: np.ndarray):
    """Yield (start, end) pairs where bool mask is True."""
    in_block, start = False, 0
    for i, v in enumerate(mask):
        if v and not in_block:
            start, in_block = i, True
        elif not v and in_block:
            yield start, i
            in_block = False
    if in_block:
        yield start, len(mask)


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_backtest(T=6_000, seed=42)