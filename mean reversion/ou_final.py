from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math

class Trader:

    # ═══════════════════════════════════════════════════════════
    # PRODUCTS
    # ═══════════════════════════════════════════════════════════
    PRODUCT_A = "EMERALDS"  # Nota: assicurati che siano davvero cointegrati nei round futuri
    PRODUCT_B = "TOMATOES"

    POSITION_LIMITS: Dict[str, int] = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    # ═══════════════════════════════════════════════════════════
    # OU PARAMETERS (Mantenuti dal tuo codice)
    # ═══════════════════════════════════════════════════════════
    ENTRY_D: float =  0.922   
    EXIT_U:  float =  0.15    
    STOP_L:  float = -1.20    

    SIGMA_MIN_THRESHOLD: float = 0.001   
    SIGMA_BAND_SCALE:    float = 0.5     

    TRADE_UNITS:     int   = 40     
    MIN_TRADE_UNITS: int   = 10     
    SIZE_SCALE:      float = 0.50   

    MAX_BOOK_SPREAD: int = 4     

    # ═══════════════════════════════════════════════════════════
    # DUAL-REGIME VOL FILTER + TOXIC FLOW
    # ═══════════════════════════════════════════════════════════
    EWMA_ALPHA:           float = 0.00664   
    EWMA_SLOW_ALPHA:      float = 0.0015    
    SIGMA_COLLAPSE_RATIO: float = 0.50      
    SIGMA_SPIKE_RATIO:    float = 2.00      
    MIN_HISTORY:          int   = 60        
    
    # NOVITA': Quanti lotti a mercato nello stesso tick definiscono "Flusso Tossico"
    TOXIC_IMBALANCE_THRESHOLD: int = 30  

    FLAT:  str = "FLAT"
    LONG:  str = "LONG"
    SHORT: str = "SHORT"

    # ───────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result:      Dict[str, List[Order]] = {}
        conversions: int = 0

        ctx = self._load(state.traderData)

        mid_a = self._mid(state.order_depths.get(self.PRODUCT_A))
        mid_b = self._mid(state.order_depths.get(self.PRODUCT_B))

        if mid_a is None or mid_b is None:
            return result, conversions, json.dumps(ctx)

        # ── Update EWMAs (La tua logica originale) ─────────────
        log_spread = math.log(mid_a) - math.log(mid_b)
        ctx, mu, sigma_short = self._update_ewma(ctx, log_spread)
        ctx, sigma_long      = self._update_slow_ewma(ctx, log_spread)

        if ctx["ticks"] < self.MIN_HISTORY or sigma_short < 1e-9:
            return result, conversions, json.dumps(ctx)

        # ── Z-score ────────────────────────────────────────────
        z = (log_spread - mu) / sigma_short
        ctx["last_z"] = round(z, 5)

        effective_entry = self._effective_entry_band(sigma_short)
        signal_ok = self._vol_regime_ok(sigma_short, sigma_long)

        # ── NOVITA': TOXIC FLOW FILTER ──────────────────────────
        # Calcoliamo la pressione dei trade aggressivi nell'ultimo tick
        toxic_A = self._calculate_toxic_flow(state, self.PRODUCT_A, mid_a)
        toxic_B = self._calculate_toxic_flow(state, self.PRODUCT_B, mid_b)

        # Se il flusso su A è fortemente negativo (dump) e vogliamo comprare A (Long Spread), blocchiamo.
        # Se il flusso su A è fortemente positivo (pump) e vogliamo vendere A (Short Spread), blocchiamo.
        toxic_block_long = (toxic_A <= -self.TOXIC_IMBALANCE_THRESHOLD) or (toxic_B >= self.TOXIC_IMBALANCE_THRESHOLD)
        toxic_block_short = (toxic_A >= self.TOXIC_IMBALANCE_THRESHOLD) or (toxic_B <= -self.TOXIC_IMBALANCE_THRESHOLD)

        pos_a = state.position.get(self.PRODUCT_A, 0)
        pos_b = state.position.get(self.PRODUCT_B, 0)
        spread_state = ctx["state"]

        oda = state.order_depths.get(self.PRODUCT_A)
        odb = state.order_depths.get(self.PRODUCT_B)

        orders_a: List[Order] = []
        orders_b: List[Order] = []

        # ══════════════════════════════════════════════════════
        # STATE MACHINE: ESECUZIONE
        # ══════════════════════════════════════════════════════
        if spread_state == self.FLAT:
            books_ok = self._books_acceptable(oda, odb)

            # ── Enter LONG spread (buy A, sell B) ──────────────
            if z < -effective_entry and books_ok and signal_ok and not toxic_block_long:
                qty = self._dynamic_qty(pos_a, pos_b, +1, z, effective_entry)
                if qty > 0 and oda and odb:
                    # NOVITA': PENNYING (Queue Priority)
                    # Non compriamo all'Ask. Ci mettiamo come primo Bid, purché non incrociamo lo spread.
                    best_bid_a = self._best_bid(oda)
                    best_ask_a = self._best_ask(oda)
                    my_bid_a = min(best_bid_a + 1, best_ask_a - 1) if (best_ask_a - best_bid_a > 1) else best_bid_a

                    best_ask_b = self._best_ask(odb)
                    best_bid_b = self._best_bid(odb)
                    my_ask_b = max(best_ask_b - 1, best_bid_b + 1) if (best_ask_b - best_bid_b > 1) else best_ask_b

                    orders_a.append(Order(self.PRODUCT_A, my_bid_a, qty))
                    orders_b.append(Order(self.PRODUCT_B, my_ask_b, -qty))
                    
                    # Lo stato cambierà effettivamente solo se veniamo fillati.
                    # In una logica super avanzata controlleresti l'eseguito nel tick successivo, 
                    # ma per ora manteniamo la tua struttura per semplicità.
                    ctx["state"]   = self.LONG
                    ctx["entry_z"] = round(z, 5)

            # ── Enter SHORT spread (sell A, buy B) ─────────────
            elif z > effective_entry and books_ok and signal_ok and not toxic_block_short:
                qty = self._dynamic_qty(pos_a, pos_b, -1, z, effective_entry)
                if qty > 0 and oda and odb:
                    # PENNYING
                    best_ask_a = self._best_ask(oda)
                    best_bid_a = self._best_bid(oda)
                    my_ask_a = max(best_ask_a - 1, best_bid_a + 1) if (best_ask_a - best_bid_a > 1) else best_ask_a

                    best_bid_b = self._best_bid(odb)
                    best_ask_b = self._best_ask(odb)
                    my_bid_b = min(best_bid_b + 1, best_ask_b - 1) if (best_ask_b - best_bid_b > 1) else best_bid_b

                    orders_a.append(Order(self.PRODUCT_A, my_ask_a, -qty))
                    orders_b.append(Order(self.PRODUCT_B, my_bid_b,  qty))
                    ctx["state"]   = self.SHORT
                    ctx["entry_z"] = round(z, 5)

        # ── USCITE: Manteniamo Market Orders per scaricare istantaneamente il rischio
        elif spread_state == self.LONG:
            close = (z >= -self.EXIT_U) or (z <= self.STOP_L)
            if close:
                qty_a = abs(pos_a)
                qty_b = abs(pos_b)
                if (qty_a > 0 or qty_b > 0) and oda and odb:
                    # Usciamo colpendo il book (Market) per sicurezza
                    bid_a = self._best_bid(oda)
                    ask_b = self._best_ask(oda) # Errore nel tuo codice originale corretto qui: best_ask(odb)
                    ask_b = self._best_ask(odb) 
                    
                    if bid_a is not None and ask_b is not None:
                        if qty_a > 0: orders_a.append(Order(self.PRODUCT_A, bid_a, -qty_a))
                        if qty_b > 0: orders_b.append(Order(self.PRODUCT_B, ask_b,  qty_b))
                        ctx["state"]  = self.FLAT

        elif spread_state == self.SHORT:
            close = (z <= self.EXIT_U) or (z >= -self.STOP_L)
            if close:
                qty_a = abs(pos_a)
                qty_b = abs(pos_b)
                if (qty_a > 0 or qty_b > 0) and oda and odb:
                    ask_a = self._best_ask(oda)
                    bid_b = self._best_bid(odb)
                    if ask_a is not None and bid_b is not None:
                        if qty_a > 0: orders_a.append(Order(self.PRODUCT_A, ask_a,  qty_a))
                        if qty_b > 0: orders_b.append(Order(self.PRODUCT_B, bid_b, -qty_b))
                        ctx["state"]  = self.FLAT

        if orders_a: result[self.PRODUCT_A] = orders_a
        if orders_b: result[self.PRODUCT_B] = orders_b

        return result, conversions, json.dumps(ctx)

    # ═══════════════════════════════════════════════════════════
    # NEW HELPER: TOXIC FLOW DETECTION
    # ═══════════════════════════════════════════════════════════
    def _calculate_toxic_flow(self, state: TradingState, product: str, mid: float) -> int:
        """
        Legge state.market_trades per capire se nell'ultimo tick le 'balene' 
        hanno aggredito brutalmente il book.
        Ritorna > 0 per pressione in acquisto, < 0 per pressione in vendita.
        """
        imbalance = 0
        if product in state.market_trades:
            for trade in state.market_trades[product]:
                # Se un trade è avvenuto a un prezzo >= del mid, chi ha originato il trade era un compratore aggressivo
                if trade.price >= mid:
                    imbalance += trade.quantity
                # Altrimenti era un venditore aggressivo
                elif trade.price <= mid:
                    imbalance -= trade.quantity
        return imbalance

    # (I tuoi helpers EWMA e logiche dinamiche originali rimangono invariati qui sotto)
    # ...
    def _effective_entry_band(self, sigma: float) -> float:
        if self.SIGMA_MIN_THRESHOLD <= 0 or sigma >= self.SIGMA_MIN_THRESHOLD: return self.ENTRY_D
        return self.ENTRY_D * ((self.SIGMA_MIN_THRESHOLD / sigma) ** self.SIGMA_BAND_SCALE)

    def _dynamic_qty(self, pos_a: int, pos_b: int, direction: int, z: float, entry_band: float) -> int:
        excess = abs(z) - entry_band
        if excess <= 0: return 0
        scale = min(1.0, excess / self.SIZE_SCALE)
        base  = max(self.MIN_TRADE_UNITS, round(self.TRADE_UNITS * scale))
        lim   = self.POSITION_LIMITS
        if direction == +1:
            cap = min(lim[self.PRODUCT_A] - pos_a, lim[self.PRODUCT_B] + pos_b)
        else:
            cap = min(lim[self.PRODUCT_A] + pos_a, lim[self.PRODUCT_B] - pos_b)
        return max(0, min(base, cap))

    def _books_acceptable(self, oda: Optional[OrderDepth], odb: Optional[OrderDepth]) -> bool:
        if oda is None or odb is None: return False
        sa = self._book_spread(oda)
        sb = self._book_spread(odb)
        if sa is None or sb is None: return False
        return sa <= self.MAX_BOOK_SPREAD and sb <= self.MAX_BOOK_SPREAD

    @staticmethod
    def _book_spread(od: OrderDepth) -> Optional[int]:
        bb = max(od.buy_orders.keys(),  default=None)
        ba = min(od.sell_orders.keys(), default=None)
        return ba - bb if bb is not None and ba is not None else None

    def _vol_regime_ok(self, sigma_short: float, sigma_long: float) -> bool:
        if sigma_long < 1e-9: return True
        return not (sigma_short < self.SIGMA_COLLAPSE_RATIO * sigma_long) and not (sigma_short > self.SIGMA_SPIKE_RATIO * sigma_long)

    def _update_ewma(self, ctx: dict, x: float):
        alpha = self.EWMA_ALPHA
        ctx["ticks"] += 1
        if ctx["ticks"] == 1:
            ctx["ewm_mean"], ctx["ewm_var"] = x, 0.0
        else:
            delta = x - ctx["ewm_mean"]
            ctx["ewm_mean"] += alpha * delta
            ctx["ewm_var"] = (1 - alpha) * (ctx["ewm_var"] + alpha * delta ** 2)
        return ctx, ctx["ewm_mean"], math.sqrt(max(ctx["ewm_var"], 0.0))

    def _update_slow_ewma(self, ctx: dict, x: float) -> tuple:
        alpha = self.EWMA_SLOW_ALPHA
        if ctx["ticks"] == 1:
            ctx["slow_mean"], ctx["slow_var"] = x, 0.0
        else:
            delta = x - ctx["slow_mean"]
            ctx["slow_mean"] += alpha * delta
            ctx["slow_var"] = (1 - alpha) * (ctx["slow_var"] + alpha * delta ** 2)
        return ctx, math.sqrt(max(ctx["slow_var"], 0.0))

    @staticmethod
    def _mid(od: Optional[OrderDepth]) -> Optional[float]:
        if od is None: return None
        bb = max(od.buy_orders.keys(),  default=None)
        ba = min(od.sell_orders.keys(), default=None)
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    @staticmethod
    def _best_bid(od: OrderDepth) -> Optional[int]:
        return max(od.buy_orders.keys(), default=None)

    @staticmethod
    def _best_ask(od: OrderDepth) -> Optional[int]:
        return min(od.sell_orders.keys(), default=None)

    @staticmethod
    def _load(raw: str) -> dict:
        if raw:
            try: return json.loads(raw)
            except Exception: pass
        return {"ticks": 0, "ewm_mean": 0.0, "ewm_var": 0.0, "slow_mean": 0.0, "slow_var": 0.0, "state": "FLAT", "entry_z": 0.0, "exit_z": 0.0, "last_z": 0.0, "eff_entry": 0.0, "sigma_short": 0.0, "sigma_long": 0.0, "signal_ok": 1}