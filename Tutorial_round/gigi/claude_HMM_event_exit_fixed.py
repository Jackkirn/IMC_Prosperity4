from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import math
import json


class Trader:
    # ==========================================================
    # PRODUCTS
    # ==========================================================
    EMERALDS = "EMERALDS"
    TOMATOES  = "TOMATOES"

    ENABLE_TAKING = True
    ENABLE_MAKING = True

    # ==========================================================
    # POSITION LIMITS
    # ==========================================================
    POSITION_LIMITS = {
        EMERALDS: 80,
        TOMATOES: 80,
    }

    # ==========================================================
    # PARAMETERS — EMERALDS
    # ==========================================================
    # fair = 10000 hardcoded: valore fondamentale noto.
    # Il taking cattura l'arb puro (ask≤10000 o bid≥10000),
    # il r-ratio protegge correttamente sui 59 step con spread=8.
    EMERALDS_FAIR             = 10_000
    EMERALDS_TAKE_THRESHOLD   = 0
    EMERALDS_MAX_PASSIVE_SIZE = 80
    EMERALDS_MM_R_LOW         = 0.30
    EMERALDS_MM_R_HIGH        = 0.70

    # ==========================================================
    # PARAMETERS — TOMATOES
    # ==========================================================
    TOMATOES_TAKE_THRESHOLD   = 0
    TOMATOES_MAX_PASSIVE_SIZE = 80
    TOMATOES_MM_R_LOW         = 0.30
    TOMATOES_MM_R_HIGH        = 0.70

    # ----------------------------------------------------------
    # Fair engine aligned with GIGI_current_best:
    #
    #   fair = microprice_L1 + lambda * ridge_prediction
    #
    # No blended anchor, no tanh saturation, no extra nonlinear cap.
    # ----------------------------------------------------------
    TOMATOES_RIDGE_LAMBDA    = 0.0

    # Coefficienti ridge (target: mid_{t+1}−microprice_t,
    # calibrati su day=−2, validati su day=−1, no overfitting)
    _TOM_INTERCEPT  =  0.276679873589467
    _TOM_COEF_IMBT  = -32.853054946602477
    _TOM_COEF_IMB2  =  28.554217165148906
    _TOM_COEF_IMB1  =  21.507001248247061
    _TOM_COEF_ME    =  -3.607552682276733
    _TOM_COEF_SP    =  -0.021753165199018

    # ----------------------------------------------------------
    # Narrow-spread opportunistic taking (derivato dall'HMM)
    #
    # Quando lo spread è stretto il mercato è in un regime
    # di momentum/mean-reversion locale con segnale forte.
    # Analisi HMM su dati storici:
    #   spread≤6 + imbtot>0 → EV_buy  = +1.38 tick, acc=98.3%  (n=119)
    #   spread≤7 + imbtot<0 → EV_sell = +0.42 tick, acc=96.4%  (n=137)
    #
    # EV positivo solo con questi threshold (spread 7−9 ha EV <0).
    # ----------------------------------------------------------
    NARROW_BUY_SPREAD_MAX  = 6    # spread massimo per aprire long
    NARROW_SELL_SPREAD_MAX = 7    # spread massimo per aprire short
    NARROW_TAKE_SIZE       = 10   # unità per singola operazione narrow
    EVENT_EXIT_SIZE        = 20   # size massima di chiusura quando l'evento finisce

    TOMATOES_EVENT_NEUTRAL = 0
    TOMATOES_EVENT_LONG    = 1
    TOMATOES_EVENT_SHORT   = -1

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        memory = self._load_memory(state.traderData)
        self._trade_emeralds(state, result)
        self._trade_tomatoes(state, result, memory)
        return result, 0, self._dump_memory(memory)

    # ==========================================================
    # MEMORY HELPERS
    # ==========================================================
    def _load_memory(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            data = json.loads(trader_data)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _dump_memory(self, memory: dict) -> str:
        try:
            return json.dumps(memory, separators=(",", ":"))
        except Exception:
            return "{}"

    # ==========================================================
    # ORDER BOOK HELPERS
    # ==========================================================
    def _best_bid(self, od: OrderDepth) -> Optional[int]:
        return max(od.buy_orders.keys(), default=None)

    def _best_ask(self, od: OrderDepth) -> Optional[int]:
        return min(od.sell_orders.keys(), default=None)

    def _midprice(self, od: OrderDepth) -> Optional[float]:
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def _microprice_l1(self, od: OrderDepth) -> Optional[float]:
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return None
        bv =  od.buy_orders.get(bb, 0)
        av = -od.sell_orders.get(ba, 0)
        if bv < 0: bv = 0
        if av < 0: av = 0
        den = bv + av
        if den <= 0:
            return self._midprice(od)
        return (bb * av + ba * bv) / den

    def _level2_vols(self, od: OrderDepth) -> tuple[int, int, float, float]:
        """
        Ritorna (bid_vol_2, ask_vol_2, bid_price_2, ask_price_2).
        Se il livello 2 non esiste usa i prezzi del livello 1
        in modo che il microprice L2 collassi su quello L1.
        """
        bid_prices = sorted(od.buy_orders.keys(), reverse=True)
        ask_prices = sorted(od.sell_orders.keys())
        if len(bid_prices) >= 2 and len(ask_prices) >= 2:
            bp2 = bid_prices[1]; ap2 = ask_prices[1]
            bv2 =  od.buy_orders.get(bp2, 0)
            av2 = -od.sell_orders.get(ap2, 0)
            if bv2 < 0: bv2 = 0
            if av2 < 0: av2 = 0
            return bv2, av2, float(bp2), float(ap2)
        bb = bid_prices[0] if bid_prices else 0
        ba = ask_prices[0] if ask_prices else 0
        return 0, 0, float(bb), float(ba)

    def _anchor_lob(self, od: OrderDepth) -> Optional[float]:
        """
        Microprice pesato su L1 e L2 con pesi proporzionali a sqrt(volume).
        Cattura la profondità del book senza dipendere da L3.

        m_k = (bid_k·ask_vol_k + ask_k·bid_vol_k) / (bid_vol_k + ask_vol_k)
        w_k = sqrt(bid_vol_k + ask_vol_k)
        anchor = (w1·m1 + w2·m2) / (w1 + w2)
        """
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        mid = self._midprice(od)
        if bb is None or ba is None or mid is None:
            return mid
        bv1 =  od.buy_orders.get(bb, 0)
        av1 = -od.sell_orders.get(ba, 0)
        if bv1 < 0: bv1 = 0
        if av1 < 0: av1 = 0
        V1 = bv1 + av1
        m1 = (bb * av1 + ba * bv1) / V1 if V1 > 0 else float(mid)
        w1 = math.sqrt(max(V1, 0))
        bv2, av2, bp2, ap2 = self._level2_vols(od)
        V2 = bv2 + av2
        m2 = (bp2 * av2 + ap2 * bv2) / V2 if V2 > 0 else float(mid)
        w2 = math.sqrt(max(V2, 0))
        wsum = w1 + w2
        if wsum <= 0:
            return mid
        return (w1 * m1 + w2 * m2) / wsum

    def _position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def _buy_capacity(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] - pos

    def _sell_capacity(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] + pos

    # ==========================================================
    # TAKING HELPERS
    # ==========================================================
    def _take_asks_below(
        self, product: str, od: OrderDepth, buy_cap: int, max_ask: int
    ) -> tuple[List[Order], int]:
        orders: List[Order] = []
        for ask in sorted(od.sell_orders.keys()):
            if buy_cap <= 0:
                break
            if ask <= max_ask:
                qty = min(-od.sell_orders[ask], buy_cap)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap -= qty
            else:
                break
        return orders, buy_cap

    def _take_bids_above(
        self, product: str, od: OrderDepth, sell_cap: int, min_bid: int
    ) -> tuple[List[Order], int]:
        orders: List[Order] = []
        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if sell_cap <= 0:
                break
            if bid >= min_bid:
                qty = min(od.buy_orders[bid], sell_cap)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
            else:
                break
        return orders, sell_cap

    def _detect_narrow_event(self, od: OrderDepth) -> int:
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return self.TOMATOES_EVENT_NEUTRAL

        spread = ba - bb
        if spread <= 0:
            return self.TOMATOES_EVENT_NEUTRAL

        bv1 = od.buy_orders.get(bb, 0)
        av1 = -od.sell_orders.get(ba, 0)
        if bv1 < 0:
            bv1 = 0
        if av1 < 0:
            av1 = 0

        bv2, av2, _, _ = self._level2_vols(od)
        bid_tot = bv1 + bv2
        ask_tot = av1 + av2
        den_tot = bid_tot + ask_tot
        imbtot = (bid_tot - ask_tot) / den_tot if den_tot > 0 else 0.0

        if spread <= self.NARROW_BUY_SPREAD_MAX and imbtot > 0:
            return self.TOMATOES_EVENT_LONG
        if spread <= self.NARROW_SELL_SPREAD_MAX and imbtot < 0:
            return self.TOMATOES_EVENT_SHORT
        return self.TOMATOES_EVENT_NEUTRAL

    def _event_exit_orders(
        self,
        product: str,
        od: OrderDepth,
        event_side: int,
        event_qty: int,
        buy_cap: int,
        sell_cap: int,
    ) -> tuple[List[Order], int, int, int]:
        orders: List[Order] = []
        remaining_event_qty = max(0, event_qty)
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if remaining_event_qty <= 0 or bb is None or ba is None:
            return orders, buy_cap, sell_cap, remaining_event_qty

        if event_side == self.TOMATOES_EVENT_LONG and sell_cap > 0:
            avail = od.buy_orders.get(bb, 0)
            qty = min(remaining_event_qty, sell_cap, self.EVENT_EXIT_SIZE, max(avail, 0))
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                sell_cap -= qty
                remaining_event_qty -= qty

        elif event_side == self.TOMATOES_EVENT_SHORT and buy_cap > 0:
            avail = -od.sell_orders.get(ba, 0)
            qty = min(remaining_event_qty, buy_cap, self.EVENT_EXIT_SIZE, max(avail, 0))
            if qty > 0:
                orders.append(Order(product, ba, qty))
                buy_cap -= qty
                remaining_event_qty -= qty

        return orders, buy_cap, sell_cap, remaining_event_qty

    def _narrow_spread_taking(
        self,
        product: str,
        od: OrderDepth,
        buy_cap: int,
        sell_cap: int,
        event_side: int,
    ) -> tuple[List[Order], int, int, int]:
        """
        Esegue taking solo nella direzione dell'evento già rilevato fuori.
        Qui non rifacciamo detection: questa funzione è solo l'executor.
        Ritorna anche la quantità evento aperta in questo step.
        """
        orders: List[Order] = []
        event_opened_qty = 0

        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return orders, buy_cap, sell_cap, event_opened_qty

        if event_side == self.TOMATOES_EVENT_LONG and buy_cap > 0:
            avail_at_ask = max(-od.sell_orders.get(ba, 0), 0)
            qty = min(avail_at_ask, buy_cap, self.NARROW_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, ba, qty))
                buy_cap -= qty
                event_opened_qty = qty

        elif event_side == self.TOMATOES_EVENT_SHORT and sell_cap > 0:
            avail_at_bid = max(od.buy_orders.get(bb, 0), 0)
            qty = min(avail_at_bid, sell_cap, self.NARROW_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                sell_cap -= qty
                event_opened_qty = qty

        return orders, buy_cap, sell_cap, event_opened_qty

    # ==========================================================
    # MAKING HELPERS
    # ==========================================================
    def _most_competitive_quotes(
        self, best_bid: Optional[int], best_ask: Optional[int]
    ) -> Optional[tuple[int, int]]:
        if best_bid is None or best_ask is None:
            return None
        if best_bid + 1 < best_ask:
            return best_bid + 1, best_ask - 1
        return best_bid, best_ask

    def _r_ratio_filter(
        self, best_bid: int, best_ask: int, fair: int,
        r_low: float, r_high: float
    ) -> tuple[bool, bool]:
        """
        r = (fair − best_bid) / spread

        r ≥ r_high → fair vicino all'ask → prezzo salirà → skip ask (non vendere)
        r ≤ r_low  → fair vicino al bid  → prezzo scenderà → skip bid (non comprare)

        Protegge dai fill tossici senza ridurre la quota nel regime normale.
        """
        spread = best_ask - best_bid
        if spread <= 0:
            return True, True
        r = (fair - best_bid) / spread
        if r >= r_high:
            return True, False
        if r <= r_low:
            return False, True
        return True, True

    # ==========================================================
    # FAIR — EMERALDS
    # ==========================================================
    def _emeralds_fair(self) -> int:
        return self.EMERALDS_FAIR

    # ==========================================================
    # FAIR — TOMATOES
    # ==========================================================
    def _tomatoes_fair(self, od: OrderDepth) -> Optional[int]:
        """
        Fair engine aligned with GIGI_current_best:

            fair = microprice_L1 + lambda * y_hat

        where y_hat is the ridge forecast built from:
        - imbalance_tot (L1+L2)
        - imbalance_2
        - imbalance_1
        - micro_edge = microprice_L1 - mid
        - spread

        This removes the blended anchor and tanh correction used before.
        """
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        mid = self._midprice(od)
        mp = self._microprice_l1(od)
        if bb is None or ba is None or mid is None or mp is None:
            return None

        spread = float(ba - bb)
        micro_edge = mp - mid

        bv1 = od.buy_orders.get(bb, 0)
        av1 = -od.sell_orders.get(ba, 0)
        if bv1 < 0:
            bv1 = 0
        if av1 < 0:
            av1 = 0
        den1 = bv1 + av1
        imbalance_1 = (bv1 - av1) / den1 if den1 > 0 else 0.0

        bv2, av2, _, _ = self._level2_vols(od)
        den2 = bv2 + av2
        imbalance_2 = (bv2 - av2) / den2 if den2 > 0 else 0.0

        bid_tot = bv1 + bv2
        ask_tot = av1 + av2
        den_tot = bid_tot + ask_tot
        imbalance_tot = (bid_tot - ask_tot) / den_tot if den_tot > 0 else 0.0

        y_hat = (
            self._TOM_INTERCEPT
            + self._TOM_COEF_IMBT * imbalance_tot
            + self._TOM_COEF_IMB2 * imbalance_2
            + self._TOM_COEF_IMB1 * imbalance_1
            + self._TOM_COEF_ME * micro_edge
            + self._TOM_COEF_SP * spread
        )

        fair = mp + self.TOMATOES_RIDGE_LAMBDA * y_hat
        return round(fair)

    # ==========================================================
    # GENERIC PRODUCT PIPELINE
    # ==========================================================
    def _trade_product(
        self,
        state: TradingState,
        result: Dict[str, List[Order]],
        product: str,
        fair: int,
        take_threshold: int,
        max_passive_size: int,
        mm_r_low: float,
        mm_r_high: float,
    ) -> None:
        od = state.order_depths.get(product)
        if od is None:
            return

        orders: List[Order] = []
        pos      = self._position(state, product)
        buy_cap  = self._buy_capacity(product, pos)
        sell_cap = self._sell_capacity(product, pos)
        bb = self._best_bid(od)
        ba = self._best_ask(od)

        if bb is None or ba is None:
            result[product] = orders
            return

        # ── 1. Standard taking vs fair ─────────────────────────
        # Cattura arb puro: ask≤fair o bid≥fair.
        # Per EMERALDS (fair=10000): i 29 fill storici avvengono
        # esattamente sui 59 step con spread=8 in cui ask=10000
        # oppure bid=10000.
        if self.ENABLE_TAKING:
            take_buy, buy_cap = self._take_asks_below(
                product=product, od=od,
                buy_cap=buy_cap, max_ask=fair - take_threshold,
            )
            orders.extend(take_buy)

            take_sell, sell_cap = self._take_bids_above(
                product=product, od=od,
                sell_cap=sell_cap, min_bid=fair + take_threshold,
            )
            orders.extend(take_sell)

        # ── 2. Passive market-making con r-ratio filter ─────────
        # Quota al bid+1 / ask−1 (massima competitività).
        # Il r-ratio sopprime il lato avverso quando il fair è molto
        # distante dal centro dello spread.
        if self.ENABLE_MAKING:
            quotes = self._most_competitive_quotes(bb, ba)
            if quotes is not None:
                bid_q, ask_q = quotes

                place_bid, place_ask = self._r_ratio_filter(
                    best_bid=bb, best_ask=ba,
                    fair=fair, r_low=mm_r_low, r_high=mm_r_high,
                )

                # Ricalcola capacità residua dopo taking
                taken_buy  = sum(o.quantity for o in orders if o.quantity > 0)
                taken_sell = sum(o.quantity for o in orders if o.quantity < 0)
                buy_cap  = self._buy_capacity(product, pos)  - taken_buy
                sell_cap = self._sell_capacity(product, pos) + taken_sell

                buy_size  = min(buy_cap,  max_passive_size)
                sell_size = min(sell_cap, max_passive_size)

                if place_bid and buy_size > 0:
                    orders.append(Order(product, bid_q,  buy_size))
                if place_ask and sell_size > 0:
                    orders.append(Order(product, ask_q, -sell_size))

        result[product] = orders

    # ==========================================================
    # EMERALDS
    # ==========================================================
    def _trade_emeralds(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        product = self.EMERALDS
        if product not in state.order_depths:
            return
        self._trade_product(
            state=state, result=result, product=product,
            fair=self._emeralds_fair(),
            take_threshold=self.EMERALDS_TAKE_THRESHOLD,
            max_passive_size=self.EMERALDS_MAX_PASSIVE_SIZE,
            mm_r_low=self.EMERALDS_MM_R_LOW,
            mm_r_high=self.EMERALDS_MM_R_HIGH,
        )

    # ==========================================================
    # TOMATOES
    # ==========================================================
    def _trade_tomatoes(self, state: TradingState, result: Dict[str, List[Order]], memory: dict) -> None:
        product = self.TOMATOES
        if product not in state.order_depths:
            return

        od = state.order_depths[product]
        fair = self._tomatoes_fair(od)
        if fair is None:
            return

        key = "tomatoes_event"
        event_mem = memory.get(key, {})
        prev_event_state = int(event_mem.get("state", self.TOMATOES_EVENT_NEUTRAL))
        event_qty = int(event_mem.get("qty", 0))

        pos = self._position(state, product)
        buy_cap = self._buy_capacity(product, pos)
        sell_cap = self._sell_capacity(product, pos)

        current_event_state = self._detect_narrow_event(od)
        orders: List[Order] = []

        # 1) standard taking vs fair
        if self.ENABLE_TAKING:
            take_buy, buy_cap = self._take_asks_below(
                product=product, od=od,
                buy_cap=buy_cap, max_ask=fair - self.TOMATOES_TAKE_THRESHOLD,
            )
            orders.extend(take_buy)

            take_sell, sell_cap = self._take_bids_above(
                product=product, od=od,
                sell_cap=sell_cap, min_bid=fair + self.TOMATOES_TAKE_THRESHOLD,
            )
            orders.extend(take_sell)

        # 2) if the previous event has ended or flipped, close the inventory opened by that event
        if prev_event_state != self.TOMATOES_EVENT_NEUTRAL and current_event_state != prev_event_state and event_qty > 0:
            exit_orders, buy_cap, sell_cap, event_qty = self._event_exit_orders(
                product=product,
                od=od,
                event_side=prev_event_state,
                event_qty=event_qty,
                buy_cap=buy_cap,
                sell_cap=sell_cap,
            )
            orders.extend(exit_orders)

        # 3) if the current event is active, trade in its direction and track only the event inventory
        if self.ENABLE_TAKING and current_event_state != self.TOMATOES_EVENT_NEUTRAL:
            narrow_orders, buy_cap, sell_cap, opened_qty = self._narrow_spread_taking(
                product=product,
                od=od,
                buy_cap=buy_cap,
                sell_cap=sell_cap,
                event_side=current_event_state,
            )
            orders.extend(narrow_orders)
            event_qty += opened_qty

        # 4) passive market making
        if self.ENABLE_MAKING:
            bb = self._best_bid(od)
            ba = self._best_ask(od)
            if bb is not None and ba is not None:
                quotes = self._most_competitive_quotes(bb, ba)
                if quotes is not None:
                    bid_q, ask_q = quotes
                    place_bid, place_ask = self._r_ratio_filter(
                        best_bid=bb, best_ask=ba,
                        fair=fair, r_low=self.TOMATOES_MM_R_LOW, r_high=self.TOMATOES_MM_R_HIGH,
                    )

                    taken_buy = sum(o.quantity for o in orders if o.quantity > 0)
                    taken_sell = -sum(o.quantity for o in orders if o.quantity < 0)
                    buy_cap_mm = self._buy_capacity(product, pos) - taken_buy
                    sell_cap_mm = self._sell_capacity(product, pos) - taken_sell

                    buy_size = min(buy_cap_mm, self.TOMATOES_MAX_PASSIVE_SIZE)
                    sell_size = min(sell_cap_mm, self.TOMATOES_MAX_PASSIVE_SIZE)

                    if place_bid and buy_size > 0:
                        orders.append(Order(product, bid_q, buy_size))
                    if place_ask and sell_size > 0:
                        orders.append(Order(product, ask_q, -sell_size))

        result[product] = orders

        next_state = current_event_state
        if next_state == self.TOMATOES_EVENT_NEUTRAL and event_qty > 0:
            # stay attached to the previous side until the event inventory is fully closed
            next_state = prev_event_state if prev_event_state != self.TOMATOES_EVENT_NEUTRAL else self.TOMATOES_EVENT_NEUTRAL

        if event_qty <= 0:
            event_qty = 0
            if current_event_state == self.TOMATOES_EVENT_NEUTRAL:
                next_state = self.TOMATOES_EVENT_NEUTRAL

        memory[key] = {
            "state": int(next_state),
            "qty": int(event_qty),
        }

        self._log_narrow_regime_snapshot(state, product, order_depth=od, tracked_event_state=next_state, tracked_event_qty=event_qty)

    def _log_narrow_regime_snapshot(
            self,
            state: TradingState,
            product: str,
            order_depth: OrderDepth,
            tracked_event_state: int = 0,
            tracked_event_qty: int = 0,
    ) -> None:
        if state.timestamp % 1000 != 0:
            return

        best_bid = self._best_bid(order_depth)
        best_ask = self._best_ask(order_depth)

        if best_bid is None or best_ask is None:
            print(
                f"NARROW_REGIME_SNAPSHOT ts={state.timestamp} product={product} book=EMPTY"
            )
            return

        bid_vol_1 = order_depth.buy_orders.get(best_bid, 0)
        ask_vol_1 = -order_depth.sell_orders.get(best_ask, 0)

        if bid_vol_1 < 0:
            bid_vol_1 = 0
        if ask_vol_1 < 0:
            ask_vol_1 = 0

        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        den1 = bid_vol_1 + ask_vol_1
        if den1 > 0:
            micro = (best_bid * ask_vol_1 + best_ask * bid_vol_1) / den1
            imbalance_1 = (bid_vol_1 - ask_vol_1) / den1
        else:
            micro = mid
            imbalance_1 = 0.0

        bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
        ask_prices = sorted(order_depth.sell_orders.keys())

        bid_vol_2 = 0
        ask_vol_2 = 0
        if len(bid_prices) >= 2 and len(ask_prices) >= 2:
            bid_vol_2 = order_depth.buy_orders.get(bid_prices[1], 0)
            ask_vol_2 = -order_depth.sell_orders.get(ask_prices[1], 0)
            if bid_vol_2 < 0:
                bid_vol_2 = 0
            if ask_vol_2 < 0:
                ask_vol_2 = 0


        bid_depth_tot = bid_vol_1 + bid_vol_2
        ask_depth_tot = ask_vol_1 + ask_vol_2

        den_tot = bid_depth_tot + ask_depth_tot
        imbalance_tot = (bid_depth_tot - ask_depth_tot) / den_tot if den_tot > 0 else 0.0

        bullish_trigger = int(spread <= 6 and imbalance_tot > 0)
        bearish_trigger = int(spread <= 7 and imbalance_tot < 0)

        if bullish_trigger:
            regime = "bullish_narrow"
        elif bearish_trigger:
            regime = "bearish_narrow"
        else:
            regime = "neutral"

        pos = state.position.get(product, 0)
        tracked_label = "neutral"
        if tracked_event_state == self.TOMATOES_EVENT_LONG:
            tracked_label = "long_event"
        elif tracked_event_state == self.TOMATOES_EVENT_SHORT:
            tracked_label = "short_event"

        print(
            f"NARROW_REGIME_SNAPSHOT "
            f"ts={state.timestamp} "
            f"product={product} "
            f"regime={regime} "
            f"best_bid={best_bid} "
            f"best_ask={best_ask} "
            f"mid={mid:.4f} "
            f"micro={micro:.4f} "
            f"spread={spread} "
            f"imb1={imbalance_1:.6f} "
            f"imbtot={imbalance_tot:.6f} "
            f"bullish_trigger={bullish_trigger} "
            f"bearish_trigger={bearish_trigger} "
            f"pos={pos}"
        )