from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import math


class Trader:
    # ==========================================================
    # PRODUCTS
    # ==========================================================
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    ENABLE_TAKING = True
    ENABLE_MAKING = False

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
    EMERALDS_FAIR = 10_000
    EMERALDS_TAKE_THRESHOLD = 0
    EMERALDS_MAX_PASSIVE_SIZE = 80
    EMERALDS_MM_R_LOW = 0.30
    EMERALDS_MM_R_HIGH = 0.70

    # ==========================================================
    # PARAMETERS — TOMATOES
    # ==========================================================
    TOMATOES_TAKE_THRESHOLD = 0
    TOMATOES_MAX_PASSIVE_SIZE = 80
    TOMATOES_MM_R_LOW = 0.30
    TOMATOES_MM_R_HIGH = 0.70

    # ----------------------------------------------------------
    # Fair engine aligned with GIGI_current_best:
    #
    #   fair = microprice_L1 + lambda * ridge_prediction
    #
    # No blended anchor, no tanh saturation, no extra nonlinear cap.
    # ----------------------------------------------------------
    TOMATOES_RIDGE_LAMBDA = 0.0

    # Coefficienti ridge (target: mid_{t+1}−microprice_t,
    # calibrati su day=−2, validati su day=−1, no overfitting)
    _TOM_INTERCEPT = 0.276679873589467
    _TOM_COEF_IMBT = -32.853054946602477
    _TOM_COEF_IMB2 = 28.554217165148906
    _TOM_COEF_IMB1 = 21.507001248247061
    _TOM_COEF_ME = -3.607552682276733
    _TOM_COEF_SP = -0.021753165199018

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
    NARROW_BUY_SPREAD_MAX = 6  # spread massimo per aprire long
    NARROW_SELL_SPREAD_MAX = 7  # spread massimo per aprire short
    NARROW_TAKE_SIZE = 20  # unità per singola operazione narrow

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        self._trade_emeralds(state, result)
        self._trade_tomatoes(state, result)
        return result, 0, ""

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
        bv = od.buy_orders.get(bb, 0)
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
            bp2 = bid_prices[1];
            ap2 = ask_prices[1]
            bv2 = od.buy_orders.get(bp2, 0)
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
        bv1 = od.buy_orders.get(bb, 0)
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

    def _narrow_spread_taking(
            self,
            product: str,
            od: OrderDepth,
            pos: int,
            buy_cap: int,
            sell_cap: int,
            persistent: dict,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []

        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return orders, buy_cap, sell_cap

        spread = ba - bb
        if spread <= 0:
            return orders, buy_cap, sell_cap

        # Calcola imbalance_tot (identico a prima)
        bv1 = od.buy_orders.get(bb, 0)
        av1 = -od.sell_orders.get(ba, 0)
        if bv1 < 0: bv1 = 0
        if av1 < 0: av1 = 0
        bv2, av2, _, _ = self._level2_vols(od)
        bid_tot = bv1 + bv2
        ask_tot = av1 + av2
        den_tot = bid_tot + ask_tot
        imbtot = (bid_tot - ask_tot) / den_tot if den_tot > 0 else 0.0

        bullish = spread <= self.NARROW_BUY_SPREAD_MAX and imbtot > 0
        bearish = spread <= self.NARROW_SELL_SPREAD_MAX and imbtot < 0

        # Posizione attribuita al regime narrow (tracciata nello stato)
        narrow_pos = persistent.get(f"narrow_pos_{product}", 0)

        # ── CHIUDI posizione se il regime è cambiato ────────────────
        if narrow_pos > 0 and not bullish:
            # Eravamo long da segnale bullish, ora regime neutro/bearish → chiudi
            qty = min(narrow_pos, sell_cap, od.buy_orders.get(bb, 0))
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                sell_cap -= qty
                persistent[f"narrow_pos_{product}"] = narrow_pos - qty
                return orders, buy_cap, sell_cap

        if narrow_pos < 0 and not bearish:
            # Eravamo short da segnale bearish, ora regime neutro/bullish → chiudi
            qty = min(-narrow_pos, buy_cap, -od.sell_orders.get(ba, 0))
            if qty > 0:
                orders.append(Order(product, ba, qty))
                buy_cap -= qty
                persistent[f"narrow_pos_{product}"] = narrow_pos + qty
                return orders, buy_cap, sell_cap

        # ── APRI nuova posizione solo se non siamo già esposti ─────
        if bullish and buy_cap > 0 and narrow_pos <= 0:
            avail = -od.sell_orders.get(ba, 0)
            qty = min(avail, buy_cap, self.NARROW_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, ba, qty))
                buy_cap -= qty
                persistent[f"narrow_pos_{product}"] = narrow_pos + qty

        elif bearish and sell_cap > 0 and narrow_pos >= 0:
            avail = od.buy_orders.get(bb, 0)
            qty = min(avail, sell_cap, self.NARROW_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                sell_cap -= qty
                persistent[f"narrow_pos_{product}"] = narrow_pos - qty

        return orders, buy_cap, sell_cap
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
            use_narrow_taking: bool = False,
    ) -> None:
        od = state.order_depths.get(product)
        if od is None:
            return

        orders: List[Order] = []
        pos = self._position(state, product)
        buy_cap = self._buy_capacity(product, pos)
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

            # ── 2. Narrow-spread taking (regime HMM) ───────────
            # Attivo solo per TOMATOES (use_narrow_taking=True).
            # Condizione: spread stretto + imbtot direzionale → EV>0.
            # Eseguito DOPO il standard taking per non sprecare capacità.
            if use_narrow_taking:
                narrow_orders, buy_cap, sell_cap = self._narrow_spread_taking(
                    product=product, od=od,
                    buy_cap=buy_cap, sell_cap=sell_cap,
                )
                orders.extend(narrow_orders)

        # ── 3. Passive market-making con r-ratio filter ─────────
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
                taken_buy = sum(o.quantity for o in orders if o.quantity > 0)
                taken_sell = sum(o.quantity for o in orders if o.quantity < 0)
                buy_cap = self._buy_capacity(product, pos) - taken_buy
                sell_cap = self._sell_capacity(product, pos) + taken_sell

                buy_size = min(buy_cap, max_passive_size)
                sell_size = min(sell_cap, max_passive_size)

                if place_bid and buy_size > 0:
                    orders.append(Order(product, bid_q, buy_size))
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
            use_narrow_taking=False,  # EMERALDS ha spread fisso 8/16, no narrow regime
        )

    # ==========================================================
    # TOMATOES
    # ==========================================================
    def _trade_tomatoes(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        product = self.TOMATOES
        if product not in state.order_depths:
            return
        fair = self._tomatoes_fair(state.order_depths[product])
        if fair is None:
            return
        self._trade_product(
            state=state, result=result, product=product,
            fair=fair,
            take_threshold=self.TOMATOES_TAKE_THRESHOLD,
            max_passive_size=self.TOMATOES_MAX_PASSIVE_SIZE,
            mm_r_low=self.TOMATOES_MM_R_LOW,
            mm_r_high=self.TOMATOES_MM_R_HIGH,
            use_narrow_taking=True,  # narrow-spread taking abilitato (derivato da HMM)
        )


