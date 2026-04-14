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

    # Fair = solo microprice L1.
    # (ridge calibrato ma non usato — manteniamo i coefficienti
    # per eventuale riattivazione futura)
    _TOM_INTERCEPT  =  0.276679873589467
    _TOM_COEF_IMBT  = -32.853054946602477
    _TOM_COEF_IMB2  =  28.554217165148906
    _TOM_COEF_IMB1  =  21.507001248247061
    _TOM_COEF_ME    =  -3.607552682276733
    _TOM_COEF_SP    =  -0.021753165199018

    # ==========================================================
    # NARROW-SPREAD MAKING (da HMM)
    #
    # Invece di fare taking (market order), quotiamo UN SOLO lato
    # in modo molto competitivo (bid+1 / ask-1) così veniamo
    # fillati passivamente senza pagare lo spread.
    #
    # Regime identificato dall'HMM:
    #   spread<=6 + imbtot>0 → entry BID  (EV +1.38t, acc 98.3%)
    #   spread<=7 + imbtot<0 → entry ASK  (EV +0.42t, acc 96.4%)
    #
    # Chiusura: appena il regime cambia, quotiamo solo il lato
    # opposto finché non siamo flat → usiamo solo il making engine.
    # ==========================================================
    NARROW_BUY_SPREAD_MAX       = 6
    NARROW_SELL_SPREAD_MAX      = 7
    NARROW_TAKE_SIZE            = 10
    NARROW_POS_BIAS_THRESHOLD   = 0   # qualsiasi pos narrow attiva il close

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        trader_state = self._load_state(state.traderData)

        self._trade_emeralds(state, result)
        self._trade_tomatoes(state, result, trader_state)

        new_trader_data = self._save_state(trader_state)
        return result, 0, new_trader_data

    # ==========================================================
    # STATO PERSISTENTE
    # narrow_pos   : posizione accumulata nel regime narrow
    # narrow_intent: "entering_long" / "entering_short" / ""
    # ==========================================================
    def _load_state(self, trader_data: str) -> dict:
        if not trader_data:
            return {"narrow_pos": 0, "narrow_intent": ""}
        try:
            s = json.loads(trader_data)
            if "narrow_intent" not in s:
                s["narrow_intent"] = ""
            return s
        except Exception:
            return {"narrow_pos": 0, "narrow_intent": ""}

    def _save_state(self, state: dict) -> str:
        return json.dumps(state, separators=(",", ":"))

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

    def _position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def _buy_capacity(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] - pos

    def _sell_capacity(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] + pos

    def _imbtot(self, od: OrderDepth) -> float:
        """Imbalance L1+L2. Usato per il regime HMM e per il fair."""
        bb = self._best_bid(od); ba = self._best_ask(od)
        if bb is None or ba is None:
            return 0.0
        bv1 =  od.buy_orders.get(bb, 0)
        av1 = -od.sell_orders.get(ba, 0)
        if bv1 < 0: bv1 = 0
        if av1 < 0: av1 = 0
        bv2, av2, _, _ = self._level2_vols(od)
        bid_tot = bv1 + bv2; ask_tot = av1 + av2
        den_tot = bid_tot + ask_tot
        return (bid_tot - ask_tot) / den_tot if den_tot > 0 else 0.0

    # ==========================================================
    # TAKING HELPERS
    # ==========================================================
    def _take_asks_below(
        self, product: str, od: OrderDepth, buy_cap: int, max_ask: int
    ) -> tuple[List[Order], int]:
        orders: List[Order] = []
        for ask in sorted(od.sell_orders.keys()):
            if buy_cap <= 0: break
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
            if sell_cap <= 0: break
            if bid >= min_bid:
                qty = min(od.buy_orders[bid], sell_cap)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
            else:
                break
        return orders, sell_cap

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
        spread = best_ask - best_bid
        if spread <= 0:
            return True, True
        r = (fair - best_bid) / spread
        if r >= r_high:
            return True, False
        if r <= r_low:
            return False, True
        return True, True

    def _narrow_regime_making(
        self,
        product: str,
        od: OrderDepth,
        orders: List[Order],
        pos: int,
        fair: int,
        max_passive_size: int,
        mm_r_low: float,
        mm_r_high: float,
        narrow_pos: int,
        narrow_intent: str,
        trader_state: dict,
        taken_buy: int,
        taken_sell: int,
    ) -> None:
        """
        Making con logica HMM narrow-spread.

        Flusso:
        1. Se abbiamo una posizione narrow da chiudere (narrow_pos != 0):
           quota SOLO il lato che riduce la posizione.
        2. Se siamo in attesa di riempire un'entry (narrow_intent != ""):
           continua a quotare lo stesso lato dell'entry.
        3. Se rilevato regime bullish/bearish senza posizione aperta:
           quota SOLO il lato di entrata, imposta narrow_intent.
        4. Altrimenti: r-ratio normale su entrambi i lati.

        Vantaggi vs taking:
          - Compriamo a bid+1 invece di ask → risparmio di 1 spread
          - Usiamo solo l'engine passivo, nessun market order
        """
        bb = self._best_bid(od); ba = self._best_ask(od)
        if bb is None or ba is None:
            return

        quotes = self._most_competitive_quotes(bb, ba)
        if quotes is None:
            return
        bid_q, ask_q = quotes

        limit = self.POSITION_LIMITS[product]
        buy_cap  = (limit - pos) - taken_buy
        sell_cap = (limit + pos) + taken_sell

        spread = ba - bb
        imbt   = self._imbtot(od)
        bullish_narrow = (spread <= self.NARROW_BUY_SPREAD_MAX  and imbt > 0)
        bearish_narrow = (spread <= self.NARROW_SELL_SPREAD_MAX and imbt < 0)

        # ── Priorità 1: chiusura posizione narrow esistente ──────
        if narrow_pos > self.NARROW_POS_BIAS_THRESHOLD:
            # Long da chiudere → quota solo ask
            sell_size = min(sell_cap, max_passive_size)
            if sell_size > 0:
                orders.append(Order(product, ask_q, -sell_size))
            return

        if narrow_pos < -self.NARROW_POS_BIAS_THRESHOLD:
            # Short da chiudere → quota solo bid
            buy_size = min(buy_cap, max_passive_size)
            if buy_size > 0:
                orders.append(Order(product, bid_q, buy_size))
            return

        # ── Priorità 2: entry narrow in corso (non ancora fillato) ─
        if narrow_intent == "entering_long":
            buy_size = min(buy_cap, self.NARROW_TAKE_SIZE)
            if buy_size > 0:
                orders.append(Order(product, bid_q, buy_size))
            return

        if narrow_intent == "entering_short":
            sell_size = min(sell_cap, self.NARROW_TAKE_SIZE)
            if sell_size > 0:
                orders.append(Order(product, ask_q, -sell_size))
            return

        # ── Priorità 3: nuovo entry narrow (regime rilevato) ──────
        if bullish_narrow and buy_cap > 0:
            # Regime bullish: quota solo bid → entrata long passiva
            buy_size = min(buy_cap, self.NARROW_TAKE_SIZE)
            if buy_size > 0:
                orders.append(Order(product, bid_q, buy_size))
                trader_state["narrow_intent"] = "entering_long"
            return

        if bearish_narrow and sell_cap > 0:
            # Regime bearish: quota solo ask → entrata short passiva
            sell_size = min(sell_cap, self.NARROW_TAKE_SIZE)
            if sell_size > 0:
                orders.append(Order(product, ask_q, -sell_size))
                trader_state["narrow_intent"] = "entering_short"
            return

        # ── Priorità 4: making normale con r-ratio ────────────────
        place_bid, place_ask = self._r_ratio_filter(
            best_bid=bb, best_ask=ba,
            fair=fair, r_low=mm_r_low, r_high=mm_r_high,
        )
        buy_size  = min(buy_cap,  max_passive_size)
        sell_size = min(sell_cap, max_passive_size)
        if place_bid and buy_size > 0:
            orders.append(Order(product, bid_q,  buy_size))
        if place_ask and sell_size > 0:
            orders.append(Order(product, ask_q, -sell_size))

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
        Fair = solo microprice L1.
        Il ridge è calibrato e mantenuto nei coefficienti sopra
        per eventuale riattivazione, ma non viene usato.
        """
        mp = self._microprice_l1(od)
        if mp is None:
            return None
        return round(mp)

    # ==========================================================
    # PRODUCT PIPELINES
    # ==========================================================
    def _trade_emeralds(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        product = self.EMERALDS
        od = state.order_depths.get(product)
        if od is None:
            return

        orders: List[Order] = []
        fair = self._emeralds_fair()
        pos  = self._position(state, product)
        bb = self._best_bid(od); ba = self._best_ask(od)
        if bb is None or ba is None:
            result[product] = orders
            return

        buy_cap  = self._buy_capacity(product, pos)
        sell_cap = self._sell_capacity(product, pos)

        if self.ENABLE_TAKING:
            take_buy, buy_cap = self._take_asks_below(
                product=product, od=od, buy_cap=buy_cap,
                max_ask=fair - self.EMERALDS_TAKE_THRESHOLD)
            orders.extend(take_buy)
            take_sell, sell_cap = self._take_bids_above(
                product=product, od=od, sell_cap=sell_cap,
                min_bid=fair + self.EMERALDS_TAKE_THRESHOLD)
            orders.extend(take_sell)

        if self.ENABLE_MAKING:
            quotes = self._most_competitive_quotes(bb, ba)
            if quotes is not None:
                bid_q, ask_q = quotes
                place_bid, place_ask = self._r_ratio_filter(
                    bb, ba, fair,
                    self.EMERALDS_MM_R_LOW, self.EMERALDS_MM_R_HIGH)
                taken_buy  = sum(o.quantity for o in orders if o.quantity > 0)
                taken_sell = sum(o.quantity for o in orders if o.quantity < 0)
                bc = self._buy_capacity(product, pos)  - taken_buy
                sc = self._sell_capacity(product, pos) + taken_sell
                if place_bid and min(bc, self.EMERALDS_MAX_PASSIVE_SIZE) > 0:
                    orders.append(Order(product, bid_q,  min(bc, self.EMERALDS_MAX_PASSIVE_SIZE)))
                if place_ask and min(sc, self.EMERALDS_MAX_PASSIVE_SIZE) > 0:
                    orders.append(Order(product, ask_q, -min(sc, self.EMERALDS_MAX_PASSIVE_SIZE)))

        result[product] = orders

    def _trade_tomatoes(
        self,
        state: TradingState,
        result: Dict[str, List[Order]],
        trader_state: dict,
    ) -> None:
        product = self.TOMATOES
        od = state.order_depths.get(product)
        if od is None:
            return

        orders: List[Order] = []
        fair = self._tomatoes_fair(od)
        if fair is None:
            return

        pos  = self._position(state, product)
        bb = self._best_bid(od); ba = self._best_ask(od)
        if bb is None or ba is None:
            result[product] = orders
            return

        # ── Sync narrow_pos con la posizione reale (inizio tick) ──
        # Prima di decidere gli ordini, aggiorniamo lo stato basandoci
        # su cosa è successo al tick precedente (i fill sono già in pos).
        narrow_pos    = trader_state["narrow_pos"]
        narrow_intent = trader_state.get("narrow_intent", "")

        if narrow_intent == "entering_long":
            if pos > 0:
                # Entry bid fillato (almeno in parte)
                trader_state["narrow_pos"] = pos
                trader_state["narrow_intent"] = ""
            elif pos < 0:
                # Posizione inversa inattesa → reset
                trader_state["narrow_intent"] = ""
        elif narrow_intent == "entering_short":
            if pos < 0:
                # Entry ask fillato
                trader_state["narrow_pos"] = pos
                trader_state["narrow_intent"] = ""
            elif pos > 0:
                trader_state["narrow_intent"] = ""
        elif narrow_pos > 0:
            # Stiamo chiudendo un long: narrow_pos scende con la posizione
            trader_state["narrow_pos"] = min(narrow_pos, max(pos, 0))
        elif narrow_pos < 0:
            # Stiamo chiudendo uno short
            trader_state["narrow_pos"] = max(narrow_pos, min(pos, 0))

        # Rileggiamo i valori aggiornati
        narrow_pos    = trader_state["narrow_pos"]
        narrow_intent = trader_state.get("narrow_intent", "")

        buy_cap  = self._buy_capacity(product, pos)
        sell_cap = self._sell_capacity(product, pos)

        # ── 1. Standard taking vs fair ──────────────────────────
        if self.ENABLE_TAKING:
            take_buy, buy_cap = self._take_asks_below(
                product=product, od=od, buy_cap=buy_cap,
                max_ask=fair - self.TOMATOES_TAKE_THRESHOLD)
            orders.extend(take_buy)

            take_sell, sell_cap = self._take_bids_above(
                product=product, od=od, sell_cap=sell_cap,
                min_bid=fair + self.TOMATOES_TAKE_THRESHOLD)
            orders.extend(take_sell)

        # ── 2. Making con logica HMM (niente taking narrow) ─────
        # Quotiamo UN SOLO lato in base al regime:
        #   - regime bullish → bid aggressivo (entry long)
        #   - regime bearish → ask aggressivo (entry short)
        #   - posizione narrow aperta → lato chiusura
        #   - neutro → making normale r-ratio
        if self.ENABLE_MAKING:
            taken_buy  = sum(o.quantity for o in orders if o.quantity > 0)
            taken_sell = sum(o.quantity for o in orders if o.quantity < 0)

            self._narrow_regime_making(
                product=product, od=od,
                orders=orders, pos=pos,
                fair=fair,
                max_passive_size=self.TOMATOES_MAX_PASSIVE_SIZE,
                mm_r_low=self.TOMATOES_MM_R_LOW,
                mm_r_high=self.TOMATOES_MM_R_HIGH,
                narrow_pos=narrow_pos,
                narrow_intent=narrow_intent,
                trader_state=trader_state,
                taken_buy=taken_buy,
                taken_sell=taken_sell,
            )

        result[product] = orders