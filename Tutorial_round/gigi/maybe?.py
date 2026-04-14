from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json


class Trader:
    # ==========================================================
    # PRODUCTS
    # ==========================================================
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

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

    # fair semplice: microprice L1 puro
    TOMATOES_RIDGE_LAMBDA = 0.0

    # ==========================================================
    # NARROW-SPREAD REGIME (HMM-derived)
    # ==========================================================
    NARROW_BUY_SPREAD_MAX = 6
    NARROW_SELL_SPREAD_MAX = 7

    # entry piccola
    NARROW_TAKE_SIZE = 10

    # quando il regime torna neutral, monetizza in modo aggressivo
    NARROW_UNWIND_SIZE = 20

    # se la posizione narrow cresce troppo, scarica comunque
    NARROW_POS_BIAS_THRESHOLD = 5
    NARROW_SAFETY_POS = 40
    NARROW_SAFETY_UNWIND_SIZE = 10

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        trader_state = self._load_state(state.traderData)

        self._trade_emeralds(state, result)
        self._trade_tomatoes(state, result, trader_state)

        return result, 0, self._save_state(trader_state)

    # ==========================================================
    # STATE
    # ==========================================================
    def _load_state(self, trader_data: str) -> dict:
        if not trader_data:
            return {"narrow_pos": 0}
        try:
            data = json.loads(trader_data)
            if "narrow_pos" not in data:
                data["narrow_pos"] = 0
            return data
        except Exception:
            return {"narrow_pos": 0}

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

        bv = od.buy_orders.get(bb, 0)
        av = -od.sell_orders.get(ba, 0)

        if bv < 0:
            bv = 0
        if av < 0:
            av = 0

        den = bv + av
        if den <= 0:
            return self._midprice(od)

        return (bb * av + ba * bv) / den

    def _level2_vols(self, od: OrderDepth) -> tuple[int, int, float, float]:
        bid_prices = sorted(od.buy_orders.keys(), reverse=True)
        ask_prices = sorted(od.sell_orders.keys())

        if len(bid_prices) >= 2 and len(ask_prices) >= 2:
            bp2 = bid_prices[1]
            ap2 = ask_prices[1]
            bv2 = od.buy_orders.get(bp2, 0)
            av2 = -od.sell_orders.get(ap2, 0)
            if bv2 < 0:
                bv2 = 0
            if av2 < 0:
                av2 = 0
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

    def _imbtot_l1_l2(self, od: OrderDepth) -> float:
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return 0.0

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

        return (bid_tot - ask_tot) / den_tot if den_tot > 0 else 0.0

    def _narrow_regime(self, od: OrderDepth) -> str:
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return "neutral"

        spread = ba - bb
        imbt = self._imbtot_l1_l2(od)

        if spread <= self.NARROW_BUY_SPREAD_MAX and imbt > 0:
            return "bullish_narrow"
        if spread <= self.NARROW_SELL_SPREAD_MAX and imbt < 0:
            return "bearish_narrow"
        return "neutral"

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
        buy_cap: int,
        sell_cap: int,
        trader_state: dict,
    ) -> tuple[List[Order], int, int]:
        """
        Entry HMM piccola:
        - bullish_narrow -> buy al best ask
        - bearish_narrow -> sell al best bid

        Non aprire nella direzione opposta se narrow_pos è già dall'altro lato.
        """
        orders: List[Order] = []

        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return orders, buy_cap, sell_cap

        spread = ba - bb
        if spread <= 0:
            return orders, buy_cap, sell_cap

        imbt = self._imbtot_l1_l2(od)

        if (
            spread <= self.NARROW_BUY_SPREAD_MAX
            and imbt > 0
            and buy_cap > 0
            and trader_state["narrow_pos"] >= 0
        ):
            avail = -od.sell_orders.get(ba, 0)
            qty = min(avail, buy_cap, self.NARROW_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, ba, qty))
                buy_cap -= qty
                trader_state["narrow_pos"] += qty

        elif (
            spread <= self.NARROW_SELL_SPREAD_MAX
            and imbt < 0
            and sell_cap > 0
            and trader_state["narrow_pos"] <= 0
        ):
            avail = od.buy_orders.get(bb, 0)
            qty = min(avail, sell_cap, self.NARROW_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                sell_cap -= qty
                trader_state["narrow_pos"] -= qty

        return orders, buy_cap, sell_cap

    def _neutral_unwind(
        self,
        product: str,
        od: OrderDepth,
        trader_state: dict,
        buy_cap: int,
        sell_cap: int,
    ) -> tuple[List[Order], int, int]:
        """
        Quando il regime torna neutral, monetizza aggressivamente parte
        della posizione narrow al best price disponibile.
        """
        orders: List[Order] = []

        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return orders, buy_cap, sell_cap

        narrow_pos = trader_state["narrow_pos"]

        if narrow_pos > 0 and sell_cap > 0:
            avail = od.buy_orders.get(bb, 0)
            qty = min(avail, sell_cap, narrow_pos, self.NARROW_UNWIND_SIZE)
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                sell_cap -= qty
                trader_state["narrow_pos"] -= qty

        elif narrow_pos < 0 and buy_cap > 0:
            avail = -od.sell_orders.get(ba, 0)
            qty = min(avail, buy_cap, -narrow_pos, self.NARROW_UNWIND_SIZE)
            if qty > 0:
                orders.append(Order(product, ba, qty))
                buy_cap -= qty
                trader_state["narrow_pos"] += qty

        return orders, buy_cap, sell_cap

    def _inventory_safety_unwind(
        self,
        product: str,
        od: OrderDepth,
        pos: int,
        buy_cap: int,
        sell_cap: int,
    ) -> tuple[List[Order], int, int]:
        """
        Safety unwind sulla posizione totale, non solo narrow_pos.
        """
        orders: List[Order] = []

        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return orders, buy_cap, sell_cap

        if pos >= self.NARROW_SAFETY_POS and sell_cap > 0:
            avail = od.buy_orders.get(bb, 0)
            qty = min(avail, sell_cap, pos, self.NARROW_SAFETY_UNWIND_SIZE)
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                sell_cap -= qty

        elif pos <= -self.NARROW_SAFETY_POS and buy_cap > 0:
            avail = -od.sell_orders.get(ba, 0)
            qty = min(avail, buy_cap, -pos, self.NARROW_SAFETY_UNWIND_SIZE)
            if qty > 0:
                orders.append(Order(product, ba, qty))
                buy_cap -= qty

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
        spread = best_ask - best_bid
        if spread <= 0:
            return True, True
        r = (fair - best_bid) / spread
        if r >= r_high:
            return True, False
        if r <= r_low:
            return False, True
        return True, True

    def _position_biased_making(
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
    ) -> None:
        """
        Struttura vecchia, ma con logica corretta:
        - se narrow_pos > threshold -> quota solo ask per monetizzare il long
        - se narrow_pos < -threshold -> quota solo bid per monetizzare lo short
        - altrimenti MM normale con r-ratio
        - capacità residua calcolata correttamente per lato
        """
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return

        quotes = self._most_competitive_quotes(bb, ba)
        if quotes is None:
            return
        bid_q, ask_q = quotes

        already_buy = sum(o.quantity for o in orders if o.quantity > 0)
        already_sell = -sum(o.quantity for o in orders if o.quantity < 0)

        buy_cap = max(0, self.POSITION_LIMITS[product] - pos - already_buy)
        sell_cap = max(0, self.POSITION_LIMITS[product] + pos - already_sell)

        if narrow_pos > self.NARROW_POS_BIAS_THRESHOLD:
            place_bid = False
            place_ask = True
            bid_mult = 0.0
            ask_mult = 1.0
        elif narrow_pos < -self.NARROW_POS_BIAS_THRESHOLD:
            place_bid = True
            place_ask = False
            bid_mult = 1.0
            ask_mult = 0.0
        else:
            place_bid, place_ask = self._r_ratio_filter(
                best_bid=bb,
                best_ask=ba,
                fair=fair,
                r_low=mm_r_low,
                r_high=mm_r_high,
            )
            bid_mult = 1.0
            ask_mult = 1.0

        buy_size = min(buy_cap, max(0, round(max_passive_size * bid_mult)))
        sell_size = min(sell_cap, max(0, round(max_passive_size * ask_mult)))

        if place_bid and buy_size > 0:
            orders.append(Order(product, bid_q, buy_size))
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
        Fair semplice: microprice L1 puro.
        Niente blended anchor, niente tanh, niente correzioni strane.
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
        pos = self._position(state, product)
        bb = self._best_bid(od)
        ba = self._best_ask(od)

        if bb is None or ba is None:
            result[product] = orders
            return

        buy_cap = self._buy_capacity(product, pos)
        sell_cap = self._sell_capacity(product, pos)

        if self.ENABLE_TAKING:
            take_buy, buy_cap = self._take_asks_below(
                product=product,
                od=od,
                buy_cap=buy_cap,
                max_ask=fair - self.EMERALDS_TAKE_THRESHOLD,
            )
            orders.extend(take_buy)

            take_sell, sell_cap = self._take_bids_above(
                product=product,
                od=od,
                sell_cap=sell_cap,
                min_bid=fair + self.EMERALDS_TAKE_THRESHOLD,
            )
            orders.extend(take_sell)

        if self.ENABLE_MAKING:
            quotes = self._most_competitive_quotes(bb, ba)
            if quotes is not None:
                bid_q, ask_q = quotes
                place_bid, place_ask = self._r_ratio_filter(
                    bb, ba, fair,
                    self.EMERALDS_MM_R_LOW, self.EMERALDS_MM_R_HIGH
                )

                taken_buy = sum(o.quantity for o in orders if o.quantity > 0)
                taken_sell = -sum(o.quantity for o in orders if o.quantity < 0)

                bc = max(0, self._buy_capacity(product, pos) - taken_buy)
                sc = max(0, self._sell_capacity(product, pos) - taken_sell)

                if place_bid and min(bc, self.EMERALDS_MAX_PASSIVE_SIZE) > 0:
                    orders.append(Order(product, bid_q, min(bc, self.EMERALDS_MAX_PASSIVE_SIZE)))
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

        pos = self._position(state, product)
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            result[product] = orders
            return

        buy_cap = self._buy_capacity(product, pos)
        sell_cap = self._sell_capacity(product, pos)

        regime = self._narrow_regime(od)

        # 1) standard taking vs fair
        if self.ENABLE_TAKING:
            take_buy, buy_cap = self._take_asks_below(
                product=product,
                od=od,
                buy_cap=buy_cap,
                max_ask=fair - self.TOMATOES_TAKE_THRESHOLD,
            )
            orders.extend(take_buy)

            take_sell, sell_cap = self._take_bids_above(
                product=product,
                od=od,
                sell_cap=sell_cap,
                min_bid=fair + self.TOMATOES_TAKE_THRESHOLD,
            )
            orders.extend(take_sell)

        # 2) se il regime è attivo, entry narrow piccola
        if regime != "neutral":
            already_buy = sum(o.quantity for o in orders if o.quantity > 0)
            already_sell = -sum(o.quantity for o in orders if o.quantity < 0)
            buy_cap = max(0, self._buy_capacity(product, pos) - already_buy)
            sell_cap = max(0, self._sell_capacity(product, pos) - already_sell)

            narrow_orders, buy_cap, sell_cap = self._narrow_spread_taking(
                product=product,
                od=od,
                buy_cap=buy_cap,
                sell_cap=sell_cap,
                trader_state=trader_state,
            )
            orders.extend(narrow_orders)

        # 3) se il regime torna neutral, monetizza subito parte del narrow_pos
        if regime == "neutral":
            already_buy = sum(o.quantity for o in orders if o.quantity > 0)
            already_sell = -sum(o.quantity for o in orders if o.quantity < 0)
            buy_cap = max(0, self._buy_capacity(product, pos) - already_buy)
            sell_cap = max(0, self._sell_capacity(product, pos) - already_sell)

            unwind_orders, buy_cap, sell_cap = self._neutral_unwind(
                product=product,
                od=od,
                trader_state=trader_state,
                buy_cap=buy_cap,
                sell_cap=sell_cap,
            )
            orders.extend(unwind_orders)

        # 4) safety unwind sulla posizione totale
        pos_after_taking = pos + sum(o.quantity for o in orders)
        already_buy = sum(o.quantity for o in orders if o.quantity > 0)
        already_sell = -sum(o.quantity for o in orders if o.quantity < 0)
        buy_cap = max(0, self._buy_capacity(product, pos) - already_buy)
        sell_cap = max(0, self._sell_capacity(product, pos) - already_sell)

        safety_orders, buy_cap, sell_cap = self._inventory_safety_unwind(
            product=product,
            od=od,
            pos=pos_after_taking,
            buy_cap=buy_cap,
            sell_cap=sell_cap,
        )
        orders.extend(safety_orders)

        # 5) passive MM con bias verso monetizzazione della posizione narrow
        if self.ENABLE_MAKING:
            self._position_biased_making(
                product=product,
                od=od,
                orders=orders,
                pos=pos,
                fair=fair,
                max_passive_size=self.TOMATOES_MAX_PASSIVE_SIZE,
                mm_r_low=self.TOMATOES_MM_R_LOW,
                mm_r_high=self.TOMATOES_MM_R_HIGH,
                narrow_pos=trader_state["narrow_pos"],
            )

        result[product] = orders