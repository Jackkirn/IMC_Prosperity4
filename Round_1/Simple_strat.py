from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional


class Trader:
    # ==========================================================
    # PRODUCTS
    # ==========================================================
    PEPPER = "INTARIAN_PEPPER_ROOT"
    ASH = "ASH_COATED_OSMIUM"

    # ==========================================================
    # GLOBAL SWITCHES
    # ==========================================================
    ENABLE_TAKING = True
    ENABLE_MAKING = True

    # ==========================================================
    # POSITION LIMITS
    # ==========================================================
    POSITION_LIMITS = {
        PEPPER: 80,
        ASH: 80,
    }

    # ==========================================================
    # PARAMETERS — ASH (Market Making)
    # ==========================================================
    ASH_TAKE_THRESHOLD = 0
    ASH_MAX_PASSIVE_SIZE = 20
    ASH_MM_R_LOW = 0.00
    ASH_MM_R_HIGH = 1.00

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        self._trade_pepper(state, result)
        self._trade_ash(state, result)

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

        if bv < 0:
            bv = 0
        if av < 0:
            av = 0

        den = bv + av
        if den <= 0:
            return self._midprice(od)

        return (bb * av + ba * bv) / den

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
            self,
            product: str,
            od: OrderDepth,
            buy_cap: int,
            max_ask: int,
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
            self,
            product: str,
            od: OrderDepth,
            sell_cap: int,
            min_bid: int,
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

    # ==========================================================
    # MAKING HELPERS
    # ==========================================================
    def _most_competitive_quotes(
            self,
            best_bid: Optional[int],
            best_ask: Optional[int],
    ) -> Optional[tuple[int, int]]:
        if best_bid is None or best_ask is None:
            return None
        if best_bid + 1 < best_ask:
            return best_bid + 1, best_ask - 1
        return best_bid, best_ask

    def _r_ratio_filter(
            self,
            best_bid: int,
            best_ask: int,
            fair: int,
            r_low: float,
            r_high: float,
    ) -> tuple[bool, bool]:
        """
        r = (fair - best_bid) / spread
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
    # FAIR — ASH
    # ==========================================================
    def _ash_fair(self, od: OrderDepth) -> Optional[int]:
        mp = self._microprice_l1(od)
        if mp is None:
            return None
        return round(mp)

    # ==========================================================
    # GENERIC PRODUCT PIPELINE (Per asset come Ash)
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
        pos = self._position(state, product)

        bb = self._best_bid(od)
        ba = self._best_ask(od)

        if bb is None or ba is None:
            result[product] = orders
            return

        buy_cap = self._buy_capacity(product, pos)
        sell_cap = self._sell_capacity(product, pos)

        # ------------------------------------------------------
        # 1) Standard taking vs fair
        # ------------------------------------------------------
        if self.ENABLE_TAKING:
            take_buy_orders, buy_cap = self._take_asks_below(
                product=product,
                od=od,
                buy_cap=buy_cap,
                max_ask=fair - take_threshold,
            )
            orders.extend(take_buy_orders)

            take_sell_orders, sell_cap = self._take_bids_above(
                product=product,
                od=od,
                sell_cap=sell_cap,
                min_bid=fair + take_threshold,
            )
            orders.extend(take_sell_orders)

        # ------------------------------------------------------
        # 2) Passive market making
        # ------------------------------------------------------
        if self.ENABLE_MAKING:
            quotes = self._most_competitive_quotes(bb, ba)
            if quotes is not None:
                bid_q, ask_q = quotes

                place_bid, place_ask = self._r_ratio_filter(
                    best_bid=bb,
                    best_ask=ba,
                    fair=fair,
                    r_low=mm_r_low,
                    r_high=mm_r_high,
                )

                taken_buy = sum(o.quantity for o in orders if o.quantity > 0)
                taken_sell = -sum(o.quantity for o in orders if o.quantity < 0)

                buy_cap_mm = max(0, self._buy_capacity(product, pos) - taken_buy)
                sell_cap_mm = max(0, self._sell_capacity(product, pos) - taken_sell)

                buy_size = min(buy_cap_mm, max_passive_size)
                sell_size = min(sell_cap_mm, max_passive_size)

                if place_bid and buy_size > 0:
                    orders.append(Order(product, bid_q, buy_size))

                if place_ask and sell_size > 0:
                    orders.append(Order(product, ask_q, -sell_size))

        result[product] = orders

    # ==========================================================
    # PEPPER (LONG ONLY STRATEGY)
    # ==========================================================
    def _trade_pepper(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        product = self.PEPPER
        if product not in state.order_depths:
            return

        od = state.order_depths[product]
        pos = self._position(state, product)
        buy_cap = self._buy_capacity(product, pos)

        orders: List[Order] = []

        # Riempiamo la capacità Long il prima possibile
        if buy_cap > 0:

            best_bid = self._best_bid(od)
            best_ask = self._best_ask(od)
            # Colpiamo tutti gli ask visibili
            take_orders, buy_cap = self._take_asks_below(
                product=product,
                od=od,
                buy_cap=buy_cap,
                max_ask=best_ask,
            )
            orders.extend(take_orders)

            # Se dopo aver colpito a mercato abbiamo ancora margine, piazziamo ordini bid aggressivi

            if buy_cap > 0 and best_bid is not None:
                # Quotiamo al best_bid + 1 per farci fillare
                orders.append(Order(product, best_bid + 1, buy_cap))

        result[product] = orders

    # ==========================================================
    # ASH (MARKET MAKING STRATEGY)
    # ==========================================================
    def _trade_ash(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        product = self.ASH
        if product not in state.order_depths:
            return

        fair = self._ash_fair(state.order_depths[product])
        if fair is None:
            return

        self._trade_product(
            state=state,
            result=result,
            product=product,
            fair=fair,
            take_threshold=self.ASH_TAKE_THRESHOLD,
            max_passive_size=self.ASH_MAX_PASSIVE_SIZE,
            mm_r_low=self.ASH_MM_R_LOW,
            mm_r_high=self.ASH_MM_R_HIGH,
        )