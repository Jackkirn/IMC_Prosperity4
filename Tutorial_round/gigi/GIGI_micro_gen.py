from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional


class Trader:
    # ==========================================================
    # PRODUCTS
    # ==========================================================
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    ENABLE_TAKING = False
    ENABLE_MAKING = True

    # ==========================================================
    # POSITION LIMITS
    # ==========================================================
    POSITION_LIMITS = {
        EMERALDS: 80,
        TOMATOES: 80,
    }

    # ==========================================================
    # PARAMETERS - EMERALDS
    # ==========================================================
    EMERALDS_TAKE_THRESHOLD = 0
    EMERALDS_MAX_PASSIVE_SIZE = 80
    EMERALDS_RIDGE_LAMBDA = 0

    # Opportunistic taking parameters (dynamic via spread)
    EMERALDS_OPP_TAKE_SIGNAL_FRAC = 0.25
    EMERALDS_OPP_TAKE_PRICE_TOL_FRAC = 0.25
    EMERALDS_OPP_TAKE_MAX_SIZE = 80

    # Fair-based MM filter via relative fair position r
    EMERALDS_MM_R_LOW = 0.30
    EMERALDS_MM_R_HIGH = 0.70

    # ==========================================================
    # PARAMETERS - TOMATOES
    # ==========================================================
    TOMATOES_TAKE_THRESHOLD = 0
    TOMATOES_MAX_PASSIVE_SIZE = 80

    # Opportunistic taking parameters (dynamic via spread)
    TOMATOES_OPP_TAKE_SIGNAL_FRAC = 0.25
    TOMATOES_OPP_TAKE_PRICE_TOL_FRAC = 0.4
    TOMATOES_OPP_TAKE_MAX_SIZE = 80

    # Fair-based MM filter via relative fair position r
    TOMATOES_MM_R_LOW = 0.30
    TOMATOES_MM_R_HIGH = 0.70

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        self._trade_emeralds(state, result)
        self._trade_tomatoes(state, result)

        return result, conversions, traderData

    # ==========================================================
    # GENERIC HELPERS
    # ==========================================================
    def _best_bid(self, order_depth: OrderDepth) -> Optional[int]:
        return max(order_depth.buy_orders.keys(), default=None)

    def _best_ask(self, order_depth: OrderDepth) -> Optional[int]:
        return min(order_depth.sell_orders.keys(), default=None)

    def _midprice(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid = self._best_bid(order_depth)
        best_ask = self._best_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2

    def _microprice_level1(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid = self._best_bid(order_depth)
        best_ask = self._best_ask(order_depth)

        if best_bid is None or best_ask is None:
            return None

        bid_vol = order_depth.buy_orders.get(best_bid, 0)
        ask_vol = -order_depth.sell_orders.get(best_ask, 0)

        den = bid_vol + ask_vol
        if den <= 0:
            return self._midprice(order_depth)

        return (best_bid * ask_vol + best_ask * bid_vol) / den

    def _microprice_level1(self, order_depth: OrderDepth) -> Optional[float]:
        bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
        ask_prices = sorted(order_depth.sell_orders.keys())

        max_levels = min(2, len(bid_prices), len(ask_prices))
        if max_levels == 0:
            return self._midprice(order_depth)

        level_micros = []
        level_weights_raw = []

        for k in range(max_levels):
            bid_p = bid_prices[k]
            ask_p = ask_prices[k]

            bid_v = order_depth.buy_orders.get(bid_p, 0)
            ask_v = -order_depth.sell_orders.get(ask_p, 0)

            if bid_v < 0:
                bid_v = 0
            if ask_v < 0:
                ask_v = 0

            level_total = bid_v + ask_v
            if level_total <= 0:
                continue

            level_micro = (bid_p * ask_v + ask_p * bid_v) / level_total
            level_micros.append(level_micro)
            level_weights_raw.append(level_total ** 0.5)

        if not level_micros:
            return self._midprice(order_depth)

        weight_sum = sum(level_weights_raw)
        if weight_sum <= 0:
            return self._midprice(order_depth)

        micro = sum(
            (w / weight_sum) * m
            for w, m in zip(level_weights_raw, level_micros)
        )

        return micro


    def _position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def _buy_capacity(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] - pos

    def _sell_capacity(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] + pos

    def _take_asks_below(
        self,
        product: str,
        order_depth: OrderDepth,
        buy_capacity: int,
        max_acceptable_ask: int,
    ) -> tuple[List[Order], int]:
        orders: List[Order] = []

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if buy_capacity <= 0:
                break

            if ask_price <= max_acceptable_ask:
                available = -order_depth.sell_orders[ask_price]
                qty = min(available, buy_capacity)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_capacity -= qty
            else:
                break

        return orders, buy_capacity

    def _take_bids_above(
        self,
        product: str,
        order_depth: OrderDepth,
        sell_capacity: int,
        min_acceptable_bid: int,
    ) -> tuple[List[Order], int]:
        orders: List[Order] = []

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if sell_capacity <= 0:
                break

            if bid_price >= min_acceptable_bid:
                available = order_depth.buy_orders[bid_price]
                qty = min(available, sell_capacity)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_capacity -= qty
            else:
                break

        return orders, sell_capacity

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

    def _add_passive_quotes(
        self,
        product: str,
        orders: List[Order],
        bid_quote: int,
        ask_quote: int,
        buy_capacity: int,
        sell_capacity: int,
        max_passive_size: int,
    ) -> None:
        buy_quote_size = min(buy_capacity, max_passive_size)
        sell_quote_size = min(sell_capacity, max_passive_size)

        if buy_quote_size > 0:
            orders.append(Order(product, bid_quote, buy_quote_size))

        if sell_quote_size > 0:
            orders.append(Order(product, ask_quote, -sell_quote_size))

    def _fair_based_mm_filter(
        self,
        best_bid: int,
        best_ask: int,
        fair: int,
        r_low: float,
        r_high: float,
    ) -> tuple[bool, bool]:
        """
        Use the relative position of fair inside the current spread:

            r = (fair - best_bid) / (best_ask - best_bid)

        - r high  -> fair near ask -> keep bid only
        - r low   -> fair near bid -> keep ask only
        - r mid   -> keep both
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
    # RIDGE HELPERS
    # ==========================================================
    def _level2_volumes(self, order_depth: OrderDepth) -> tuple[int, int]:
        bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
        ask_prices = sorted(order_depth.sell_orders.keys())

        bid_vol_2 = order_depth.buy_orders.get(bid_prices[1], 0) if len(bid_prices) > 1 else 0
        ask_vol_2 = -order_depth.sell_orders.get(ask_prices[1], 0) if len(ask_prices) > 1 else 0

        return bid_vol_2, ask_vol_2

    def _level3_volumes(self, order_depth: OrderDepth) -> tuple[int, int]:
        bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
        ask_prices = sorted(order_depth.sell_orders.keys())

        bid_vol_3 = order_depth.buy_orders.get(bid_prices[2], 0) if len(bid_prices) > 2 else 0
        ask_vol_3 = -order_depth.sell_orders.get(ask_prices[2], 0) if len(ask_prices) > 2 else 0

        return bid_vol_3, ask_vol_3

    def _emeralds_fair_with_ridge(self, order_depth: OrderDepth) -> Optional[int]:
        best_bid = self._best_bid(order_depth)
        best_ask = self._best_ask(order_depth)
        microprice = self._microprice_level1(order_depth)
        mid = self._midprice(order_depth)

        if best_bid is None or best_ask is None or microprice is None or mid is None:
            return None

        spread = best_ask - best_bid

        bid_vol_1 = order_depth.buy_orders.get(best_bid, 0)
        ask_vol_1 = -order_depth.sell_orders.get(best_ask, 0)
        den1 = bid_vol_1 + ask_vol_1
        imbalance_1 = (bid_vol_1 - ask_vol_1) / den1 if den1 > 0 else 0.0

        bid_vol_2, ask_vol_2 = self._level2_volumes(order_depth)
        den2 = bid_vol_2 + ask_vol_2
        imbalance_2 = (bid_vol_2 - ask_vol_2) / den2 if den2 > 0 else 0.0

        bid_vol_3, ask_vol_3  = self._level3_volumes(order_depth)
        den3 = bid_vol_3 + ask_vol_3
        imbalance_3 = (bid_vol_3 - ask_vol_3) / den3 if den3 > 0 else 0.0

        imbalance_spread = imbalance_1/spread

        y_hat = (
            + 0.001192
            - 3.896924 * imbalance_3
            - 1.063487 * imbalance_2
            - 10.433977 * imbalance_1
            + 38.526487 * imbalance_spread
            - 0.030359 * spread
        )

        fair = microprice + self.EMERALDS_RIDGE_LAMBDA * y_hat
        return fair

    def _tomatoes_fair_with_ridge(self, order_depth: OrderDepth) -> Optional[int]:
        best_bid = self._best_bid(order_depth)
        best_ask = self._best_ask(order_depth)
        microprice = self._microprice_level1(order_depth)
        mid = self._midprice(order_depth)

        if best_bid is None or best_ask is None or microprice is None or mid is None:
            return None

        spread = best_ask - best_bid

        bid_vol_1 = order_depth.buy_orders.get(best_bid, 0)
        ask_vol_1 = -order_depth.sell_orders.get(best_ask, 0)
        den1 = bid_vol_1 + ask_vol_1
        imbalance_1 = (bid_vol_1 - ask_vol_1) / den1 if den1 > 0 else 0.0

        bid_vol_2, ask_vol_2 = self._level2_volumes(order_depth)
        den2 = bid_vol_2 + ask_vol_2
        imbalance_2 = (bid_vol_2 - ask_vol_2) / den2 if den2 > 0 else 0.0

        bid_vol_3, ask_vol_3 = self._level3_volumes(order_depth)
        den3 = bid_vol_3 + ask_vol_3
        imbalance_3 = (bid_vol_3 - ask_vol_3) / den3 if den3 > 0 else 0.0

        imbalance_spread = imbalance_1 / spread

        y_hat = (
            + 0.001192
            - 3.896924 * imbalance_3
            - 1.063487 * imbalance_2
            - 10.433977 * imbalance_1
            + 38.526487 * imbalance_spread
            - 0.030359 * spread
        )

        fair = microprice + self.TOMATOES_RIDGE_LAMBDA * y_hat
        return fair

    # ==========================================================
    # GENERAL OPPORTUNISTIC TAKING (DYNAMIC VIA SPREAD)
    # ==========================================================
    def _opportunistic_taking(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: int,
        mid: float,
        buy_capacity: int,
        sell_capacity: int,
        signal_frac: float,
        price_tol_frac: float,
        max_take_size: int,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []

        if max_take_size <= 0:
            return orders, buy_capacity, sell_capacity

        best_bid = self._best_bid(order_depth)
        best_ask = self._best_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders, buy_capacity, sell_capacity

        spread = best_ask - best_bid
        if spread <= 0:
            return orders, buy_capacity, sell_capacity

        signal = fair - mid

        dyn_signal_threshold = max(1, (signal_frac * spread))
        dyn_price_tol = max(1, (price_tol_frac * spread))

        # Opportunistic BUY taking
        if signal >= dyn_signal_threshold and buy_capacity > 0:
            max_buy_price = fair + dyn_price_tol
            max_buy_qty = min(buy_capacity, max_take_size)

            for ask_price in sorted(order_depth.sell_orders.keys()):
                if max_buy_qty <= 0:
                    break

                if ask_price <= max_buy_price:
                    available = -order_depth.sell_orders[ask_price]
                    qty = min(available, max_buy_qty)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty
                        max_buy_qty -= qty
                else:
                    break

        # Opportunistic SELL taking
        if signal <= -dyn_signal_threshold and sell_capacity > 0:
            min_sell_price = fair - dyn_price_tol
            max_sell_qty = min(sell_capacity, max_take_size)

            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if max_sell_qty <= 0:
                    break

                if bid_price >= min_sell_price:
                    available = order_depth.buy_orders[bid_price]
                    qty = min(available, max_sell_qty)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty
                        max_sell_qty -= qty
                else:
                    break

        return orders, buy_capacity, sell_capacity

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
        opp_take_signal_frac: float,
        opp_take_price_tol_frac: float,
        opp_take_max_size: int,
        mm_r_low: float,
        mm_r_high: float,
    ) -> None:
        if product not in state.order_depths:
            return

        order_depth = state.order_depths[product]
        orders: List[Order] = []

        pos = self._position(state, product)
        buy_capacity = self._buy_capacity(product, pos)
        sell_capacity = self._sell_capacity(product, pos)

        best_bid = self._best_bid(order_depth)
        best_ask = self._best_ask(order_depth)
        mid = self._midprice(order_depth)

        if mid is None:
            return


        # 1) Static taking
        if self.ENABLE_TAKING:
            take_buy_orders, buy_capacity = self._take_asks_below(
                product=product,
                order_depth=order_depth,
                buy_capacity=buy_capacity,
                max_acceptable_ask=fair - take_threshold,
            )
            orders.extend(take_buy_orders)

            take_sell_orders, sell_capacity = self._take_bids_above(
                product=product,
                order_depth=order_depth,
                sell_capacity=sell_capacity,
                min_acceptable_bid=fair + take_threshold,
            )
            orders.extend(take_sell_orders)

            # 2) Opportunistic taking (dynamic)
            opp_orders, buy_capacity, sell_capacity = self._opportunistic_taking(
                product=product,
                order_depth=order_depth,
                fair=fair,
                mid=mid,
                buy_capacity=buy_capacity,
                sell_capacity=sell_capacity,
                signal_frac=opp_take_signal_frac,
                price_tol_frac=opp_take_price_tol_frac,
                max_take_size=opp_take_max_size,
            )
            orders.extend(opp_orders)

        # 3) Making with fair MM filter
        if self.ENABLE_MAKING:
            passive_quotes = self._most_competitive_quotes(best_bid, best_ask)
            if passive_quotes is not None and best_bid is not None and best_ask is not None:
                bid_quote, ask_quote = passive_quotes

                place_bid, place_ask = self._fair_based_mm_filter(
                    best_bid=best_bid,
                    best_ask=best_ask,
                    fair=fair,
                    r_low=mm_r_low,
                    r_high=mm_r_high,
                )

                buy_quote_size = min(buy_capacity, max_passive_size)
                sell_quote_size = min(sell_capacity, max_passive_size)

                if place_bid and buy_quote_size > 0:
                    orders.append(Order(product, bid_quote, buy_quote_size))

                if place_ask and sell_quote_size > 0:
                    orders.append(Order(product, ask_quote, -sell_quote_size))

        result[product] = orders

    # ==========================================================
    # EMERALDS
    # ==========================================================
    def _trade_emeralds(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        product = self.EMERALDS
        if product not in state.order_depths:
            return

        fair = self._emeralds_fair_with_ridge(state.order_depths[product])
        if fair is None:
            return

        self._trade_product(
            state=state,
            result=result,
            product=product,
            fair=fair,
            take_threshold=self.EMERALDS_TAKE_THRESHOLD,
            max_passive_size=self.EMERALDS_MAX_PASSIVE_SIZE,
            opp_take_signal_frac=self.EMERALDS_OPP_TAKE_SIGNAL_FRAC,
            opp_take_price_tol_frac=self.EMERALDS_OPP_TAKE_PRICE_TOL_FRAC,
            opp_take_max_size=self.EMERALDS_OPP_TAKE_MAX_SIZE,
            mm_r_low=self.EMERALDS_MM_R_LOW,
            mm_r_high=self.EMERALDS_MM_R_HIGH,
        )

    # ==========================================================
    # TOMATOES
    # ==========================================================
    def _trade_tomatoes(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        product = self.TOMATOES
        if product not in state.order_depths:
            return

        # 1. Calcola il Fair Price usando il Deep Microprice sui primi 3 livelli
        fair = self._microprice_level1(state.order_depths[product])

        if fair is None:
            return

        self._trade_product(
            state=state,
            result=result,
            product=product,
            fair=fair,
            take_threshold=self.TOMATOES_TAKE_THRESHOLD,
            max_passive_size=self.TOMATOES_MAX_PASSIVE_SIZE,
            opp_take_signal_frac=self.TOMATOES_OPP_TAKE_SIGNAL_FRAC,
            opp_take_price_tol_frac=self.TOMATOES_OPP_TAKE_PRICE_TOL_FRAC,
            opp_take_max_size=self.TOMATOES_OPP_TAKE_MAX_SIZE,
            mm_r_low=self.TOMATOES_MM_R_LOW,
            mm_r_high=self.TOMATOES_MM_R_HIGH,
        )