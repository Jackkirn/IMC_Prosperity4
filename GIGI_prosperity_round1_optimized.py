from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    ASH = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"

    POSITION_LIMITS = {
        ASH: 80,
        PEPPER: 80,
    }

    # ---------- ASH: stable around 10000 ----------
    ASH_FAIR = 10000.0
    ASH_TAKE_THRESHOLD = 7
    ASH_MAKE_OFFSET = 3
    ASH_MAX_PASSIVE = 20
    ASH_INV_SKEW = 0.05

    # ---------- PEPPER: clear intraday upward drift ----------
    PEPPER_DRIFT_PER_TS = 0.001  # fair += 0.001 * timestamp
    PEPPER_TAKE_THRESHOLD = 2
    PEPPER_MAKE_OFFSET = 2
    PEPPER_MAX_PASSIVE = 16
    PEPPER_INV_SKEW = 0.08

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        # initialize pepper anchor from the first valid mid seen in the day
        pepper_depth = state.order_depths.get(self.PEPPER)
        pepper_mid = self._midprice(pepper_depth) if pepper_depth is not None else None
        if data.get("pepper_anchor") is None and pepper_mid is not None:
            data["pepper_anchor"] = round(pepper_mid)

        # fallback if the first snapshot is one-sided
        if data.get("pepper_anchor") is None:
            data["pepper_anchor"] = 12000

        self._trade_ash(state, result)
        self._trade_pepper(state, result, float(data["pepper_anchor"]))

        trader_data = json.dumps(data, separators=(",", ":"))
        return result, 0, trader_data

    # ==========================================================
    # Helpers
    # ==========================================================
    def _load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {"pepper_anchor": None}
        try:
            d = json.loads(trader_data)
            if not isinstance(d, dict):
                return {"pepper_anchor": None}
            d.setdefault("pepper_anchor", None)
            return d
        except Exception:
            return {"pepper_anchor": None}

    def _best_bid(self, depth: OrderDepth) -> Optional[int]:
        if depth is None or not depth.buy_orders:
            return None
        return max(depth.buy_orders.keys())

    def _best_ask(self, depth: OrderDepth) -> Optional[int]:
        if depth is None or not depth.sell_orders:
            return None
        return min(depth.sell_orders.keys())

    def _midprice(self, depth: OrderDepth) -> Optional[float]:
        best_bid = self._best_bid(depth)
        best_ask = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def _position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def _buy_capacity(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] - pos

    def _sell_capacity(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] + pos

    def _take_asks_below(
        self,
        product: str,
        depth: OrderDepth,
        buy_capacity: int,
        max_acceptable_ask: int,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        if buy_capacity <= 0:
            return orders, buy_capacity
        for ask_price in sorted(depth.sell_orders.keys()):
            if ask_price > max_acceptable_ask or buy_capacity <= 0:
                break
            available = -depth.sell_orders[ask_price]
            qty = min(available, buy_capacity)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
        return orders, buy_capacity

    def _take_bids_above(
        self,
        product: str,
        depth: OrderDepth,
        sell_capacity: int,
        min_acceptable_bid: int,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        if sell_capacity <= 0:
            return orders, sell_capacity
        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_price < min_acceptable_bid or sell_capacity <= 0:
                break
            available = depth.buy_orders[bid_price]
            qty = min(available, sell_capacity)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
        return orders, sell_capacity

    def _add_mm_quotes(
        self,
        product: str,
        orders: List[Order],
        depth: OrderDepth,
        fair: float,
        pos: int,
        buy_capacity: int,
        sell_capacity: int,
        make_offset: int,
        max_passive: int,
        inv_skew: float,
    ) -> None:
        best_bid = self._best_bid(depth)
        best_ask = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return

        adj_fair = fair - inv_skew * pos
        bid_target = math.floor(adj_fair - make_offset)
        ask_target = math.ceil(adj_fair + make_offset)

        # improve inside the book without crossing
        bid_quote = min(bid_target, best_ask - 1)
        ask_quote = max(ask_target, best_bid + 1)

        if bid_quote >= ask_quote:
            return

        # inventory-aware side selection
        buy_size = min(buy_capacity, max_passive)
        sell_size = min(sell_capacity, max_passive)

        if pos > 50:
            buy_size = 0
            sell_size = min(sell_capacity, max_passive + 10)
        elif pos < -50:
            sell_size = 0
            buy_size = min(buy_capacity, max_passive + 10)

        if buy_size > 0:
            orders.append(Order(product, bid_quote, buy_size))
        if sell_size > 0:
            orders.append(Order(product, ask_quote, -sell_size))

    # ==========================================================
    # Product logic
    # ==========================================================
    def _trade_ash(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        product = self.ASH
        depth = state.order_depths.get(product)
        if depth is None:
            return

        pos = self._position(state, product)
        buy_cap = self._buy_capacity(product, pos)
        sell_cap = self._sell_capacity(product, pos)
        fair = self.ASH_FAIR
        orders: List[Order] = []

        buy_orders, buy_cap = self._take_asks_below(
            product, depth, buy_cap, int(math.floor(fair - self.ASH_TAKE_THRESHOLD))
        )
        sell_orders, sell_cap = self._take_bids_above(
            product, depth, sell_cap, int(math.ceil(fair + self.ASH_TAKE_THRESHOLD))
        )
        orders.extend(buy_orders)
        orders.extend(sell_orders)

        self._add_mm_quotes(
            product=product,
            orders=orders,
            depth=depth,
            fair=fair,
            pos=pos,
            buy_capacity=buy_cap,
            sell_capacity=sell_cap,
            make_offset=self.ASH_MAKE_OFFSET,
            max_passive=self.ASH_MAX_PASSIVE,
            inv_skew=self.ASH_INV_SKEW,
        )
        result[product] = orders

    def _trade_pepper(self, state: TradingState, result: Dict[str, List[Order]], pepper_anchor: float) -> None:
        product = self.PEPPER
        depth = state.order_depths.get(product)
        if depth is None:
            return

        pos = self._position(state, product)
        buy_cap = self._buy_capacity(product, pos)
        sell_cap = self._sell_capacity(product, pos)

        # deterministic trend-like fair inferred from public data and hidden logs
        fair = pepper_anchor + self.PEPPER_DRIFT_PER_TS * state.timestamp
        orders: List[Order] = []

        buy_orders, buy_cap = self._take_asks_below(
            product, depth, buy_cap, int(math.floor(fair - self.PEPPER_TAKE_THRESHOLD))
        )
        sell_orders, sell_cap = self._take_bids_above(
            product, depth, sell_cap, int(math.ceil(fair + self.PEPPER_TAKE_THRESHOLD))
        )
        orders.extend(buy_orders)
        orders.extend(sell_orders)

        self._add_mm_quotes(
            product=product,
            orders=orders,
            depth=depth,
            fair=fair,
            pos=pos,
            buy_capacity=buy_cap,
            sell_capacity=sell_cap,
            make_offset=self.PEPPER_MAKE_OFFSET,
            max_passive=self.PEPPER_MAX_PASSIVE,
            inv_skew=self.PEPPER_INV_SKEW,
        )
        result[product] = orders
