
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

    # ---------------- ASH: stable / mean reverting around 10000 ----------------
    ASH_FAIR = 10000.0
    ASH_TAKE_EDGE = 2
    ASH_QUOTE_EDGE = 4
    ASH_PASSIVE_SIZE = 35
    ASH_INV_PENALTY = 0.12

    # --------------- PEPPER: strong intraday upward drift ----------------------
    PEPPER_DRIFT = 0.001
    PEPPER_ALPHA_BASE = 0.10  # update speed for anchor estimate
    PEPPER_TAKE_BUY_EDGE = 1
    PEPPER_TAKE_SELL_EDGE = 8
    PEPPER_BID_EDGE = 2
    PEPPER_ASK_EDGE = 5
    PEPPER_PASSIVE_SIZE = 32
    PEPPER_INV_PENALTY = 0.10
    PEPPER_TARGET_GAIN = 0.55  # turns target-pos into price shift

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        self._update_pepper_anchor(state, data)

        self._trade_ash(state, result)
        self._trade_pepper(state, result, data)

        return result, 0, json.dumps(data, separators=(",", ":"))

    # -------------------------------------------------------------------------
    # State helpers
    # -------------------------------------------------------------------------
    def _load_data(self, trader_data: str) -> dict:
        default = {"pepper_base": None}
        if not trader_data:
            return default
        try:
            d = json.loads(trader_data)
            if not isinstance(d, dict):
                return default
            d.setdefault("pepper_base", None)
            return d
        except Exception:
            return default

    def _update_pepper_anchor(self, state: TradingState, data: dict) -> None:
        depth = state.order_depths.get(self.PEPPER)
        mid = self._midprice(depth) if depth is not None else None
        if mid is None:
            if data.get("pepper_base") is None:
                data["pepper_base"] = 12000.0
            return

        implied_base = float(mid) - self.PEPPER_DRIFT * float(state.timestamp)
        if data.get("pepper_base") is None:
            data["pepper_base"] = implied_base
        else:
            data["pepper_base"] = (
                (1.0 - self.PEPPER_ALPHA_BASE) * float(data["pepper_base"])
                + self.PEPPER_ALPHA_BASE * implied_base
            )

    # -------------------------------------------------------------------------
    # Basic market helpers
    # -------------------------------------------------------------------------
    def _best_bid(self, depth: OrderDepth) -> Optional[int]:
        return max(depth.buy_orders.keys(), default=None) if depth is not None else None

    def _best_ask(self, depth: OrderDepth) -> Optional[int]:
        return min(depth.sell_orders.keys(), default=None) if depth is not None else None

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

    # -------------------------------------------------------------------------
    # Order helpers
    # -------------------------------------------------------------------------
    def _take_asks_up_to(
        self,
        product: str,
        depth: OrderDepth,
        buy_capacity: int,
        max_price: int,
        max_qty: Optional[int] = None,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        remaining = buy_capacity if max_qty is None else min(buy_capacity, max_qty)
        for ask_price in sorted(depth.sell_orders.keys()):
            if remaining <= 0 or ask_price > max_price:
                break
            available = -depth.sell_orders[ask_price]
            qty = min(available, remaining)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                remaining -= qty
        return orders, buy_capacity

    def _take_bids_down_to(
        self,
        product: str,
        depth: OrderDepth,
        sell_capacity: int,
        min_price: int,
        max_qty: Optional[int] = None,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        remaining = sell_capacity if max_qty is None else min(sell_capacity, max_qty)
        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            if remaining <= 0 or bid_price < min_price:
                break
            available = depth.buy_orders[bid_price]
            qty = min(available, remaining)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                remaining -= qty
        return orders, sell_capacity

    def _clip_quotes(self, best_bid: int, best_ask: int, bid_quote: int, ask_quote: int) -> Tuple[int, int]:
        if best_bid + 1 < best_ask:
            bid_quote = max(bid_quote, best_bid + 1)
            ask_quote = min(ask_quote, best_ask - 1)
        else:
            bid_quote = min(max(bid_quote, best_bid), best_ask)
            ask_quote = max(min(ask_quote, best_ask), best_bid)
        if bid_quote > ask_quote:
            bid_quote, ask_quote = best_bid, best_ask
        return bid_quote, ask_quote

    # -------------------------------------------------------------------------
    # ASH strategy
    # -------------------------------------------------------------------------
    def _trade_ash(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        depth = state.order_depths.get(self.ASH)
        if depth is None:
            return

        best_bid = self._best_bid(depth)
        best_ask = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return

        pos = self._position(state, self.ASH)
        buy_cap = self._buy_capacity(self.ASH, pos)
        sell_cap = self._sell_capacity(self.ASH, pos)
        orders: List[Order] = []

        reservation = self.ASH_FAIR - self.ASH_INV_PENALTY * pos

        take_buy, buy_cap = self._take_asks_up_to(
            self.ASH, depth, buy_cap, int(math.floor(reservation - self.ASH_TAKE_EDGE))
        )
        orders.extend(take_buy)

        take_sell, sell_cap = self._take_bids_down_to(
            self.ASH, depth, sell_cap, int(math.ceil(reservation + self.ASH_TAKE_EDGE))
        )
        orders.extend(take_sell)

        best_bid = self._best_bid(depth)
        best_ask = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            result[self.ASH] = orders
            return

        bid_quote = int(math.floor(reservation - self.ASH_QUOTE_EDGE))
        ask_quote = int(math.ceil(reservation + self.ASH_QUOTE_EDGE))
        bid_quote, ask_quote = self._clip_quotes(best_bid, best_ask, bid_quote, ask_quote)

        bid_size = min(buy_cap, self.ASH_PASSIVE_SIZE)
        ask_size = min(sell_cap, self.ASH_PASSIVE_SIZE)

        if bid_size > 0:
            orders.append(Order(self.ASH, bid_quote, bid_size))
        if ask_size > 0:
            orders.append(Order(self.ASH, ask_quote, -ask_size))

        result[self.ASH] = orders

    # -------------------------------------------------------------------------
    # PEPPER strategy
    # -------------------------------------------------------------------------
    def _pepper_target_position(self, timestamp: int) -> int:
        if timestamp < 20000:
            return 80
        if timestamp < 50000:
            return 60
        if timestamp < 80000:
            return 35
        if timestamp < 92000:
            return 10
        return 0

    def _trade_pepper(self, state: TradingState, result: Dict[str, List[Order]], data: dict) -> None:
        depth = state.order_depths.get(self.PEPPER)
        if depth is None:
            return

        best_bid = self._best_bid(depth)
        best_ask = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return

        pos = self._position(state, self.PEPPER)
        buy_cap = self._buy_capacity(self.PEPPER, pos)
        sell_cap = self._sell_capacity(self.PEPPER, pos)
        orders: List[Order] = []

        base = float(data.get("pepper_base", 12000.0))
        fair = base + self.PEPPER_DRIFT * float(state.timestamp)

        target = self._pepper_target_position(state.timestamp)
        target_shift = self.PEPPER_TARGET_GAIN * float(target - pos)

        reservation = fair + target_shift - self.PEPPER_INV_PENALTY * pos

        # Directionally biased taking: buy relatively easily, sell only when very rich
        take_buy, buy_cap = self._take_asks_up_to(
            self.PEPPER,
            depth,
            buy_cap,
            int(math.floor(reservation - self.PEPPER_TAKE_BUY_EDGE)),
            max_qty=40,
        )
        orders.extend(take_buy)

        take_sell, sell_cap = self._take_bids_down_to(
            self.PEPPER,
            depth,
            sell_cap,
            int(math.ceil(reservation + self.PEPPER_TAKE_SELL_EDGE)),
            max_qty=30,
        )
        orders.extend(take_sell)

        best_bid = self._best_bid(depth)
        best_ask = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            result[self.PEPPER] = orders
            return

        bid_quote = int(math.floor(reservation - self.PEPPER_BID_EDGE))
        ask_quote = int(math.ceil(reservation + self.PEPPER_ASK_EDGE))
        bid_quote, ask_quote = self._clip_quotes(best_bid, best_ask, bid_quote, ask_quote)

        if pos < target:
            bid_size = min(buy_cap, self.PEPPER_PASSIVE_SIZE + 16)
            ask_size = min(sell_cap, max(4, self.PEPPER_PASSIVE_SIZE // 3))
        elif pos > target + 20:
            bid_size = min(buy_cap, max(4, self.PEPPER_PASSIVE_SIZE // 3))
            ask_size = min(sell_cap, self.PEPPER_PASSIVE_SIZE + 12)
        else:
            bid_size = min(buy_cap, self.PEPPER_PASSIVE_SIZE)
            ask_size = min(sell_cap, self.PEPPER_PASSIVE_SIZE // 2)

        if bid_size > 0:
            orders.append(Order(self.PEPPER, bid_quote, bid_size))
        if ask_size > 0:
            orders.append(Order(self.PEPPER, ask_quote, -ask_size))

        result[self.PEPPER] = orders
