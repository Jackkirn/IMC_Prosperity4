
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json


class Trader:
    ASH = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"

    POSITION_LIMITS = {
        ASH: 80,
        PEPPER: 80,
    }

    ASH_FAIR = 10000

    def run(self, state: TradingState):
        trader_data = self._load_trader_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        self._trade_ash(state, result)
        self._trade_pepper(state, result, trader_data)

        return result, 0, json.dumps(trader_data, separators=(",", ":"))

    # -----------------------------
    # generic helpers
    # -----------------------------
    def _load_trader_data(self, raw: str) -> dict:
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _best_bid(self, od: OrderDepth) -> Optional[int]:
        return max(od.buy_orders.keys(), default=None)

    def _best_ask(self, od: OrderDepth) -> Optional[int]:
        return min(od.sell_orders.keys(), default=None)

    def _mid(self, od: OrderDepth) -> Optional[float]:
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def _microprice(self, od: OrderDepth) -> Optional[float]:
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return None

        bv = max(0, od.buy_orders.get(bb, 0))
        av = max(0, -od.sell_orders.get(ba, 0))
        den = bv + av
        if den <= 0:
            return self._mid(od)
        return (bb * av + ba * bv) / den

    def _level2_imbalance(self, od: OrderDepth) -> float:
        bids = sorted(od.buy_orders.keys(), reverse=True)
        asks = sorted(od.sell_orders.keys())
        if len(bids) < 2 or len(asks) < 2:
            return 0.0
        bv2 = max(0, od.buy_orders.get(bids[1], 0))
        av2 = max(0, -od.sell_orders.get(asks[1], 0))
        den = bv2 + av2
        if den <= 0:
            return 0.0
        return (bv2 - av2) / den

    def _l1_imbalance(self, od: OrderDepth) -> float:
        bb = self._best_bid(od)
        ba = self._best_ask(od)
        if bb is None or ba is None:
            return 0.0
        bv = max(0, od.buy_orders.get(bb, 0))
        av = max(0, -od.sell_orders.get(ba, 0))
        den = bv + av
        if den <= 0:
            return 0.0
        return (bv - av) / den

    def _buy_cap(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] - pos

    def _sell_cap(self, product: str, pos: int) -> int:
        return self.POSITION_LIMITS[product] + pos

    def _take_asks(self, product: str, od: OrderDepth, buy_cap: int, max_price: int, max_qty: int) -> tuple[List[Order], int]:
        orders: List[Order] = []
        rem = min(buy_cap, max_qty)
        for ask in sorted(od.sell_orders):
            if rem <= 0:
                break
            if ask > max_price:
                break
            avail = max(0, -od.sell_orders[ask])
            qty = min(avail, rem)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                rem -= qty
                buy_cap -= qty
        return orders, buy_cap

    def _take_bids(self, product: str, od: OrderDepth, sell_cap: int, min_price: int, max_qty: int) -> tuple[List[Order], int]:
        orders: List[Order] = []
        rem = min(sell_cap, max_qty)
        for bid in sorted(od.buy_orders, reverse=True):
            if rem <= 0:
                break
            if bid < min_price:
                break
            avail = max(0, od.buy_orders[bid])
            qty = min(avail, rem)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                rem -= qty
                sell_cap -= qty
        return orders, sell_cap

    # -----------------------------
    # ASH: mean-reversion + aggressive MM
    # -----------------------------
    def _trade_ash(self, state: TradingState, result: Dict[str, List[Order]]) -> None:
        product = self.ASH
        od = state.order_depths.get(product)
        if od is None:
            return

        bb = self._best_bid(od)
        ba = self._best_ask(od)
        mid = self._mid(od)
        micro = self._microprice(od)
        if bb is None or ba is None or mid is None or micro is None:
            result[product] = []
            return

        pos = state.position.get(product, 0)
        buy_cap = self._buy_cap(product, pos)
        sell_cap = self._sell_cap(product, pos)

        spread = ba - bb
        micro_edge = micro - mid
        imb1 = self._l1_imbalance(od)
        imb2 = self._level2_imbalance(od)

        # keep ASH mostly anchored to 10000, with modest local alpha only
        fair = self.ASH_FAIR + 0.35 * micro_edge + 0.75 * imb1 + 0.35 * imb2

        # stronger inventory skew so ASH recenters faster instead of sitting loaded
        fair -= 0.085 * pos

        orders: List[Order] = []

        # Taking around static value; allow a bit more aggression when spread is decent.
        buy_take_px = min(int(fair - 1), 9999)
        sell_take_px = max(int(fair + 1), 10001)

        take_buy_qty = 24 if spread >= 6 else 12
        take_sell_qty = 24 if spread >= 6 else 12

        take_orders, buy_cap = self._take_asks(product, od, buy_cap, buy_take_px, take_buy_qty)
        orders.extend(take_orders)
        take_orders, sell_cap = self._take_bids(product, od, sell_cap, sell_take_px, take_sell_qty)
        orders.extend(take_orders)

        # Recompute residual capacity after taking
        bought = sum(o.quantity for o in orders if o.quantity > 0)
        sold = -sum(o.quantity for o in orders if o.quantity < 0)
        buy_cap = max(0, self._buy_cap(product, pos) - bought)
        sell_cap = max(0, self._sell_cap(product, pos) - sold)

        # Aggressive two-level MM; quote width expands a touch only if spread is tiny.
        inner_bid = min(bb + 1, ba - 1) if bb + 1 <= ba - 1 else bb
        inner_ask = max(ba - 1, bb + 1) if bb + 1 <= ba - 1 else ba
        outer_bid = bb
        outer_ask = ba

        # inventory-aware skew
        if pos > 50:
            inner_bid -= 1
            outer_bid -= 1
        elif pos < -50:
            inner_ask += 1
            outer_ask += 1

        # parameterized sizes
        inner_size = 34
        outer_size = 24

        # suppress adverse side when far from fair
        if fair > mid + 0.8:
            place_bid_inner, place_ask_inner = True, False
        elif fair < mid - 0.8:
            place_bid_inner, place_ask_inner = False, True
        else:
            place_bid_inner, place_ask_inner = True, True

        if place_bid_inner and buy_cap > 0:
            q = min(inner_size, buy_cap)
            if q > 0:
                orders.append(Order(product, inner_bid, q))
                buy_cap -= q
        if place_ask_inner and sell_cap > 0:
            q = min(inner_size, sell_cap)
            if q > 0:
                orders.append(Order(product, inner_ask, -q))
                sell_cap -= q

        # second layer more neutral and smaller
        if buy_cap > 0:
            q = min(outer_size, buy_cap)
            if q > 0:
                orders.append(Order(product, outer_bid, q))
                buy_cap -= q
        if sell_cap > 0:
            q = min(outer_size, sell_cap)
            if q > 0:
                orders.append(Order(product, outer_ask, -q))
                sell_cap -= q

        result[product] = orders

    # -----------------------------
    # PEPPER: trend carry + scalp around trend
    # -----------------------------
    def _pepper_anchor(self, state: TradingState, od: OrderDepth, trader_data: dict) -> float:
        key = "pepper_anchor"
        mp = self._microprice(od)
        mid = self._mid(od)
        spot = mp if mp is not None else (mid if mid is not None else 0.0)
        if key not in trader_data:
            # remove early-path noise from first print
            trader_data[key] = float(spot - 0.0092 * state.timestamp)
        return float(trader_data[key])

    def _pepper_target(self, ts: int) -> int:
        # smoother ramp: fix early deep drawdown while still harvesting the carry
        if ts < 8_000:
            return 12
        if ts < 16_000:
            return 24
        if ts < 28_000:
            return 40
        if ts < 42_000:
            return 56
        if ts < 82_000:
            return 72
        if ts < 92_000:
            return 48
        return 16

    def _trade_pepper(self, state: TradingState, result: Dict[str, List[Order]], trader_data: dict) -> None:
        product = self.PEPPER
        od = state.order_depths.get(product)
        if od is None:
            return

        bb = self._best_bid(od)
        ba = self._best_ask(od)
        mid = self._mid(od)
        micro = self._microprice(od)
        if bb is None or ba is None or mid is None or micro is None:
            result[product] = []
            return

        pos = state.position.get(product, 0)
        buy_cap = self._buy_cap(product, pos)
        sell_cap = self._sell_cap(product, pos)
        ts = state.timestamp

        anchor = self._pepper_anchor(state, od, trader_data)
        spread = ba - bb
        imb1 = self._l1_imbalance(od)
        imb2 = self._level2_imbalance(od)
        micro_edge = micro - mid

        # public days suggest roughly +0.009~0.01 per timestamp.
        trend_fair = anchor + 0.0092 * ts

        # local alpha as timing, not as main engine
        local_alpha = 2.2 * micro_edge + 3.0 * imb1 + 1.2 * imb2

        target = self._pepper_target(ts)
        inv_gap = target - pos

        # encourage holding the target, but not with absurd force
        inventory_bias = 0.06 * inv_gap

        fair = trend_fair + local_alpha + inventory_bias

        orders: List[Order] = []

        # Early session: less crossing. Mid/late: more willing to pay for inventory if below target.
        if ts < 12_000:
            base_buy_extra = 0
            take_buy_qty = 8
        elif ts < 40_000:
            base_buy_extra = 1
            take_buy_qty = 16
        else:
            base_buy_extra = 2
            take_buy_qty = 22

        # Asymmetric crossing: buy to build carry, sell only tactically or when over target.
        buy_cross_px = int(fair - 1 + base_buy_extra)
        sell_cross_px = int(fair + 5)

        if inv_gap > 0:
            buy_cross_px += 1
        if inv_gap < -8:
            sell_cross_px -= 2

        take_orders, buy_cap = self._take_asks(product, od, buy_cap, buy_cross_px, take_buy_qty)
        orders.extend(take_orders)

        tactical_sell_qty = 0
        if pos > target + 12:
            tactical_sell_qty = 14
        elif ts > 88_000 and pos > 8:
            tactical_sell_qty = 18

        if tactical_sell_qty > 0:
            take_orders, sell_cap = self._take_bids(product, od, sell_cap, sell_cross_px, tactical_sell_qty)
            orders.extend(take_orders)

        bought = sum(o.quantity for o in orders if o.quantity > 0)
        sold = -sum(o.quantity for o in orders if o.quantity < 0)
        buy_cap = max(0, self._buy_cap(product, pos) - bought)
        sell_cap = max(0, self._sell_cap(product, pos) - sold)

        # Passive quoting around fair, with strong bid skew when below target.
        inner_bid = min(bb + 1, ba - 1) if bb + 1 <= ba - 1 else bb
        inner_ask = max(ba - 1, bb + 1) if bb + 1 <= ba - 1 else ba
        outer_bid = bb
        outer_ask = ba

        bid_size = 0
        ask_size = 0

        if inv_gap > 24:
            bid_size = 30
            ask_size = 0
        elif inv_gap > 8:
            bid_size = 22
            ask_size = 6
        elif inv_gap > -8:
            bid_size = 18
            ask_size = 10
        else:
            bid_size = 10
            ask_size = 18

        # opening is intentionally softer to avoid deep early drawdown
        if ts < 10_000:
            bid_size = max(8, bid_size - 8)
            ask_size = max(0, ask_size - 4)

        # late unwind
        if ts > 90_000:
            ask_size += 12
            bid_size = max(0, bid_size - 6)

        if buy_cap > 0 and bid_size > 0:
            q = min(bid_size, buy_cap)
            if q > 0:
                orders.append(Order(product, inner_bid, q))
                buy_cap -= q
        if sell_cap > 0 and ask_size > 0:
            q = min(ask_size, sell_cap)
            if q > 0:
                orders.append(Order(product, inner_ask, -q))
                sell_cap -= q

        # extra outer bid for carry, but only while still below target
        if inv_gap > 10 and buy_cap > 0:
            q = min(14, buy_cap)
            if q > 0:
                orders.append(Order(product, outer_bid, q))
                buy_cap -= q

        # extra outer ask only late or clearly over target
        if (ts > 92_000 or pos > target + 20) and sell_cap > 0:
            q = min(16, sell_cap)
            if q > 0:
                orders.append(Order(product, outer_ask, -q))
                sell_cap -= q

        result[product] = orders
