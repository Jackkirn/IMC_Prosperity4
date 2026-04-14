from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math


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
    ASH_MM_R_LOW = 0.40
    ASH_MM_R_HIGH = 0.60

    # ==========================================================
    # OU OVERLAY — ASH treated as synthetic log-ratio itself
    # Calibrated on round-1 data using x_t = log(mid_ash)
    # AR(1): x_{t+1} = a + b x_t + eps
    #   b ≈ 0.7573, mu ≈ 9.21036, sigma_stat ≈ 5.35e-4
    # ==========================================================
    ASH_OU_ENABLED = True
    ASH_OU_MIN_HISTORY = 40
    ASH_OU_ALPHA = 0.075          # reactive online mean/var for synthetic spread
    ASH_OU_ENTRY_D = 0.922        # benchmark entry band in sigma units
    ASH_OU_EXIT_U = 0.15          # overlay neutral zone around mean
    ASH_OU_STOP_L = 1.645         # hard cap for signal clipping / de-risk
    ASH_OU_FAIR_WEIGHT = 0.90     # how much of the signal shifts the fair
    ASH_OU_MAX_SKEW_TICKS = 4     # max fair displacement vs microprice
    ASH_OU_EXTREME_Z = 1.50       # stronger skew beyond this level
    ASH_OU_EXTRA_TAKE = 1         # extra taking aggressiveness when extreme

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        ctx = self._load_ctx(state.traderData)

        self._trade_pepper(state, result)
        self._trade_ash(state, result, ctx)

        return result, 0, json.dumps(ctx)

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
    # FAIR — ASH base + OU overlay
    # ==========================================================
    def _ash_fair_with_ou(self, od: OrderDepth, ctx: dict) -> Optional[tuple[int, dict]]:
        mp = self._microprice_l1(od)
        if mp is None or mp <= 0:
            return None

        x = math.log(mp)
        self._update_ash_ou(ctx, x)
        ou = ctx["ash_ou"]

        mu = ou["mu"]
        sigma = math.sqrt(max(ou["var"], 0.0))
        ticks = ou["ticks"]
        z = 0.0
        fair = round(mp)

        if ticks >= self.ASH_OU_MIN_HISTORY and sigma > 1e-9:
            z = (x - mu) / sigma
            z_clip = max(-self.ASH_OU_STOP_L, min(self.ASH_OU_STOP_L, z))

            if abs(z_clip) >= self.ASH_OU_EXIT_U:
                shift = -self.ASH_OU_FAIR_WEIGHT * z_clip * sigma
                shift_ticks = int(round(shift * mp))
                shift_ticks = max(-self.ASH_OU_MAX_SKEW_TICKS,
                                  min(self.ASH_OU_MAX_SKEW_TICKS, shift_ticks))
                fair += shift_ticks

        ou["last_x"] = round(x, 8)
        ou["last_mu"] = round(mu, 8)
        ou["last_sigma"] = round(sigma, 8)
        ou["last_z"] = round(z, 5)
        ou["last_fair"] = fair

        return fair, ou

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

        if buy_cap > 0:
            best_bid = self._best_bid(od)
            best_ask = self._best_ask(od)
            if best_ask is not None:
                take_orders, buy_cap = self._take_asks_below(
                    product=product,
                    od=od,
                    buy_cap=buy_cap,
                    max_ask=best_ask,
                )
                orders.extend(take_orders)

            if buy_cap > 0 and best_bid is not None:
                orders.append(Order(product, best_bid + 1, buy_cap))

        result[product] = orders

    # ==========================================================
    # ASH (MARKET MAKING + OU MR OVERLAY)
    # ==========================================================
    def _trade_ash(self, state: TradingState, result: Dict[str, List[Order]], ctx: dict) -> None:
        product = self.ASH
        if product not in state.order_depths:
            return

        od = state.order_depths[product]
        fair_pack = self._ash_fair_with_ou(od, ctx)
        if fair_pack is None:
            return
        fair, ou = fair_pack

        take_threshold = self.ASH_TAKE_THRESHOLD
        max_passive_size = self.ASH_MAX_PASSIVE_SIZE
        r_low = self.ASH_MM_R_LOW
        r_high = self.ASH_MM_R_HIGH

        z = abs(ou.get("last_z", 0.0))
        if z >= self.ASH_OU_EXTREME_Z:
            take_threshold = max(0, take_threshold - self.ASH_OU_EXTRA_TAKE)
            max_passive_size = min(self.POSITION_LIMITS[product], max_passive_size + 5)
            if ou.get("last_z", 0.0) > 0:
                r_low, r_high = 0.25, 0.55
            else:
                r_low, r_high = 0.45, 0.75

        self._trade_product(
            state=state,
            result=result,
            product=product,
            fair=fair,
            take_threshold=take_threshold,
            max_passive_size=max_passive_size,
            mm_r_low=r_low,
            mm_r_high=r_high,
        )

    # ==========================================================
    # OU STATE HELPERS
    # ==========================================================
    def _update_ash_ou(self, ctx: dict, x: float) -> None:
        ou = ctx["ash_ou"]
        alpha = self.ASH_OU_ALPHA
        ou["ticks"] += 1
        if ou["ticks"] == 1:
            ou["mu"] = x
            ou["var"] = 0.0
            return

        delta = x - ou["mu"]
        ou["mu"] += alpha * delta
        ou["var"] = (1 - alpha) * (ou["var"] + alpha * delta * delta)

    @staticmethod
    def _load_ctx(raw: str) -> dict:
        if raw:
            try:
                ctx = json.loads(raw)
                if isinstance(ctx, dict) and "ash_ou" in ctx:
                    return ctx
            except Exception:
                pass
        return {
            "ash_ou": {
                "ticks": 0,
                "mu": math.log(10000.0),
                "var": 0.0,
                "last_x": 0.0,
                "last_mu": 0.0,
                "last_sigma": 0.0,
                "last_z": 0.0,
                "last_fair": 10000,
            }
        }
