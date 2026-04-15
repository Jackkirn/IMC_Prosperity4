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
    # PARAMETERS — PEPPER
    # ==========================================================
    PEPPER_MAX_PASSIVE_SIZE = 80
    PEPPER_AGGRESSIVE_ACCUMULATION = True

    # ==========================================================
    # PARAMETERS — ASH
    # ==========================================================
    ASH_TAKE_THRESHOLD = 0
    ASH_MAX_PASSIVE_SIZE = 20
    ASH_MM_R_LOW = 0.00
    ASH_MM_R_HIGH = 1.00

    # Online adaptive MR parameters
    ASH_EMA_ALPHA = 0.02               # update speed for mu/var
    ASH_MIN_SIGMA = 1.0                # floor for std dev
    ASH_FAIR_Z_BETA = 1.0              # fair = micro - beta * z
    ASH_TAKING_Z_THRESHOLD = 1.75      # take only on stronger extremes
    ASH_SIZE_SKEW_STRENGTH = 0.35      # size skew from z
    ASH_MAX_Z_FOR_SKEW = 3.0           # cap z when skewing
    ASH_MR_SIDE_OFF_Z = 2.5            # at very large z, can switch off one side

    # ==========================================================
    # ENTRY POINT
    # ==========================================================
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        trader_state = self._load_state(state.traderData)

        self._trade_pepper(state, result)
        self._trade_ash(state, result, trader_state)

        return result, 0, self._save_state(trader_state)

    # ==========================================================
    # STATE HELPERS
    # ==========================================================
    def _load_state(self, trader_data: str) -> dict:
        default_state = {
            "ash_mu": None,
            "ash_var": None,
            "ash_prev_mid": None,
        }
        if not trader_data:
            return default_state

        try:
            data = json.loads(trader_data)
            for k, v in default_state.items():
                if k not in data:
                    data[k] = v
            return data
        except Exception:
            return default_state

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
    # ASH ONLINE MR HELPERS
    # ==========================================================
    def _update_ash_online_stats(
        self,
        trader_state: dict,
        current_mid: float,
    ) -> tuple[float, float, float]:
        """
        Updates online mean and variance with EMA.
        Returns (mu, var, z).
        """
        alpha = self.ASH_EMA_ALPHA

        mu_prev = trader_state.get("ash_mu", None)
        var_prev = trader_state.get("ash_var", None)

        if mu_prev is None:
            mu = current_mid
            var = 25.0
        else:
            mu = (1.0 - alpha) * mu_prev + alpha * current_mid
            diff = current_mid - mu
            if var_prev is None:
                var_prev = 25.0
            var = (1.0 - alpha) * var_prev + alpha * (diff * diff)

        sigma = max(self.ASH_MIN_SIGMA, math.sqrt(max(var, 0.0)))
        z = (current_mid - mu) / sigma

        trader_state["ash_mu"] = mu
        trader_state["ash_var"] = var

        return mu, var, z

    def _ash_fair_and_z(
        self,
        od: OrderDepth,
        trader_state: dict,
    ) -> tuple[Optional[int], Optional[float], Optional[float]]:
        """
        fair = microprice - beta * z
        Returns (fair, mid, z)
        """
        mid = self._midprice(od)
        micro = self._microprice_l1(od)
        if mid is None or micro is None:
            return None, None, None

        _, _, z = self._update_ash_online_stats(trader_state, mid)
        fair = round(micro - self.ASH_FAIR_Z_BETA * z)
        return fair, mid, z

    def _ash_reversion_side_filter(self, z: float) -> tuple[bool, bool]:
        """
        Very strong positive z => overbought => no bid
        Very strong negative z => oversold => no ask
        """
        place_bid = True
        place_ask = True

        if z >= self.ASH_MR_SIDE_OFF_Z:
            place_bid = False
        elif z <= -self.ASH_MR_SIDE_OFF_Z:
            place_ask = False

        return place_bid, place_ask

    def _ash_size_skew(self, z: float) -> tuple[float, float]:
        """
        Positive z => more ask, less bid
        Negative z => more bid, less ask
        """
        z_capped = max(-self.ASH_MAX_Z_FOR_SKEW, min(self.ASH_MAX_Z_FOR_SKEW, z))
        x = z_capped / self.ASH_MAX_Z_FOR_SKEW

        bid_mult = 1.0 - self.ASH_SIZE_SKEW_STRENGTH * x
        ask_mult = 1.0 + self.ASH_SIZE_SKEW_STRENGTH * x

        bid_mult = max(0.25, bid_mult)
        ask_mult = max(0.25, ask_mult)

        return bid_mult, ask_mult

    def _ash_extreme_taking(
        self,
        product: str,
        od: OrderDepth,
        fair: int,
        z: float,
        buy_cap: int,
        sell_cap: int,
    ) -> tuple[List[Order], int, int]:
        """
        Extra taking only on large mean-reversion extremes.
        z << 0 => buy oversold
        z >> 0 => sell overbought
        """
        orders: List[Order] = []

        if z <= -self.ASH_TAKING_Z_THRESHOLD and buy_cap > 0:
            buy_orders, buy_cap = self._take_asks_below(
                product=product,
                od=od,
                buy_cap=buy_cap,
                max_ask=fair,
            )
            orders.extend(buy_orders)

        elif z >= self.ASH_TAKING_Z_THRESHOLD and sell_cap > 0:
            sell_orders, sell_cap = self._take_bids_above(
                product=product,
                od=od,
                sell_cap=sell_cap,
                min_bid=fair,
            )
            orders.extend(sell_orders)

        return orders, buy_cap, sell_cap

    # ==========================================================
    # GENERIC PRODUCT PIPELINE (for ASH-like assets)
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
        z: Optional[float] = None,
        use_ash_mr_logic: bool = False,
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

            # Additional extreme MR taking for ASH
            if use_ash_mr_logic and z is not None:
                extra_orders, buy_cap, sell_cap = self._ash_extreme_taking(
                    product=product,
                    od=od,
                    fair=fair,
                    z=z,
                    buy_cap=buy_cap,
                    sell_cap=sell_cap,
                )
                orders.extend(extra_orders)

        # ------------------------------------------------------
        # 2) Passive market making
        # ------------------------------------------------------
        if self.ENABLE_MAKING:
            quotes = self._most_competitive_quotes(bb, ba)
            if quotes is not None:
                bid_q, ask_q = quotes

                rr_bid, rr_ask = self._r_ratio_filter(
                    best_bid=bb,
                    best_ask=ba,
                    fair=fair,
                    r_low=mm_r_low,
                    r_high=mm_r_high,
                )

                place_bid = rr_bid
                place_ask = rr_ask

                bid_mult = 1.0
                ask_mult = 1.0

                if use_ash_mr_logic and z is not None:
                    mr_bid, mr_ask = self._ash_reversion_side_filter(z)
                    place_bid = place_bid and mr_bid
                    place_ask = place_ask and mr_ask

                    bid_mult, ask_mult = self._ash_size_skew(z)

                taken_buy = sum(o.quantity for o in orders if o.quantity > 0)
                taken_sell = -sum(o.quantity for o in orders if o.quantity < 0)

                buy_cap_mm = max(0, self._buy_capacity(product, pos) - taken_buy)
                sell_cap_mm = max(0, self._sell_capacity(product, pos) - taken_sell)

                buy_size = min(buy_cap_mm, max(1, round(max_passive_size * bid_mult)))
                sell_size = min(sell_cap_mm, max(1, round(max_passive_size * ask_mult)))

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
    # ASH (ADAPTIVE MEAN REVERSION + MM)
    # ==========================================================
    def _trade_ash(self, state: TradingState, result: Dict[str, List[Order]], trader_state: dict) -> None:
        product = self.ASH
        if product not in state.order_depths:
            return

        od = state.order_depths[product]
        fair, mid, z = self._ash_fair_and_z(od, trader_state)
        if fair is None or mid is None or z is None:
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
            z=z,
            use_ash_mr_logic=True,
        )

        trader_state["ash_prev_mid"] = mid