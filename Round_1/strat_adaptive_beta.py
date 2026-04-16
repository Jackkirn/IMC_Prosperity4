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
    PEPPER_MAX_PASSIVE_SIZE = 20
    PEPPER_AGGRESSIVE_ACCUMULATION = True

    # ==========================================================
    # PARAMETERS — ASH
    # ==========================================================
    ASH_TAKE_THRESHOLD = 0
    ASH_MAX_PASSIVE_SIZE = 20
    ASH_MM_R_LOW = 0.00
    ASH_MM_R_HIGH = 1.00

    # Mean reversion online
    ASH_EMA_ALPHA = 0.08
    ASH_MIN_SIGMA = 1.0
    ASH_FAIR_Z_BETA = 1.15
    ASH_SIZE_SKEW_STRENGTH = 0.35
    ASH_MAX_Z_FOR_SKEW = 3.0

    # --- Online adaptive beta ---
    # Beta: adapted via online regression z -> next return.
    #   beta_opt = -EMA_cov(z, return) / EMA_var(z)
    #   This is the theoretically optimal mean-reversion coefficient,
    #   estimated from the actual data as it arrives.
    # Alpha: left fixed (see comments in code for why).
    ASH_USE_ADAPTIVE_BETA = True
    ASH_BETA_MIN = 0.3            # floor for beta
    ASH_BETA_MAX = 2.5            # ceiling for beta
    # The regression EMA speed is derived from ASH_EMA_ALPHA, not hardcoded.
    # Ratio k=8 means the regression window is ~8x wider than the price EMA,
    # giving ~8 * (1/alpha) ≈ 100 effective observations per beta estimate.
    # This ties the two timescales together: if you change ASH_EMA_ALPHA,
    # the regression window rescales automatically.
    ASH_BETA_REG_K = 8

    # --- Online adaptive z-thresholds (replaces static ASH_TAKING_Z_THRESHOLD / ASH_MR_SIDE_OFF_Z) ---
    # The two thresholds are now online-adaptive estimates of the q-th quantile
    # of |z|. This makes them self-calibrate to the current volatility regime.
    # If you want to fall back to the static behaviour, set ASH_USE_ADAPTIVE_Z = False
    # and the code will use ASH_TAKING_Z_THRESHOLD / ASH_MR_SIDE_OFF_Z.
    ASH_USE_ADAPTIVE_Z = True
    ASH_Q_TAKE = 0.90          # quantile of |z| to trigger aggressive take
    ASH_Q_SIDE = 0.99          # quantile of |z| to turn off one making side
    ASH_LR_TAKE = 0.01         # learning rate for q_take estimator
    ASH_LR_SIDE = 0.005        # learning rate for q_side estimator
    ASH_Q_TAKE_INIT = 1.5      # initial guess (≈ historical q90 of |z|)
    ASH_Q_SIDE_INIT = 2.5      # initial guess (≈ historical q99 of |z|)

    # Static fallback values (only used when ASH_USE_ADAPTIVE_Z = False)
    ASH_TAKING_Z_THRESHOLD = 1.25
    ASH_MR_SIDE_OFF_Z = 2.5

    # Inventory management for ASH only
    ASH_USE_INVENTORY_MANAGEMENT = True
    ASH_INVENTORY_FAIR_GAMMA = 0.00
    ASH_INVENTORY_SIZE_SKEW_STRENGTH = 0.50
    ASH_INVENTORY_SIDE_THRESHOLD_FRAC = 0.70

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
            # Online quantile estimators of |z|
            "ash_q_take_est": self.ASH_Q_TAKE_INIT,
            "ash_q_side_est": self.ASH_Q_SIDE_INIT,
            # Adaptive beta state (online regression z -> return)
            "ash_beta": self.ASH_FAIR_Z_BETA,
            "ash_ema_cov_zr": 0.0,
            "ash_ema_var_z": 1.0,
            "ash_prev_z": None,
            "ash_prev_micro": None,
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
    # GENERAL INVENTORY MANAGEMENT HELPERS
    # ==========================================================
    def _inventory_fair_adjustment(
        self,
        fair: float,
        pos: int,
        gamma: float,
    ) -> float:
        """
        Inventory-aware fair:
            fair_adj = fair - gamma * pos

        If long -> lower fair
        If short -> raise fair
        """
        return fair - gamma * pos

    def _inventory_size_skew(
        self,
        pos: int,
        limit: int,
        base_bid_mult: float = 1.0,
        base_ask_mult: float = 1.0,
        skew_strength: float = 0.5,
    ) -> tuple[float, float]:
        """
        Inventory-aware size skew:
        if long -> smaller bid, larger ask
        if short -> larger bid, smaller ask
        """
        if limit <= 0:
            return base_bid_mult, base_ask_mult

        inv = pos / limit

        bid_mult = base_bid_mult * (1.0 - skew_strength * inv)
        ask_mult = base_ask_mult * (1.0 + skew_strength * inv)

        bid_mult = max(0.0, bid_mult)
        ask_mult = max(0.0, ask_mult)

        return bid_mult, ask_mult

    def _inventory_side_filter(
        self,
        pos: int,
        limit: int,
        threshold_frac: float,
    ) -> tuple[bool, bool]:
        """
        If too long -> no bid
        If too short -> no ask
        """
        if limit <= 0:
            return True, True

        threshold = threshold_frac * limit

        if pos >= threshold:
            return False, True
        if pos <= -threshold:
            return True, False
        return True, True

    # ==========================================================
    # ONLINE QUANTILE TRACKER (for adaptive z thresholds)
    # ==========================================================
    def _update_quantile(self, est: float, x: float, q: float, lr: float) -> float:
        """
        Robbins-Monro stochastic approximation for the q-th quantile.
        Update rule: Δest = lr * (1[x > est] - (1 - q))
        Equivalently:
            if x > est:  est += lr * q
            else:        est -= lr * (1 - q)
        At equilibrium P(x > est) = 1 - q, i.e. est is the q-th quantile.
        """
        if x > est:
            return est + lr * q
        return est - lr * (1.0 - q)

    # ==========================================================
    # ASH ONLINE MR HELPERS
    # ==========================================================
    def _update_online_stats(
        self,
        trader_state: dict,
        key_mu: str,
        key_var: str,
        current_price: float,
        alpha: float,
        min_sigma: float,
        init_var: float = 25.0,
    ) -> tuple[float, float, float]:
        mu_prev = trader_state.get(key_mu, None)
        var_prev = trader_state.get(key_var, None)

        if mu_prev is None:
            mu = current_price
            var = init_var
        else:
            mu = (1.0 - alpha) * mu_prev + alpha * current_price
            diff = current_price - mu
            if var_prev is None:
                var_prev = init_var
            var = (1.0 - alpha) * var_prev + alpha * (diff * diff)

        sigma = max(min_sigma, math.sqrt(max(var, 0.0)))
        z = (current_price - mu) / sigma

        trader_state[key_mu] = mu
        trader_state[key_var] = var

        return mu, sigma, z

    def _ash_fair_and_z(
        self,
        od: OrderDepth,
        trader_state: dict,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:

        mid = self._midprice(od)
        micro = self._microprice_l1(od)
        if mid is None or micro is None:
            return None, None, None

        # --- Resolve current beta (adaptive or static) ---
        if self.ASH_USE_ADAPTIVE_BETA:
            beta = trader_state.get("ash_beta", self.ASH_FAIR_Z_BETA)
        else:
            beta = self.ASH_FAIR_Z_BETA

        # --- Online EMA stats (alpha is fixed) ---
        _, _, z = self._update_online_stats(
            trader_state,
            key_mu="ash_mu",
            key_var="ash_var",
            current_price=micro,
            alpha=self.ASH_EMA_ALPHA,
            min_sigma=self.ASH_MIN_SIGMA,
        )

        # --- Adapt beta for NEXT tick ---
        if self.ASH_USE_ADAPTIVE_BETA:
            self._adapt_beta(trader_state, micro, z)

        # --- Online update of quantile estimates of |z| ---
        if self.ASH_USE_ADAPTIVE_Z:
            az = abs(z)
            trader_state["ash_q_take_est"] = self._update_quantile(
                trader_state.get("ash_q_take_est", self.ASH_Q_TAKE_INIT),
                az, self.ASH_Q_TAKE, self.ASH_LR_TAKE,
            )
            trader_state["ash_q_side_est"] = self._update_quantile(
                trader_state.get("ash_q_side_est", self.ASH_Q_SIDE_INIT),
                az, self.ASH_Q_SIDE, self.ASH_LR_SIDE,
            )

        fair = micro - beta * z
        return fair, mid, z

    # ==========================================================
    # ADAPTIVE BETA
    # ==========================================================
    def _adapt_beta(self, trader_state: dict, micro: float, z: float) -> None:
        """
        Adapt beta via online regression: return_{t} = -beta * z_{t-1} + noise.

        We maintain EMA estimates of cov(z_{t-1}, return_t) and var(z_{t-1}).
        beta = -ema_cov / ema_var_z

        This gives us the theoretically optimal mean-reversion multiplier,
        estimated from the actual data as it arrives.
        """
        prev_z = trader_state.get("ash_prev_z")
        prev_micro = trader_state.get("ash_prev_micro")

        # Save current values for next tick
        trader_state["ash_prev_z"] = z
        trader_state["ash_prev_micro"] = micro

        if prev_z is None or prev_micro is None:
            return

        # Return from previous tick to this tick
        ret = micro - prev_micro

        # Update EMA of cov(z_{t-1}, return_t) and var(z_{t-1})
        # Regression EMA speed is derived from the price EMA speed:
        # alpha_reg = ASH_EMA_ALPHA / ASH_BETA_REG_K
        a = self.ASH_EMA_ALPHA / self.ASH_BETA_REG_K
        ema_cov = trader_state.get("ash_ema_cov_zr", 0.0)
        ema_var_z = trader_state.get("ash_ema_var_z", 1.0)

        ema_cov = (1.0 - a) * ema_cov + a * (prev_z * ret)
        ema_var_z = (1.0 - a) * ema_var_z + a * (prev_z * prev_z)

        trader_state["ash_ema_cov_zr"] = ema_cov
        trader_state["ash_ema_var_z"] = ema_var_z

        # Compute beta = -cov / var (with protection against division by tiny var)
        if ema_var_z > 0.01:
            beta_new = -ema_cov / ema_var_z
            beta_new = max(self.ASH_BETA_MIN, min(self.ASH_BETA_MAX, beta_new))
            trader_state["ash_beta"] = beta_new

    def _ash_get_z_thresholds(self, trader_state: dict) -> tuple[float, float]:
        """Return (taking_z_threshold, mr_side_off_z), adaptive or static."""
        if self.ASH_USE_ADAPTIVE_Z:
            take_z = trader_state.get("ash_q_take_est", self.ASH_Q_TAKE_INIT)
            side_z = trader_state.get("ash_q_side_est", self.ASH_Q_SIDE_INIT)
            return take_z, side_z
        return self.ASH_TAKING_Z_THRESHOLD, self.ASH_MR_SIDE_OFF_Z

    def _ash_reversion_side_filter(self, z: float, side_off_z: float) -> tuple[bool, bool]:
        place_bid = True
        place_ask = True

        if z >= side_off_z:
            place_bid = False
        elif z <= -side_off_z:
            place_ask = False

        return place_bid, place_ask

    def _ash_size_skew(self, z: float) -> tuple[float, float]:
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
        taking_z_threshold: float,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []

        if z <= -taking_z_threshold and buy_cap > 0:
            buy_orders, buy_cap = self._take_asks_below(
                product=product,
                od=od,
                buy_cap=buy_cap,
                max_ask=fair,
            )
            orders.extend(buy_orders)

        elif z >= taking_z_threshold and sell_cap > 0:
            sell_orders, sell_cap = self._take_bids_above(
                product=product,
                od=od,
                sell_cap=sell_cap,
                min_bid=fair,
            )
            orders.extend(sell_orders)

        return orders, buy_cap, sell_cap

    # ==========================================================
    # FAIR — ASH
    # ==========================================================
    def _ash_fair(self, od: OrderDepth) -> Optional[int]:
        mp = self._microprice_l1(od)
        if mp is None:
            return None
        return round(mp)

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
        z: Optional[float] = None,
        use_ash_mr_logic: bool = False,
        use_inventory_management: bool = False,
        inventory_fair_gamma: float = 0.0,
        inventory_size_skew_strength: float = 0.0,
        inventory_side_threshold_frac: float = 1.0,
        ash_taking_z_threshold: float = 1.25,
        ash_mr_side_off_z: float = 2.5,
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

        # ------------------------------------------------------
        # Inventory-adjusted fair
        # ------------------------------------------------------
        fair_for_trading = float(fair)
        if use_inventory_management:
            fair_for_trading = self._inventory_fair_adjustment(
                fair=fair_for_trading,
                pos=pos,
                gamma=inventory_fair_gamma,
            )

        fair_int = round(fair_for_trading)

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
                max_ask=fair_int - take_threshold,
            )
            orders.extend(take_buy_orders)

            take_sell_orders, sell_cap = self._take_bids_above(
                product=product,
                od=od,
                sell_cap=sell_cap,
                min_bid=fair_int + take_threshold,
            )
            orders.extend(take_sell_orders)

            if use_ash_mr_logic and z is not None:
                extra_orders, buy_cap, sell_cap = self._ash_extreme_taking(
                    product=product,
                    od=od,
                    fair=fair_int,
                    z=z,
                    buy_cap=buy_cap,
                    sell_cap=sell_cap,
                    taking_z_threshold=ash_taking_z_threshold,
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
                    fair=fair_int,
                    r_low=mm_r_low,
                    r_high=mm_r_high,
                )

                place_bid = rr_bid
                place_ask = rr_ask

                bid_mult = 1.0
                ask_mult = 1.0

                if use_ash_mr_logic and z is not None:
                    mr_bid, mr_ask = self._ash_reversion_side_filter(z, ash_mr_side_off_z)
                    place_bid = place_bid and mr_bid
                    place_ask = place_ask and mr_ask

                    mr_bid_mult, mr_ask_mult = self._ash_size_skew(z)
                    bid_mult *= mr_bid_mult
                    ask_mult *= mr_ask_mult

                if use_inventory_management:
                    inv_bid_allowed, inv_ask_allowed = self._inventory_side_filter(
                        pos=pos,
                        limit=self.POSITION_LIMITS[product],
                        threshold_frac=inventory_side_threshold_frac,
                    )
                    place_bid = place_bid and inv_bid_allowed
                    place_ask = place_ask and inv_ask_allowed

                    inv_bid_mult, inv_ask_mult = self._inventory_size_skew(
                        pos=pos,
                        limit=self.POSITION_LIMITS[product],
                        base_bid_mult=1.0,
                        base_ask_mult=1.0,
                        skew_strength=inventory_size_skew_strength,
                    )
                    bid_mult *= inv_bid_mult
                    ask_mult *= inv_ask_mult

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
    # ASH (ADAPTIVE MEAN REVERSION + MM + INVENTORY MGMT)
    # ==========================================================
    def _trade_ash(self, state: TradingState, result: Dict[str, List[Order]], trader_state: dict) -> None:
        product = self.ASH
        if product not in state.order_depths:
            return

        od = state.order_depths[product]
        fair_float, mid, z = self._ash_fair_and_z(od, trader_state)
        if fair_float is None or mid is None or z is None:
            return

        # Get adaptive (or static fallback) z thresholds
        take_z, side_z = self._ash_get_z_thresholds(trader_state)

        self._trade_product(
            state=state,
            result=result,
            product=product,
            fair=round(fair_float),
            take_threshold=self.ASH_TAKE_THRESHOLD,
            max_passive_size=self.ASH_MAX_PASSIVE_SIZE,
            mm_r_low=self.ASH_MM_R_LOW,
            mm_r_high=self.ASH_MM_R_HIGH,
            z=z,
            use_ash_mr_logic=True,
            use_inventory_management=self.ASH_USE_INVENTORY_MANAGEMENT,
            inventory_fair_gamma=self.ASH_INVENTORY_FAIR_GAMMA,
            inventory_size_skew_strength=self.ASH_INVENTORY_SIZE_SKEW_STRENGTH,
            inventory_side_threshold_frac=self.ASH_INVENTORY_SIDE_THRESHOLD_FRAC,
            ash_taking_z_threshold=take_z,
            ash_mr_side_off_z=side_z,
        )

        trader_state["ash_prev_mid"] = mid