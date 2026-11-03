"""Annual equivalents of native asset and shared infrastructure costs.

Asset cost equations remain in hydesign.costs.costs. Economic recovery periods
are independent of the representative-year simulation. One scheduled battery
replacement is discounted before annualization; initial CAPEX is unchanged.
No physical degradation, price decline, salvage or decommissioning is modelled.
Shared BOS and land CAPEX come from the unchanged native shared_cost component.
"""

from hydesign.costs.cost_simple_datacenter import (
    _capital_recovery_factor, _validate_scalar,
)
from hydesign.openmdao_wrapper import ComponentWrapper


RAW_HPP_COST_NAMES = ("CAPEX_w", "OPEX_w", "CAPEX_s", "OPEX_s", "CAPEX_b", "OPEX_b",
                      "CAPEX_sh", "OPEX_sh")
SHARED_REPORT_NAMES = ("G_proxy_MW", "CAPEX_sh_fixed", "CAPEX_sh_land")
BATTERY_REPLACEMENT_COST_NAMES = ("battery_replacement_cost", "discounted_battery_replacement_cost")
ANNUAL_HPP_COST_NAMES = (
    "annualized_CAPEX_wind", "annualized_cost_wind",
    "annualized_CAPEX_pv", "annualized_cost_pv",
    "annualized_CAPEX_battery", "annualized_cost_battery",
    "annualized_CAPEX_shared", "annualized_cost_shared",
    *BATTERY_REPLACEMENT_COST_NAMES,
    "CAPEX_hpp", "OPEX_hpp", "annualized_CAPEX_hpp", "annualized_cost_hpp",
)
HPP_COST_NAMES = RAW_HPP_COST_NAMES + SHARED_REPORT_NAMES + ANNUAL_HPP_COST_NAMES


class hpp_cost_annualized:
    """Annualize native CAPEX [EUR] and add native OPEX [EUR/year]."""

    def __init__(self, wind_WACC, solar_WACC, battery_WACC, hpp_economic_lifetime_years,
                 battery_replacement_year=15, battery_replacement_cost_fraction=1.0):
        lifetime = _validate_scalar(
            hpp_economic_lifetime_years, "hpp_economic_lifetime_years", positive=True,
        )
        self.CRF_wind = _capital_recovery_factor(_validate_scalar(wind_WACC, "wind_WACC"), lifetime)
        self.CRF_pv = _capital_recovery_factor(_validate_scalar(solar_WACC, "solar_WACC"), lifetime)
        self.CRF_battery = _capital_recovery_factor(_validate_scalar(battery_WACC, "battery_WACC"), lifetime)
        self.battery_WACC = _validate_scalar(battery_WACC, "battery_WACC")
        self.battery_replacement_year = _validate_scalar(
            battery_replacement_year, "battery_replacement_year", positive=True)
        if self.battery_replacement_year != int(self.battery_replacement_year) or self.battery_replacement_year >= lifetime:
            raise ValueError("One battery replacement must occur in an integer year strictly within the HPP economic lifetime")
        self.battery_replacement_cost_fraction = _validate_scalar(
            battery_replacement_cost_fraction, "battery_replacement_cost_fraction", positive=True)
        # Native HyDesign uses the mean asset WACC for shared infrastructure.
        # The thesis rates are all 6%, with a 25-year economic lifetime.
        self.CRF_shared = _capital_recovery_factor((wind_WACC + solar_WACC + battery_WACC) / 3, lifetime)

    def compute(self, CAPEX_w, OPEX_w, CAPEX_s, OPEX_s, CAPEX_b, OPEX_b,
                CAPEX_sh=0, OPEX_sh=0, **kwargs):
        CAPEX_w, OPEX_w, CAPEX_s, OPEX_s, CAPEX_b, OPEX_b, CAPEX_sh, OPEX_sh = (
            _validate_scalar(value, name) for name, value in zip(
                RAW_HPP_COST_NAMES, (CAPEX_w, OPEX_w, CAPEX_s, OPEX_s, CAPEX_b, OPEX_b, CAPEX_sh, OPEX_sh),
            )
        )
        annualized_CAPEX_wind = CAPEX_w * self.CRF_wind
        annualized_CAPEX_pv = CAPEX_s * self.CRF_pv
        battery_replacement_cost = CAPEX_b * self.battery_replacement_cost_fraction
        discounted_battery_replacement_cost = battery_replacement_cost / (
            (1 + self.battery_WACC) ** self.battery_replacement_year)
        annualized_CAPEX_battery = (CAPEX_b + discounted_battery_replacement_cost) * self.CRF_battery
        annualized_CAPEX_shared = CAPEX_sh * self.CRF_shared
        annualized_CAPEX_hpp = (annualized_CAPEX_wind + annualized_CAPEX_pv
                               + annualized_CAPEX_battery + annualized_CAPEX_shared)
        OPEX_hpp = OPEX_w + OPEX_s + OPEX_b + OPEX_sh
        return (
            annualized_CAPEX_wind, annualized_CAPEX_wind + OPEX_w,
            annualized_CAPEX_pv, annualized_CAPEX_pv + OPEX_s,
            annualized_CAPEX_battery, annualized_CAPEX_battery + OPEX_b,
            annualized_CAPEX_shared, annualized_CAPEX_shared + OPEX_sh,
            battery_replacement_cost, discounted_battery_replacement_cost,
            CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_sh, OPEX_hpp,
            annualized_CAPEX_hpp, annualized_CAPEX_hpp + OPEX_hpp,
        )


class hpp_cost_annualized_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = hpp_cost_annualized(**insta_inp)
        super().__init__(
            inputs=[(name, {"desc": name + (" [EUR]" if name.startswith("CAPEX_") else " [EUR/year]")})
                    for name in RAW_HPP_COST_NAMES],
            outputs=[(name, {"desc": name + (" [EUR]" if name == "CAPEX_hpp" or
                                                          name in BATTERY_REPLACEMENT_COST_NAMES else " [EUR/year]")})
                     for name in ANNUAL_HPP_COST_NAMES],
            function=model.compute,
        )
