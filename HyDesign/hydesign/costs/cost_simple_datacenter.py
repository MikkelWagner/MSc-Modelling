"""Standalone aggregate data-center costs, expressed per installed MW_IT."""

import numpy as np

from hydesign.openmdao_wrapper import ComponentWrapper


class datacenter_cost:
    """Initial facility and IT CAPEX, and fixed annual OPEX.

    The coefficients represent aggregate costs; no subcomponent breakdown,
    energy consumption or replacement schedule is calculated here.
    """

    def __init__(
        self,
        datacenter_facility_capex_per_mw_it,
        datacenter_it_capex_per_mw_it,
        datacenter_fixed_opex_per_mw_it_year,
    ):
        """Store facility/IT CAPEX [EUR/MW_IT] and OPEX [EUR/MW_IT/year]."""
        self.datacenter_facility_capex_per_mw_it = _validate_scalar(
            datacenter_facility_capex_per_mw_it, "datacenter_facility_capex_per_mw_it"
        )
        self.datacenter_it_capex_per_mw_it = _validate_scalar(
            datacenter_it_capex_per_mw_it, "datacenter_it_capex_per_mw_it"
        )
        self.datacenter_fixed_opex_per_mw_it_year = _validate_scalar(
            datacenter_fixed_opex_per_mw_it_year, "datacenter_fixed_opex_per_mw_it_year"
        )

    def compute(self, it_capacity_mw, **kwargs):
        """Calculate installed costs from IT electrical capacity [MW_IT].

        Returns
        -------
        CAPEX_dc_facility, CAPEX_dc_it, CAPEX_dc_total : Initial CAPEX [EUR].
        OPEX_dc : Fixed annual operating cost [EUR/year].
        """
        it_capacity_mw = _validate_scalar(it_capacity_mw, "it_capacity_mw")
        CAPEX_dc_facility = self.datacenter_facility_capex_per_mw_it * it_capacity_mw
        CAPEX_dc_it = self.datacenter_it_capex_per_mw_it * it_capacity_mw
        CAPEX_dc_total = CAPEX_dc_facility + CAPEX_dc_it
        OPEX_dc = self.datacenter_fixed_opex_per_mw_it_year * it_capacity_mw
        return CAPEX_dc_facility, CAPEX_dc_it, CAPEX_dc_total, OPEX_dc


class datacenter_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = datacenter_cost(**insta_inp)
        super().__init__(
            inputs=[
                (
                    "it_capacity_mw",
                    {"units": "MW", "desc": "Installed IT electrical capacity [MW_IT]"},
                ),
            ],
            outputs=[
                ("CAPEX_dc_facility", {"desc": "Initial data-center facility CAPEX [EUR]"}),
                ("CAPEX_dc_it", {"desc": "Initial data-center IT CAPEX [EUR]"}),
                ("CAPEX_dc_total", {"desc": "Total initial data-center CAPEX [EUR]"}),
                ("OPEX_dc", {"desc": "Fixed annual data-center OPEX [EUR/year]"}),
            ],
            function=model.compute,
        )


class datacenter_cost_annualized:
    """Reuse installed costs and annualize facility and IT CAPEX separately.

    The IT economic lifetime represents regular server/IT replacement through
    capital recovery. No explicit replacement events or lifetime NPV are modeled.
    """

    def __init__(
        self,
        datacenter_facility_capex_per_mw_it,
        datacenter_it_capex_per_mw_it,
        datacenter_fixed_opex_per_mw_it_year,
        datacenter_discount_rate,
        datacenter_facility_lifetime_years,
        datacenter_it_lifetime_years,
    ):
        """Use base cost coefficients, an annual rate [-] and lifetimes [years].

        All parameters are required. Economic lifetimes are independent of the
        number of weather years simulated by the operational model.
        """
        self.base_cost = datacenter_cost(
            datacenter_facility_capex_per_mw_it,
            datacenter_it_capex_per_mw_it,
            datacenter_fixed_opex_per_mw_it_year,
        )
        self.datacenter_discount_rate = _validate_scalar(
            datacenter_discount_rate, "datacenter_discount_rate"
        )
        self.datacenter_facility_lifetime_years = _validate_scalar(
            datacenter_facility_lifetime_years,
            "datacenter_facility_lifetime_years",
            positive=True,
        )
        self.datacenter_it_lifetime_years = _validate_scalar(
            datacenter_it_lifetime_years, "datacenter_it_lifetime_years", positive=True
        )
        self.CRF_facility = _capital_recovery_factor(
            self.datacenter_discount_rate, self.datacenter_facility_lifetime_years
        )
        self.CRF_it = _capital_recovery_factor(
            self.datacenter_discount_rate, self.datacenter_it_lifetime_years
        )

    def compute(self, it_capacity_mw, **kwargs):
        """Return four base costs followed by four annualized costs [EUR/year].

        The annualized outputs are facility CAPEX, IT CAPEX, total CAPEX and
        total cost including fixed annual OPEX, in that order.
        """
        CAPEX_dc_facility, CAPEX_dc_it, CAPEX_dc_total, OPEX_dc = self.base_cost.compute(
            it_capacity_mw
        )
        annualized_CAPEX_dc_facility = self.CRF_facility * CAPEX_dc_facility
        annualized_CAPEX_dc_it = self.CRF_it * CAPEX_dc_it
        annualized_CAPEX_dc_total = annualized_CAPEX_dc_facility + annualized_CAPEX_dc_it
        annualized_cost_dc = annualized_CAPEX_dc_total + OPEX_dc
        return (
            CAPEX_dc_facility,
            CAPEX_dc_it,
            CAPEX_dc_total,
            OPEX_dc,
            annualized_CAPEX_dc_facility,
            annualized_CAPEX_dc_it,
            annualized_CAPEX_dc_total,
            annualized_cost_dc,
        )


class datacenter_cost_annualized_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = datacenter_cost_annualized(**insta_inp)
        super().__init__(
            inputs=[
                (
                    "it_capacity_mw",
                    {"units": "MW", "desc": "Installed IT electrical capacity [MW_IT]"},
                ),
            ],
            outputs=[
                ("CAPEX_dc_facility", {"desc": "Initial data-center facility CAPEX [EUR]"}),
                ("CAPEX_dc_it", {"desc": "Initial data-center IT CAPEX [EUR]"}),
                ("CAPEX_dc_total", {"desc": "Total initial data-center CAPEX [EUR]"}),
                ("OPEX_dc", {"desc": "Fixed annual data-center OPEX [EUR/year]"}),
                (
                    "annualized_CAPEX_dc_facility",
                    {"desc": "Annualized data-center facility CAPEX [EUR/year]"},
                ),
                (
                    "annualized_CAPEX_dc_it",
                    {"desc": "Annualized data-center IT CAPEX [EUR/year]"},
                ),
                (
                    "annualized_CAPEX_dc_total",
                    {"desc": "Total annualized data-center CAPEX [EUR/year]"},
                ),
                (
                    "annualized_cost_dc",
                    {"desc": "Annualized data-center CAPEX plus fixed OPEX [EUR/year]"},
                ),
            ],
            function=model.compute,
        )


def _validate_scalar(value, name, positive=False):
    """Accept a number or OpenMDAO scalar array; reject missing/invalid inputs."""
    requirement = "positive" if positive else "nonnegative"
    message = f"{name} must be a finite {requirement} scalar; provide an explicit value"
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if value.size != 1:
        raise ValueError(message)
    value = value.item()
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(message)
    return value


def _capital_recovery_factor(rate, lifetime):
    """CRF = r*(1+r)**n / ((1+r)**n - 1), with its zero-rate limit."""
    if rate == 0:
        return 1.0 / lifetime
    # Equivalent r / (1 - (1+r)**(-n)); stable for small positive rates.
    return rate / -np.expm1(-lifetime * np.log1p(rate))
