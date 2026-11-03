"""Fixed-HPP cost reporting around the unchanged strengthened DC-only sizer."""

from hydesign.costs.cost_simple_hpp_with_hpp_cost import HPP_COST_NAMES, hpp_cost_annualized, ANNUAL_HPP_COST_NAMES
from hydesign.costs.costs import shared_cost
from hydesign.sizing_simple_datacenter_with_dc_cost_strengthened import (
    size_datacenter_with_dc_cost,
)


def size_datacenter_with_hpp_cost(hpp, **sizing_options):
    """Use generation, ratings and costs from one successful assembly evaluation.

Pass the existing DC/technical/solver options unchanged. Generation and battery
ratings cannot be overridden separately: their costs belong to this fixed HPP.
Shared infrastructure is recalculated at the CHOSEN facility load. This remains
a DC-only sizing objective, not integrated system-value optimization.
No HPP cost or constant objective offset is passed to CPLEX. Solver gaps and
bounds describe the DC-only objective including any artificial fluctuation penalty.
Reported net annual DC/system values exclude that penalty.
"""
    if getattr(hpp, "economics", None) is None or getattr(hpp, "hourly", None) is None:
        raise ValueError("Run the with_hpp_cost assembly evaluate() successfully before sizing")
    reserved = {"wind_ts", "solar_ts", "b_P", "b_E"}.intersection(sizing_options)
    if reserved:
        raise ValueError("Generation and battery ratings must come from hpp: " + ", ".join(sorted(reserved)))
    hpp_costs = {name: hpp.economics[name] for name in HPP_COST_NAMES}
    b_P, b_E_h = hpp.inputs[-2:]
    result = size_datacenter_with_dc_cost(
        wind_ts=hpp.hourly["wind_mw"].copy(),
        solar_ts=hpp.hourly["pv_mw"].copy(),
        b_P=b_P, b_E=b_P * b_E_h,
        **sizing_options,
    )
    economics = dict(result["economics"])
    # The legacy inner MILP chooses a new load; shared HPP costs are no longer
    # constant. Recompute reporting at that load without changing its objective.
    cfg = hpp.sim_pars
    proxy = result["facility_load_mw"] + b_P
    csh, osh, fixed, wind_land, pv_land = shared_cost(
        cfg["hpp_BOS_soft_cost"], cfg["hpp_grid_connection_cost"], cfg["land_cost"],
    ).compute(proxy, hpp.prob.get_val("Awpp").item(), hpp.prob.get_val("Apvp").item())
    hpp_costs.update(G_proxy_MW=proxy, CAPEX_sh=csh, OPEX_sh=osh,
                     CAPEX_sh_fixed=fixed, CAPEX_sh_land=max(wind_land, pv_land))
    annual = hpp_cost_annualized(
        cfg["wind_WACC"], cfg["solar_WACC"], cfg["battery_WACC"], cfg["hpp_economic_lifetime_years"],
        battery_replacement_year=cfg["battery_replacement_year"],
        battery_replacement_cost_fraction=cfg["battery_replacement_cost_fraction"])
    hpp_costs.update(zip(ANNUAL_HPP_COST_NAMES, annual.compute(
        *[hpp_costs[k] for k in ("CAPEX_w", "OPEX_w", "CAPEX_s", "OPEX_s",
                                "CAPEX_b", "OPEX_b", "CAPEX_sh", "OPEX_sh")])))
    economics.update(hpp_costs)
    economics["net_annual_value_dc_only"] = economics["net_annual_value"]
    economics["net_annual_value_system"] = economics["net_annual_value"] - economics["annualized_cost_hpp"]
    return {**result, "economics": economics}
