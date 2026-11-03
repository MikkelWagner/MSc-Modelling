"""Run with python -m unittest hydesign.tests.test_cost_simple_datacenter -v."""

from pathlib import Path
import unittest

import numpy as np
import openmdao.api as om
import yaml

from hydesign.costs.cost_simple_datacenter import (
    datacenter_cost,
    datacenter_cost_comp,
    datacenter_cost_annualized,
    datacenter_cost_annualized_comp,
)


BASE_PARAMETERS = {
    "datacenter_facility_capex_per_mw_it": 9_600_000,
    "datacenter_it_capex_per_mw_it": 15_330_000,
    "datacenter_fixed_opex_per_mw_it_year": 250_000,
}
ANNUAL_PARAMETERS = {
    **BASE_PARAMETERS,
    "datacenter_discount_rate": 0.05,  # Synthetic test rate, not a thesis assumption.
    "datacenter_facility_lifetime_years": 20,
    "datacenter_it_lifetime_years": 5,
}
OUTPUT_NAMES = (
    "CAPEX_dc_facility", "CAPEX_dc_it", "CAPEX_dc_total", "OPEX_dc",
    "annualized_CAPEX_dc_facility", "annualized_CAPEX_dc_it",
    "annualized_CAPEX_dc_total", "annualized_cost_dc",
)


class DatacenterCostTests(unittest.TestCase):
    def test_base_cost_arithmetic(self):
        values = datacenter_cost(**BASE_PARAMETERS).compute(it_capacity_mw=2)
        self.assertEqual(values, (19_200_000, 30_660_000, 49_860_000, 500_000))
        self.assertEqual(values[2], values[0] + values[1])

    def test_zero_capacity(self):
        for model in (datacenter_cost(**BASE_PARAMETERS), datacenter_cost_annualized(**ANNUAL_PARAMETERS)):
            with self.subTest(model=type(model).__name__):
                np.testing.assert_array_equal(model.compute(0), np.zeros(len(model.compute(1))))

    def test_linear_scaling_and_scalar_array(self):
        for model in (datacenter_cost(**BASE_PARAMETERS), datacenter_cost_annualized(**ANNUAL_PARAMETERS)):
            with self.subTest(model=type(model).__name__):
                np.testing.assert_allclose(model.compute(4), 2 * np.asarray(model.compute(2)))
                np.testing.assert_allclose(model.compute(np.array([2.0])), model.compute(2))

    def test_separate_lifetimes_and_annual_cost_totals(self):
        values = datacenter_cost_annualized(**ANNUAL_PARAMETERS).compute(2)
        rate = ANNUAL_PARAMETERS["datacenter_discount_rate"]
        crf_facility = rate * (1 + rate)**20 / ((1 + rate)**20 - 1)
        crf_it = rate * (1 + rate)**5 / ((1 + rate)**5 - 1)
        np.testing.assert_allclose(values[4:6], [values[0] * crf_facility, values[1] * crf_it])
        self.assertGreater(values[5] / values[1], values[4] / values[0])
        self.assertEqual(values[6], values[4] + values[5])
        self.assertEqual(values[7], values[6] + values[3])

    def test_annualized_and_base_raw_costs_match(self):
        base = datacenter_cost(**BASE_PARAMETERS).compute(2)
        annualized = datacenter_cost_annualized(**ANNUAL_PARAMETERS).compute(2)
        self.assertEqual(base, annualized[:4])

    def test_zero_discount_rate(self):
        model = datacenter_cost_annualized(**{**ANNUAL_PARAMETERS, "datacenter_discount_rate": 0})
        values = model.compute(1)
        np.testing.assert_allclose(values[4:6], [9_600_000 / 20, 15_330_000 / 5])
        self.assertAlmostEqual(values[7], 3_796_000)

    def test_near_zero_discount_rate(self):
        model = datacenter_cost_annualized(**{**ANNUAL_PARAMETERS, "datacenter_discount_rate": 1e-14})
        values = model.compute(1)
        np.testing.assert_allclose(values[4:6], [9_600_000 / 20, 15_330_000 / 5], rtol=1e-12)

    def test_invalid_cost_coefficients(self):
        for name in BASE_PARAMETERS:
            for value in (-1, np.nan, np.inf, -np.inf, None):
                for model_class, parameters in (
                    (datacenter_cost, BASE_PARAMETERS),
                    (datacenter_cost_annualized, ANNUAL_PARAMETERS),
                ):
                    with self.subTest(name=name, value=value, model=model_class.__name__):
                        with self.assertRaisesRegex(ValueError, name):
                            model_class(**{**parameters, name: value})

    def test_invalid_it_capacity(self):
        for value in (-1, np.nan, np.inf, -np.inf, None, [], [1, 2]):
            for model in (datacenter_cost(**BASE_PARAMETERS), datacenter_cost_annualized(**ANNUAL_PARAMETERS)):
                with self.subTest(value=value, model=type(model).__name__):
                    with self.assertRaisesRegex(ValueError, "it_capacity_mw"):
                        model.compute(value)

    def test_invalid_discount_rate(self):
        for value in (-1, np.nan, np.inf, -np.inf, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "datacenter_discount_rate"):
                    datacenter_cost_annualized(**{**ANNUAL_PARAMETERS, "datacenter_discount_rate": value})

    def test_invalid_lifetimes(self):
        for name in ("datacenter_facility_lifetime_years", "datacenter_it_lifetime_years"):
            for value in (0, -1, np.nan, np.inf, -np.inf, None):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        datacenter_cost_annualized(**{**ANNUAL_PARAMETERS, name: value})

    def test_missing_required_parameters(self):
        with self.assertRaisesRegex(TypeError, "datacenter_it_capex_per_mw_it"):
            datacenter_cost(datacenter_facility_capex_per_mw_it=1, datacenter_fixed_opex_per_mw_it_year=1)
        with self.assertRaisesRegex(TypeError, "datacenter_discount_rate"):
            datacenter_cost_annualized(**{
                name: value for name, value in ANNUAL_PARAMETERS.items()
                if name != "datacenter_discount_rate"
            })

    def test_wrappers_match_plain_models(self):
        for model_class, wrapper_class, parameters in (
            (datacenter_cost, datacenter_cost_comp, BASE_PARAMETERS),
            (datacenter_cost_annualized, datacenter_cost_annualized_comp, ANNUAL_PARAMETERS),
        ):
            with self.subTest(wrapper=wrapper_class.__name__):
                expected = model_class(**parameters).compute(2)
                names = OUTPUT_NAMES[:len(expected)]
                prob = om.Problem(reports=False)
                try:
                    prob.model.add_subsystem("cost", wrapper_class(**parameters), promotes=["*"])
                    prob.setup()
                    # Check unit conversion and the OpenMDAO scalar-array input.
                    prob.set_val("it_capacity_mw", 2000, units="kW")
                    prob.run_model()
                    np.testing.assert_allclose([prob.get_val(name).item() for name in names], expected)
                    totals = prob.compute_totals(of=list(names), wrt=["it_capacity_mw"], return_format="array")
                    np.testing.assert_allclose(totals[:, 0], np.asarray(expected) / 2, rtol=1e-7)
                finally:
                    prob.cleanup()

    def test_shared_yaml_cost_assumptions_and_missing_rate_rejection(self):
        path = Path(__file__).resolve().parents[1] / "examples/thesis_simple_datacenter/simple_datacenter.yml"
        with path.open() as stream:
            parameters = yaml.safe_load(stream)
        for name, expected in BASE_PARAMETERS.items():
            self.assertEqual(parameters[name], expected)
        self.assertEqual(parameters["datacenter_facility_lifetime_years"], 20)
        self.assertEqual(parameters["datacenter_it_lifetime_years"], 5)
        self.assertEqual(parameters["datacenter_discount_rate"], 0.075)
        base = {name: parameters[name] for name in BASE_PARAMETERS}
        self.assertEqual(datacenter_cost(**base).compute(1)[2], 24_930_000)
        annual = {name: parameters[name] for name in ANNUAL_PARAMETERS}
        self.assertTrue(np.isfinite(datacenter_cost_annualized(**annual).compute(1)).all())
        annual["datacenter_discount_rate"] = None
        with self.assertRaisesRegex(ValueError, "datacenter_discount_rate"):
            datacenter_cost_annualized(**annual)


if __name__ == "__main__":
    unittest.main()
