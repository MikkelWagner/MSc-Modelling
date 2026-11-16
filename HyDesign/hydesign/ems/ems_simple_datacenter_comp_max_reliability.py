"""OpenMDAO interface to the unchanged Phase 1 availability EMS."""
import numpy as np
from hydesign.openmdao_wrapper import ComponentWrapper
from hydesign.ems.ems_simple_datacenter_max_reliability import ems_cplex_simple_datacenter_max_reliability


class ems_simple_datacenter_max_reliability:
    def __init__(self, N_time, ems_type='cplex', time_limit_seconds=30, threads=1,
                 beta=.25, kappa=.10, epsilon_mw=.001, mip_relative_gap=1e-4,
                 random_seed=0):
        if ems_type != 'cplex':
            raise ValueError('Only CPLEX is supported')
        self.N_time = int(N_time)
        self.options = dict(time_limit_seconds=time_limit_seconds, threads=threads,
                            beta=beta, kappa=kappa, epsilon_mw=epsilon_mw,
                            mip_relative_gap=mip_relative_gap, random_seed=random_seed)
        self.solver_status = None

    def compute(self, wind_t, solar_t, b_P, b_E, battery_depth_of_discharge,
                battery_charge_efficiency, battery_initial_soc, facility_load_mw, pue):
        self.solver_status = None
        if np.shape(wind_t) != (self.N_time,) or np.shape(solar_t) != (self.N_time,):
            raise ValueError('Generation arrays must have N_time entries')
        values = [np.asarray(v).item() for v in (b_P, b_E, battery_depth_of_discharge,
            battery_charge_efficiency, battery_initial_soc, facility_load_mw, pue)]
        result = ems_cplex_simple_datacenter_max_reliability(wind_t, solar_t, *values, **self.options)
        self.solver_status = result[-1]
        return result[:-1]


class ems_simple_datacenter_comp_max_reliability(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = ems_simple_datacenter_max_reliability(**insta_inp)
        self.ems_model = model
        super().__init__(
            inputs=[('wind_t', {'units':'MW', 'shape':[model.N_time]}),
                    ('solar_t', {'units':'MW', 'shape':[model.N_time]}),
                    ('b_P', {'units':'MW'}), ('b_E', {'units':'MW*h'}),
                    ('battery_depth_of_discharge', {}), ('battery_charge_efficiency', {}),
                    ('battery_initial_soc', {}), ('facility_load_mw', {'units':'MW'}), ('pue', {})],
            outputs=[(name, {'units': 'MW*h' if name == 'b_E_SOC_t' else 'MW',
                             'shape': [model.N_time + (name == 'b_E_SOC_t')]})
                     for name in ('hpp_t', 'hpp_curt_t', 'b_t', 'b_E_SOC_t', 'charge_mw',
                                  'discharge_mw', 'served_firm_it_mw', 'unserved_firm_it_mw',
                                  'unserved_facility_mw')],
            function=model.compute, partial_options=[{'dependent':False, 'val':0}])
