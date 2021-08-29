"jupytext_formats: ipynb,py"
import sys
import numpy as np
import ipdb  # noqa
import seaborn
sys.path.append('./src/build/lib.macosx-10.7-x86_64-3.7')
seaborn.set()
from simulate import runsim  # noqa
#, update_sim_param, display_sim_results, check_balance_conditions, mean_field_rates, restore_default_params  # noqa

# ####


def sim_loop_ove_Jei(t_stop=1000.0):
    # Jei_list = np.arange(-0.5, -2.6, -0.5)
    Jei_list = np.arange(-2.5, -1.4, 0.5)
    for Jei in Jei_list:
        print(f'Jei = {Jei}')
        runsim(Jei=Jei,
               STDP_ON=0,
               STP_ON=0,
               t_stop=t_stop,
               discard_time=100.0,
               NE=1000,
               NI=1000,
               K=100,
               v_ext=0.0052)


sim_loop_ove_Jei()

import simulate

# ## Square wave FF input

simulate.runsim(STDP_ON=0, STP_ON=0, t_stop=1000.0, discard_time=100.0)

import simulate

simulate.runsim(STDP_ON=0, STP_ON=0, t_stop=1000.0, discard_time=100.0, v_ext=0.0125)


