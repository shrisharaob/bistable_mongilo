"jupytext_formats: ipynb,py"

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ipdb  # noqa
import argparse
import toml
from configobj import ConfigObj
from scipy.signal import square as square_wave
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn
import glob
import platform
if platform.platform().find('Linux') >= 0:
    os_type = 'linux'
else:
    os_type = 'macosx'
build_files = glob.glob('./src/cpp/build/*')
shared_lib = ''
print(build_files)
for build_file_id, f in enumerate(build_files):
    if f.find(f'lib.{os_type}') > 0:
        shared_lib = f
print(f'loading shared lib from {shared_lib}')
sys.path.append(shared_lib)

import matplotlib  # noqa
matplotlib.use('Agg')
plt.ioff()

# sys.path.append('/Users/shrisha/code/lif/src/build/lib.macosx-10.7-x86_64-3.7')
import stdp  # noqa

# set src paths
home_dir = Path.home()
script_path = str(Path(__file__).resolve().parent)
pyconf = toml.load(script_path + '/config.toml')
rbm_analysis_scr = home_dir / pyconf['rbm_analysis']['rbm_analysis_src']
sys.path.append(str(rbm_analysis_scr))
sys.path.append(str(rbm_analysis_scr / 'helpers'))
import analyse_dataset as ad  # noqa
import plot_utils as pu  # noqa
import matrix_funcs  # noqa

seaborn.set()


def runsim(**params):

    # load default params
    # conf = ConfigObj(script_path + '/src/params_bkp.cfg')
    # conf.filename = 'params.cfg'
    # conf.write()
    # make changes in parameters if provided
    update_sim_param(**params)
    new_conf = ConfigObj('params.cfg')

    # ipdb.set_trace()
    # generate_ff_input_file(new_conf)

    # run the simulations
    stdp.simu()

    # # display the simulation results
    # display_sim_results(new_conf)

    # if (int(new_conf['scaling_type']) == 0):
    #     check_balance_conditions(new_conf)
    #     mean_field_rates(new_conf)
    # #
    # restore_default_params()


def runsim_brunel(**params):

    # load default params
    conf = ConfigObj('../src/params_brunel.cfg')
    conf.filename = 'params.cfg'
    conf.write()

    # make changes in parameters if provided
    update_sim_param(**params)
    new_conf = ConfigObj('params.cfg')
    # generate_ff_input_file(new_conf)

    # run the simulations
    stdp.simu()

    # display the simulation results
    display_sim_results(new_conf)


def generate_ff_input_file(conf, duty_val=1.0):
    dt = float(conf['dt'])
    t_stop = float(conf['t_stop'])
    discard_time = float(conf['discard_time'])
    print(t_stop)
    n_steps = int((t_stop) / dt)
    n_discard_steps = int((discard_time) / dt)
    ff_input_zeros = np.zeros((n_discard_steps, ))
    t = np.linspace(0, t_stop, n_steps)
    duty_cycle = np.zeros_like(t)
    idx = np.random.randint(10, duty_cycle.size, 10)
    idx_start = idx - 100
    for i in range(idx.size):
        duty_cycle[idx_start[i]:idx[i]] = duty_val
    #
    ff_input = 4.0 * np.heaviside(square_wave(t, duty_cycle), 0.0)

    ff_input = np.hstack((ff_input_zeros, ff_input))

    #
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[:, :-1])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, 2])
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    ax2.set_ylabel(' ')
    ax1.set_ylabel(' ')
    fig.set_size_inches(20, 4.8)

    ax0.plot(t * 1e-3, ff_input[n_discard_steps:])

    np.savetxt('./data/ff_input.txt', ff_input, delimiter="\n")


def runsim_paramfile(paramfile):

    # load default params
    conf = ConfigObj(paramfile)
    conf.filename = 'params.cfg'
    conf.write()

    #
    stdp.simu()

    display_sim_results(conf)

    if (int(conf['scaling_type']) == 0):
        check_balance_conditions(conf)
        mean_field_rates(conf)
    #
    # restore_default_params()


def restore_default_params():
    conf = ConfigObj(script_path + 'src/params_bkp.cfg')
    conf.filename = 'params.cfg'
    conf.write()


def slow_aux(dt=0.1, smoothin_window=3):
    st = np.loadtxt('data/spikes.txt')
    st[:, 0] *= 1e3
    t_min = 0  # st[:, 0].min()
    t_max = st[:, 0].max()
    # idx = np.logical_and(st[:, 0] > t_min, st[:, 0] <= t_max)
    bins = np.arange(t_min, t_max + dt, dt)
    cnts, bins = np.histogram(st[:, 0], bins)
    acfunc = auto_corr
    smth_cnts = matrix_funcs.smooth_signal(cnts, smoothin_window)
    ac_re = acfunc(smth_cnts)
    # max_lag = 100  # ms  ac_re.size
    lags = bins[:ac_re.size]
    return lags, ac_re


def recip_vs_ac(rel_in_strengths=[0.0, 0.5, 0.75, 1.0], smoothin_window=11):
    ac = {}
    for p in rel_in_strengths:
        print(f'p={p}')
        runsim(con_symmetry=p,
               g_brunel=1.0,
               t_stop=2000.0,
               discard_time=250.0,
               STDP_ON=0,
               STP_ON=0)
        lags, ac_re = slow_aux(smoothin_window=11)
        ac[f'p{p}'] = (lags, ac_re)
        plt.close('all')
    #
    np.save('./data/ac_vs_g', ac)
    fig, ax = plt.subplots()
    for p in rel_in_strengths:
        ax.plot(ac[f'p{p}'][0], ac[f'p{p}'][1], label=f'p = {p}')
    ax.set_xlim(0, 100)
    plt.legend(frameon=False)


def compare_stdp_on_vs_off(**params):
    # load default params
    conf = ConfigObj('src/params_bkp.cfg')
    conf.filename = 'params.cfg'
    conf.write()

    #
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # STDP OFF
    params['STDP_ON'] = 0
    params['STP_ON'] = 0
    update_sim_param(**params)
    conf_updated = ConfigObj('src/params.cfg')
    NE = conf_updated['NE']
    K = conf_updated['K']

    stdp.simu()
    st = np.loadtxt('data/spikes.txt')
    pu.plot_raster(st, ax=ax[0])
    ax[0].set_title('STDP & STD OFF')
    fig.suptitle(f'(NE = {NE}, K = {K})', fontsize=20, fontweight='bold')

    fig, axr = plt.subplots()
    plot_avg_rates(ax=axr, label_tag='Plasticity OFF')

    # STDP ON
    params['STDP_ON'] = 1
    params['STP_ON'] = 1
    update_sim_param(**params)
    stdp.simu()
    st = np.loadtxt('data/spikes.txt')
    pu.plot_raster(st, ax=ax[1])
    ax[1].set_title('STDP & STD ON')
    ax[1].set_ylabel('Neuron #')
    ax[1].set_xlabel('Time (s)')

    #
    plot_avg_rates(ax=axr, label_tag='Plasticity ON')
    plt.legend(frameon=False,
               prop={'size': 16},
               bbox_to_anchor=(1.05, 1),
               loc='upper left')

    #
    restore_default_params()


def update_sim_param(filename='params.cfg', **params):
    # overwrites the file
    conf = ConfigObj(filename)
    for key, value in params.items():
        conf[key] = value
    conf.write()


def check_balance_conditions(params, verbose=True):
    JE0 = float(params["Je0"])
    JI0 = float(params["Ji0"])
    JEE = float(params["Jee"])
    JIE = float(params["Jie"])
    JEI = float(params["Jei"])
    JII = float(params["Jii"])

    JE = -JEI / JEE
    JI = -JII / JIE
    # E = JE0
    # I = JI0  # noqa

    condition1 = (JE <= JI)
    condition2 = (JE0 / JI0 <= JE / JI)
    condition3 = (JE0 / JI0 <= 1)
    condition4 = (JE / JI <= 1)
    # print(f"JE = -JEI/JEE = {JE}")
    # print(f"JI = -JII/JIE = {JI}")

    if verbose:
        if condition1:
            print("NOT IN BALANCED REGIME!: ")
            print("JE > JI must be True")
            print(f"JE = {JE} JI = {JI}")
        if condition2:
            print("NOT IN BALANCED REGIME!: ")
            print("JE0/JI0 > JE/JI must be True")
            print(f"JE0/JI0 = {JE0/JI0}")
            print(f"JE/JI = {JE/JI}")
        if condition3:
            print("NOT IN BALANCED REGIME!: ")
            print("JE0/JI0 > 1 must be True")
            print(f"JE0/JI0 = {JE0/JI0}")
        if condition4:
            print("NOT IN BALANCED REGIME!: ")
            print("JE/JI > 1 must be True")
            print(f"JE/JI = {JE/JI}")
    elif condition1 or condition2 or condition3 or condition4:
        print("NOT IN BALANCED REGIME!: ")

        # raise SystemExit


def set_rbm_analysis_ARGS():
    PARSER = argparse.ArgumentParser(description="rbm analysis")
    PARSER.add_argument('--dataset', default='model')
    PARSER.add_argument('-p', '--p_day', type=int, default=0)
    PARSER.add_argument('-t', '--trial_idx', type=int, default=0)
    PARSER.add_argument('-s', '--selectivity_thresh', type=float, default=0.5)
    PARSER.add_argument('-c', '--clustering_param', type=float, default=0.75)
    PARSER.add_argument('--dataset_folder', default='')
    rbm_analysis_ARGS = PARSER.parse_args(
        "")  # works with jupyter notebook only when empty string passed!
    return rbm_analysis_ARGS


def call_rbm_analysis():
    rbm_analysis_ARGS = set_rbm_analysis_ARGS()
    ad.setup_folders(dataset='model')
    ad.main(rbm_analysis_ARGS)


def plot_avg_rates(axobj=None, label_tag='', inset=False):
    re = np.loadtxt('data/pop_rates_e.txt')
    ri = np.loadtxt('data/pop_rates_i.txt')

    # acfunc = ad.matrix_funcs.autocorrelation
    acfunc = auto_corr
    ac_re = acfunc(re[:, 1])
    ac_ri = acfunc(ri)
    # ac_ri = ad.matrix_funcs.autocorrelation(re[:, 1])

    if axobj is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 4.8)
    else:
        if inset:
            # ax = inset_axes(axobj, width="25%", height="25%", loc=1)
            ax = inset_axes(axobj,
                            width="100%",
                            height="100%",
                            bbox_to_anchor=(1.05, .6, .5, .4),
                            bbox_transform=axobj.transAxes,
                            loc=2,
                            borderpad=0)
        else:
            ax1 = axobj[0]
            ax2 = axobj[1]

    t = re[:, 0] * 1e-3
    ax1.plot(t,
             re[:, 1],
             label=r'$r_E$' + f'({re[:, 1].mean():.2f}Hz) ' + label_tag)
    ax1.plot(t, ri, label=r'$r_I$' + f'({ri.mean():.2f})Hz ' + label_tag)
    ax1.set_ylabel('Avg pop rates')
    ax1.set_xlabel('Time (s)')
    ax1.legend(frameon=False, loc=0)  #, prop={'size': 16})
    # plt.legend(frameon=False,
    #            prop={'size': 16},
    #            bbox_to_anchor=(1.05, 1),
    #            loc='upper left')

    max_lag = ac_re.size // 2
    ax2.plot(t[:max_lag], ac_re[:max_lag])
    ax2.plot(t[:max_lag], ac_ri[:max_lag])
    ax2.set_ylabel('Auto-Corr')
    ax2.set_xlabel('lag (s)')
    return t


def auto_corr(y):
    yunbiased = y - np.mean(y)
    ynorm = np.sum(yunbiased**2)
    acor = np.correlate(yunbiased, yunbiased, "same") / ynorm
    # use only second half
    acor = acor[len(acor) // 2:]
    return acor


def display_sim_results(params):
    #
    print("loading spikes")
    st = np.loadtxt('data/spikes.txt')
    print("done")

    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[:, :-1])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, 2])

    pu.plot_raster(st, ax=ax0)
    # if int(params["IS_RANDOM"]):
    #     pu.plot_raster(st)
    # else:
    #     raster = ad.raster_from_spike_times(st, float(params['dt']))
    #     seaborn.heatmap(raster)
    fig.set_size_inches(20, 4.8)
    plt.ylabel('Neuron #')
    plt.xlabel('Time (s)')
    #
    # t = plot_avg_rates(axobj=plt.gca(), inset=True)

    t = plot_avg_rates(axobj=[ax1, ax2])

    # vm = np.loadtxt('data/weights.txt')
    # plt.figure()
    # plt.gcf().set_size_inches(10, 4.8)
    # plt.plot(vm[:, 0] * 1e-3, vm[:, 3])
    # ax = plt.gca()
    # st0 = st[st[:, 1] == 0, 0]
    # for ist in st0:
    #     ax.axvline(ist, alpha=0.5, color='k')

    #
    if int(params["STDP_ON"]):
        print("loading weights")
        aw = np.loadtxt('data/all_weights.txt')
        print("done")
        # stdp_cons = np.loadtxt(
        #     'data/is_stdp_con.txt')  # stdp_con_idx = np.where(stdp_cons)[0]
        # aw = aw[:, stdp_cons]
        plt.figure()
        plt.gcf().set_size_inches(10, 4.8)
        plt.plot(t, aw)
        plt.ylabel('stdp weights')
        plt.xlabel('Time (s)')


def mean_field_rates(cur_conf):
    JE0 = float(cur_conf['Je0'])
    JI0 = float(cur_conf['Ji0'])
    JEE = float(cur_conf['Jee'])
    JIE = float(cur_conf['Jie'])
    JEI = float(cur_conf['Jei'])
    JII = float(cur_conf['Jii'])
    #
    # v_threshold = float(cur_conf['V_threshold_initial'])
    # v_rest = float(cur_conf['V_rest'])
    v_ext = float(cur_conf['v_ext'])
    # tau_mem = float(cur_conf['tau_mem'])

    # - - - -
    JA0 = np.array([[JE0], [JI0]])
    JAB = np.array([[JEE, JEI], [JIE, JII]])

    I_FF = np.array([[v_ext], [v_ext]]) * 1e3

    rates = -np.linalg.inv(JAB).dot(JA0) * I_FF
    print('Mean field rates:')
    print(f'r_E = {rates[0][0]}')
    print(f'r_I = {rates[1][0]}')


def pop_spike_count_ac(dt=0.1, smoothin_window=3):
    st = np.loadtxt('data/spikes.txt')
    st[:, 0] *= 1e3
    t_min = 0  # st[:, 0].min()
    t_max = st[:, 0].max()
    # idx = np.logical_and(st[:, 0] > t_min, st[:, 0] <= t_max)
    bins = np.arange(t_min, t_max + dt, dt)
    cnts, bins = np.histogram(st[:, 0], bins)
    acfunc = auto_corr
    smth_cnts = matrix_funcs.smooth_signal(cnts, smoothin_window)
    ac_re = acfunc(smth_cnts)
    # max_lag = 100  # ms  ac_re.size
    lags = bins[:ac_re.size]
    return lags, ac_re


def g_vs_autocorr(rel_in_strengths=[1.0, 2.0], t_stop=2000.0):
    #
    try:
        ac = np.load('data/ac_vs_g.npy', allow_pickle=True)[()]
    except FileNotFoundError:
        ac = {}

    for g in rel_in_strengths:
        print(f'g={g}')
        runsim(g_brunel=g, t_stop=t_stop, discard_time=250.0)
        lags, ac_re = pop_spike_count_ac()
        ac[f'g{g}'] = (lags, ac_re)
        plt.close('all')
    #
    np.save('./data/ac_vs_g', ac)
    fig, ax = plt.subplots()
    for g in rel_in_strengths:
        ax.plot(ac[f'g{g}'][0], ac[f'g{g}'][1], label=f'g = {g}')
    ax.set_xlim(0, 50)
    ax.set_xlabel('lag')
    plt.legend(frameon=False)


def plot_ac_vs_param(win_size=21, param='g'):
    if param == "g":
        acvsg = np.load('data/ac_vs_g.npy', allow_pickle=True)[()]
    elif param == "delay":
        acvsg = np.load('data/delay_vs_ac.npy', allow_pickle=True)[()]
    fig, ax = plt.subplots()
    # acvsglist = {}
    for key, value in acvsg.items():
        if param == 'g':
            param_val = float(key[1:])
        elif param == 'delay':
            param_val = float(key[5:])
        lags, ac = value
        ac_smthd = matrix_funcs.smooth_signal(ac, win_size)
        ax.plot(lags, ac_smthd[:ac.size], lw=2, label=f'{param}={param_val}')
    ax.axhline(0, color='k', lw=1.0)
    ax.set_xlim(0, 50)
    ax.legend(frameon=False)


if __name__ == "__main__":
    # create data directory in current folder
    cwd = Path.cwd()
    print(cwd)
    ad.create_directory(cwd / Path('data/'))

    print("Running simulation")

    # simulation
    runsim()

    #
    # display_sim_results()

    # ipdb.set_trace()

    # call_rbm_analysis()
