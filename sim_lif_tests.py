import numpy as np
import matplotlib.pyplot as plt
import ipdb
from scipy.optimize import curve_fit
import sys
sys.path.append('/Users/shrisha/code/rbm_analysis/helpers')
import plot_utils as pp  # noqa

sys.path.append('./src/build/lib.macosx-10.7-x86_64-3.7')
import stdp

from configobj import ConfigObj


def update_sim_param(filename='params.cfg', **kwargs):
    # overwrites the file
    conf = ConfigObj(filename)
    for key, value in kwargs.items():
        conf[key] = value
    conf.write()


def run_stdp_test(**params):
    # load test params
    conf = ConfigObj('src/test_params.cfg')
    conf.filename = 'params.cfg'
    conf.write()
    update_sim_param(**params)
    stdp.simu()
    conf = ConfigObj('params.cfg')
    NE = int(conf['NE'])
    # print(NE)

    stdp_tests_display_results(NE - 1)

    # restore default params
    conf = ConfigObj('src/params_bkp.cfg')
    conf.filename = 'params.cfg'
    conf.write()


def double_exponential(t, t0, w0, wp, wn, tau):
    return w0 + np.where(t >= t0, wp * np.exp(-(t - t0) / tau),
                         wn * np.exp((t - t0) / tau))


def stdp_tests_display_results(n_cons=101):
    # df = './src/build/lib.macosx-10.7-x86_64-3.7/'
    df = './data/'
    aw = np.loadtxt(df + 'all_weights.txt')
    st = np.loadtxt(df + 'spikes.txt')
    t = aw[:, 0]
    aw = aw[:, 1:]
    st[:, 0] *= 1000
    fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(3 * 4, 10))
    fig.suptitle('STDP test')
    pp.plot_raster(st, ax=ax[0])
    st0 = st[st[:, 1] == 0, 0]
    for ist in st0:
        ax[0].axvline(ist, alpha=0.4)
    ax[0].set_ylabel('Neuron #')
    ax[0].set_title('post syn neuron 0')
    # ax[0].set_xlim(0, 1000)

    ax[1].plot(t, aw)
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Weights')
    # ax[1].set_xlim(0, 1000)

    xaxis = np.arange(n_cons // 2, -n_cons // 2, -1)
    final_weights = aw[-1, :]
    ax[2].plot(xaxis, final_weights, '.k', alpha=0.5)

    p0 = (-1.0, 5e-1, 1e-2, -1.13e-2, 20.0)
    popt, pcov = curve_fit(double_exponential,
                           xaxis,
                           final_weights,
                           p0,
                           ftol=1e-10)
    print(
        "Best fit parameters: t0={0}, w0={1}, wp={2}, wn={3}, tau={4}".format(
            *popt))

    fit_x = np.linspace(n_cons // 2, -n_cons // 2, 100)
    fit_y = double_exponential(fit_x, *popt)
    ax[2].plot(fit_x, fit_y, 'r-', alpha=.85, lw=0.5)

    ax[2].axhline(0.5, linestyle='-', color='k', linewidth=0.75)
    ax[2].axvline(0.0, linestyle='-', color='k', linewidth=0.75)
    ax[2].set_xlabel(r"$t_{post} - t_{pre} (ms)$")
    ax[2].set_ylabel(r"final weights")
    fig.savefig('./figures/stdp_test')

    # isi = np.diff(st0)
    # st1 = st[st[:, 1] == 1, 0]
    # print(isi.max() - isi.min())
    # isi = np.diff(st1)
    # print(isi.max() - isi.min())


def stdp_tests():
    w = np.loadtxt('weights.txt')
    t = w[:, 0] * 1e-3
    pre = np.loadtxt('pre_trace.txt')
    post = np.loadtxt('post_trace.txt')
    st = np.loadtxt("spikes.txt")

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, pre)
    st0 = st[st[:, 1] == 0, 0]

    for i_st in st0:
        ax[0].axvline(i_st, color='k')
    ax[0].set_title('pre trace with pre spikes')

    ax[1].plot(t, post)
    st1 = st[st[:, 1] == 1, 0]
    for i_st in st1:
        ax[1].axvline(i_st, color='g')
    ax[1].set_title('post trace with post spikes ')
    # - - - - - #
    ax[2].plot(t, w[:, 1])
    ipdb.set_trace()


def stp_tests():
    u = np.loadtxt('u.txt')
    x = np.loadtxt('x.txt')
    Ie = np.loadtxt('I.txt')
    st = np.loadtxt("spikes.txt")

    st[:, 0] = 1e-3 * st[:, 0]
    t = Ie[:, 0] * 1e-3

    plt.figure(figsize=(7.15, 7.05))
    ax_ux = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax_Ie = plt.subplot2grid((4, 1), (2, 0))
    ax_st = plt.subplot2grid((4, 1), (3, 0))

    ax_Ie.plot(t, Ie[:, 1], 'r', label=r'$I_{post}$')
    ax_Ie.legend(frameon=False)

    ax_ux.plot(t, u, 'k-', label='u')
    ax_ux.plot(t, x, 'b-', label='x')
    ax_ux.legend(frameon=False)

    st_pre = st[st[:, 1] == 0, 0]
    st_post = st[st[:, 1] == 1, 1]
    for st_pre_k in st_pre:
        ax_st.axvline(st_pre_k, color='b')
    for st_post_k in st_post:
        ax_st.axvline(st_post_k, color='k')
    ax_st.set_xlim(t.min(), t.max())

    return (ax_ux, ax_Ie, ax_st)


def pop_stats(discard_time=1):
    re = np.loadtxt('pop_rates_e.txt')
    ri = np.loadtxt('pop_rates_i.txt')
    st = np.loadtxt('spikes.txt')
    idx = st[:, 0] > discard_time


# cv = spike_stats.cv
