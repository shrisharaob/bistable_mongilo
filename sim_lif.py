import numpy as np
import matplotlib.pyplot as plt
import ipdb
import sys
sys.path.append('/Users/shrisha/code/rbm_analysis/helpers')
import plot_utils as pp  # noqa

sys.path.append('./src/build/lib.macosx-10.7-x86_64-3.7')
import stdp  # noqa

from configobj import ConfigObj


def update_sim_param(filename='param.cfg', **kwargs):
    """  USAGE: update_sim_param(filename='param.cfg', **kwargs)

    overwrites the file

    Args
    ----
      filename: float, optional  (default='param.cfg')
      **kwargs: argtype

    Returns
    -------
      out: None

    """

    conf = ConfigObj(filename)
    for key, value in kwargs.items():
        conf[key] = value
    conf.write()


def run_stp_test(**params):
    # load test params
    conf = ConfigObj('src/params.cfg')
    conf.filename = 'params.cfg'
    conf.write()
    update_sim_param(**params)

    # simulate
    stdp.simu()


def stp_tests():
    u = np.loadtxt('u.txt')
    x = np.loadtxt('x.txt')
    Ie = np.loadtxt('I.txt')
    st = np.loadtxt("spikes.txt")

    t = Ie[:, 0] * 1e-3

    plt.figure(figsize=(7.15, 7.05), constrained_layout=True)
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

    # plt.show()

    return (ax_ux, ax_Ie, ax_st)


def main():

    # run simulation
    run_stp_test()

    # plot the figure
    stp_tests()


# def pop_stats(discard_time=1):
#     re = np.loadtxt('pop_rates_e.txt')
#     ri = np.loadtxt('pop_rates_i.txt')
#     st = np.loadtxt('spikes.txt')
#     idx = st[:, 0] > discard_time

if __name__ == "__main__":
    # main()


    # simulate
    stdp.simu(()
