import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/paper.mplstyle.py')

path_lowtol = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230822_ModelC_test_my_lowtol_opt/TEST/All_states_solutions_check_negative.pkl'
path_og = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230822_ModelC_test_my_negative_opt/TEST/All_states_solutions_check_negative.pkl'
df_opt_lowtol = pd.read_pickle(path_lowtol)
df_opt_og = pd.read_pickle(path_og)
states = [
    'ssDNA p1', 'ssDNA p2:tRNA', 'ssDNA p2:ctRNA', 'RT', 'RNase H', 'RT-ssDNA p2:tRNA', 'RT-ssDNA p1:cvRNA',
    'RT-ssDNA p2:ctRNA', 'cDNA2:tRNA', 'cDNA2:tRNA: RNase H', 'cDNA2:tRNA frag', 'cDNA2:ssDNA p1', 'cDNA2:ssDNA p1:RT',
    'T7 RNAP', 'dsDNA T7 target', 'T7: dsDNA T7 target', 'tRNA (target)', 'target aCas13a-gRNA', 'dsRNA (input:target)',
    'quench-ssRNA-fluoro', 'quencher', 'fluorophore (output)']


all_nonneg_states = []
for i, state in enumerate(states):
    state_vals_nonneg = df_opt_lowtol.at[state, '[5.0, 2.5, 0.02, 1, 90]']
    all_nonneg_states.append(state_vals_nonneg)
all_nonneg_states_arr = np.vstack(all_nonneg_states)


all_neg_states = []
for i, state in enumerate(states):
    state_vals_neg = df_opt_og.at[state, '[5.0, 2.5, 0.02, 1, 90]']
    all_neg_states.append(state_vals_neg)

all_neg_states_arr = np.vstack(all_neg_states)

# print(all_nonneg_states_arr)
# print(all_neg_states_arr)

state_diffs = np.subtract(all_nonneg_states, all_neg_states)
state_diffs_abs = np.absolute(state_diffs)
frac_change = np.divide(state_diffs_abs, all_neg_states_arr)
pct_change = frac_change*100
# for i in pct_change:
#     print(i)
pct_change_avg = np.nanmean(pct_change, axis=1)
print(pct_change_avg)


def plot_all_states(
        df_og: pd.DataFrame,
        df_lowtol: pd.DataFrame,
        dose: list, #str
        states: list
):
    
    
    time = np.linspace(0, 240, 61)


    fig, axs = plt.subplots(nrows=2, ncols=4, sharex=False, sharey=False, figsize = (8, 12))
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0)
    axs = axs.ravel()

    for i, state in enumerate(states):
        axs[i].plot(time, df_og.at[state, str(dose)], label='original')
        axs[i].plot(time, df_lowtol.at[state, str(dose)], label='low_tol')
        axs[i].set_xlabel('time (min)')
        axs[i].set_ylabel('simulation value')
        axs[i].set_title(state)
        axs[i].set_box_aspect(1)
    axs[0].legend()
    # axs[-1].axis('off')
    # axs[-2].axis('off')
    # axs[-3].axis('off')
    # plt.show()
    path_fig = '/Users/kdreyer/Documents/Github/COVID_Dx_GAMES/Results/230822_ModelC_test_my_lowtol_opt/TEST/'
    plt.savefig(path_fig+'Compare_all_non_neg_states_mid'+'.svg')
    # plt.show()

# plot_all_states(df_opt_og, df_opt_lowtol, [5.0, 2.5, 0.02, 1, 90], states)

states_mid = ['ssDNA p2:tRNA', 'ssDNA p2:ctRNA', 'cDNA2:tRNA', 'cDNA2:ssDNA p1', 'cDNA2:ssDNA p1:RT', 'dsDNA T7 target', 'tRNA (target)', 'target aCas13a-gRNA']
states_high = ['RT-ssDNA p2:tRNA','RT-ssDNA p1:cvRNA', 'RT-ssDNA p2:ctRNA', 'cDNA2:tRNA frag', 'quencher', 'fluorophore (output)']

plot_all_states(df_opt_og, df_opt_lowtol, [5.0, 2.5, 0.02, 1, 90], states_mid)
