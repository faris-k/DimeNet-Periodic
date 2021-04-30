import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rcParams.update({'font.size': 20})

# Adapted from https://github.com/kaaiian/ML_figures

def act_pred(y_act, y_pred, target, r2, mae,
             name='example',
             x_hist=True,
             y_hist=True,
             save_dir=None):

    plt.figure(1, figsize=(6, 6), dpi=300)
    left, width = 0.1, 0.65

    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.15]
    rect_histy = [left_h, bottom, 0.15, height]

    ax2 = plt.axes(rect_scatter)
    ax2.tick_params(direction='in',
                    length=7,
                    top=True,
                    right=True)

    # add minor ticks
    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax2.get_xaxis().set_minor_locator(minor_locator_x)
    ax2.get_yaxis().set_minor_locator(minor_locator_y)
    plt.tick_params(which='minor',
                    direction='in',
                    length=4,
                    right=True,
                    top=True)

    ax2.plot(y_act, y_pred, 'o', mfc='lightblue', alpha=0.5, label=None,
             mec='dimgrey', mew=1.2, ms=7)
    ax2.plot([-10**9, 10**9], [-10**9, 10**9], 'k--', alpha=0.8, lw=1.5,
             label='Ideal')

    label_units = {'matbench_jdft2d': '$E_{Exfol.}$ [eV/atom]',
                   'matbench_phonons': 'PhDOS Peak [1/cm]',
                   'matbench_dielectric': '$n$',
                   'matbench_log_kvrh': 'log$(K)$ [log(GPa)]',
                   'matbench_log_gvrh': 'log$(G)$ [log(GPa)]',
                   'matbench_perovskites': '$E_{form}$ [eV]',
                   'matbench_mp_gap': '$E_{g}$ [eV]',
                   'matbench_mp_is_metal': 'Metallicity',
                   'matbench_mp_e_form': '$E_{form}$ [eV/atom]'}
    lab = label_units[target]

    ax2.set_ylabel(f'Predicted {lab}')
    ax2.set_xlabel(f'Actual {lab}')
    x_range = max(y_act) - min(y_act)
    ax2.set_xlim(max(y_act) - x_range*1.05,
                 min(y_act) + x_range*1.05)

    ax2.set_ylim(max(y_act) - x_range*1.05,
                 min(y_act) + x_range*1.05)

    ax1 = plt.axes(rect_histx)
    ax1_n, ax1_bins, ax1_patches = ax1.hist(y_act,
                                            bins=31,
                                            density=True,
                                            color='#C0C0C0',
                                            edgecolor='black',
                                            alpha=0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(ax2.get_xlim())
    ax1.axis('off')

    if x_hist:
        [p.set_alpha(1.0) for p in ax1_patches]

    ax3 = plt.axes(rect_histy)
    ax3_n, ax3_bins, ax3_patches = ax3.hist(y_pred,
                                            bins=31,
                                            density=True,
                                            color='#C0C0C0',
                                            edgecolor='black',
                                            orientation='horizontal',
                                            alpha=0)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_ylim(ax2.get_ylim())
    ax3.axis('off')

    if y_hist:
        [p.set_alpha(1.0) for p in ax3_patches]
    
    max_ax_val = min(y_act) + x_range*1.05
    x_ = .05 * max_ax_val
    y_ = .82 * max_ax_val
    y_2 = .9 * max_ax_val

    ax2.text(x_, y_2, f'$R^2={r2:.4f}$')
    ax2.text(x_, y_, f'MAE$={mae:.4f}$')
    ax2.legend(loc='lower right', framealpha=0.35, handlelength=1.5)
    plt.draw()

    if save_dir is not None:
        fig_name = f'{save_dir}/{name}_act_pred.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.show()


def loss_curve(x_data, train_err, val_err, name='example', save_dir=None):

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.plot(x_data, train_err, '-', lw=2, color='#003f5c', alpha=0.8,
            label='train')
    ax.plot(x_data, val_err, '-', color='maroon', lw=2, alpha=0.9,
            label='validation')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('MAE')
    ax.set_ylim(0, 2 * np.mean(val_err))

    ax.legend(loc=1, framealpha=0.35, handlelength=1.5)

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    ax.tick_params(right=True,
                   top=True,
                   direction='in',
                   length=7)
    ax.tick_params(which='minor',
                   right=True,
                   top=True,
                   direction='in',
                   length=4)

    if save_dir is not None:
        fig_name = f'{save_dir}/{name}_loss_curve.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.show()
