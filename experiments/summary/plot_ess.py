"""
Plot the effective sample size (ESS) for different methods and offsets.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 20})

# Times
t0, T = 0., 2.
nsteps = 100
dt = T / nsteps
ts = np.linspace(0., T, nsteps + 1)

seed = 99

methods = ['wu-euler', 'aux']
method_labels = ['TDS', 'B$^0$SMC']
method_style = ['--', '-']
dx = 256
offsets = [0., 5., 10.]
nparticles = 16384

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)

for method, label, style in zip(methods, method_labels, method_style):
    for j, offset in enumerate(offsets):
        prefix = method + f'-{dx}-{nparticles}-{offset}'
        filename = prefix + f'-{seed}.npz'
        esss = np.load(f'./results/gms/{filename}')['esss']

        axes[j].plot(ts, esss, linewidth=2, linestyle=style, c='black', alpha=.7, label=label)
        axes[j].grid(linestyle='--', alpha=0.3, which='both')
        axes[j].set_ylim(0, 16384)
        axes[j].set_xlabel('$t$')
        axes[j].set_title(rf'$\omega = {int(offset)}$')

axes[0].set_ylabel('ESS (max. 16384)')
axes[0].legend(loc='lower right', fontsize=20)

plt.tight_layout(pad=0.1)
plt.savefig('ess.pdf', transparent=True)
plt.show()