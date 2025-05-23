import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

# Times
t0, T = 0., 2.
nsteps = 100
dt = T / nsteps
ts = np.linspace(0., T, nsteps + 1)

seed = 7

methods = ['wu-euler', 'aux']
dx = 256
offsets = [0., 5., 10.]
nparticles = 16384

fig, axes = plt.subplots(2, 3, figsize=(12, 10), sharex=True, sharey=True)

for i, method in enumerate(methods):
    for j, offset in enumerate(offsets):
        prefix = method + f'-{dx}-{nparticles}-{offset}'
        filename = prefix + f'-{seed}.npz'
        esss = np.load(f'./results/gms/{filename}')['esss']

        axes[i][j].plot(ts, esss, linewidth=2, c='black')
        axes[i][j].grid(linestyle='--', alpha=0.3, which='both')
        axes[i][j].set_ylim(0, 16384)

plt.tight_layout(pad=0.1)
plt.show()