"""
Visualise the posterior distribution.
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

seed = 6
slice1 = 6
slice2 = 66

methods = ['wu-euler', 'aux']
method_labels = ['TDS', 'B$^0$SMC']
dx = 256
offsets = [0., 10.]
nparticles = 16384

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)

for i, offset in enumerate(offsets):
    for j, (method, label) in enumerate(zip(methods, method_labels)):
        prefix = method + f'-{dx}-{nparticles}-{offset}'
        filename = prefix + f'-{seed}.npz'
        data = np.load(f'./results/gms/{filename}')
        samples = data['samples']
        ws = np.exp(data['log_ws'])
        post_samples = data['post_samples']

        axes[i, j].hist2d(samples[:, slice1], samples[:, slice2], bins=50, weights=ws, density=True, cmap=plt.cm.binary)
        axes[0, j].set_title(label)

    axes[i, -1].hist2d(post_samples[:, slice1], post_samples[:, slice2], bins=50, density=True, cmap=plt.cm.binary)
    axes[0, -1].set_title('True posterior')

    axes[i, 0].set_ylabel(rf'$\omega = {int(offset)}$')

plt.tight_layout(pad=0.1)
plt.savefig('post.pdf', transparent=True)
plt.show()
