import numpy as np

methods = ['wu-tweedie', 'wu-euler', 'dps-tweedie', 'dps-euler', 'aux']
dxs = [128, 256]
offsets = [0., 5., 10.]
nparticles = 16384
nmcs = 100

for method in methods:
    for dx in dxs:
        for offset in offsets:
            prefix = method + f'-{dx}-{nparticles}-{offset}'

            swds = np.zeros(nmcs)
            for k in range(nmcs):
                filename = prefix + f'-{k}.npz'

                swds[k] = np.load(f'./results/gms/{filename}')['swd']

            mean = np.mean(swds)
            std = np.std(swds)

            print(f'Method {method} | Dim / nparticles {dx} / {nparticles} | Offset {offset} '
                  f'| Mean: {mean:.4f} | Std: {std:.4f}')
