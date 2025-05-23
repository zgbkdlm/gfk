import numpy as np

methods = ['wu-tweedie', 'wu-euler', 'dps-tweedie', 'dps-euler', 'aux']
dxs = [256, ]
offsets = [0., 5., 10.]
nparticles = 16384
nmcs = 100

for method in methods:
    for dx in dxs:
        for offset in offsets:
            prefix = method + f'-{dx}-{nparticles}-{offset}'

            swds = np.zeros(nmcs)
            esss = np.zeros(nmcs)
            for k in range(nmcs):
                filename = prefix + f'-{k}.npz'

                swds[k] = np.load(f'./results/gms/{filename}')['swd']
                esss[k] = np.mean(np.load(f'./results/gms/{filename}')['esss'])

            mean = np.mean(swds)
            std = np.std(swds)

            print(f'Method {method} | Dim / nparticles {dx} / {nparticles} | Offset {offset} '
                  f'| Mean: {mean:.4f} | Std: {std:.4f} | ESS {np.mean(esss):.1f}')

print('===========================')
methods = ['aux-noiseless', 'mcgdiff']
nparticless = [1024, 4096, 16384]
for method in methods:
    for dx in dxs:
        for nparticles in nparticless:
            prefix = method + f'-{dx}-{nparticles}-0.0'

            swds = np.zeros(nmcs)
            esss = np.zeros(nmcs)
            for k in range(nmcs):
                filename = prefix + f'-{k}.npz'

                swds[k] = np.load(f'./results/gms/{filename}')['swd']
                esss[k] = np.mean(np.load(f'./results/gms/{filename}')['esss'])

            mean = np.mean(swds)
            std = np.std(swds)

            print(f'Method {method} | Dim / nparticles {dx} / {nparticles} '
                  f'| Mean: {mean:.4f} | Std: {std:.4f} | ESS {np.mean(esss):.1f}')
