#write code to read and print a npy file

import numpy as np
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt

colors = ['brown', 'black', 'gray', 'blond']
tot = 0

filename = "/scratch/sharma96/ab/abae/data/data/celeba-{}.npy"
for i in range(4):
    celeba = np.load(filename.format(colors[i]), allow_pickle=True)
    celeba = celeba[()]
    print(celeba.keys())
    print(celeba['proxy_scores'].shape)
    print(celeba['paths'][:10])
    celeba["proxy_scores"] = celeba["proxy_scores"].flatten()
    print(type(celeba))
    print(len(celeba["proxy_scores"]))
    print(sum(celeba["predicates"]))
    tot+=sum(celeba["predicates"])

print(tot)
# brown = np.load(filename, allow_pickle=True)
# brown = brown[()]
# print(brown.keys())
# print(brown['proxy_scores'].shape)
# print(brown['paths'][:10])
# brown["proxy_scores"] = brown["proxy_scores"].flatten()
# print(type(brown))
# print(len(brown["proxy_scores"]))
# print(sum(brown["predicates"]))
