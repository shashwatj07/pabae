## preparing kde estimates

import multiprocessing
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

bandwidths = 10 ** np.linspace(-2, 0, 1)

def parallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

HOME = "/scratch/sharma96/ab/abae/data/data/"
name = "CelebARecords"
df = pd.read_csv(HOME + f"{name}.csv")
proxy_scores = df["proxy_scores"].to_numpy()
statistics = df["statistics"].to_numpy()
predicates = df["predicates"].to_numpy()
proxy_vector=proxy_scores.reshape(-1,1)

kde = KernelDensity(kernel='gaussian', bandwidth=0.01)  # You can adjust the kernel and bandwidth
# grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=5)
# grid.fit(proxy_vector)
# print("grid fit done!")
# best_bandwidth = grid.best_params_['bandwidth']
# print(grid.best_params_)
# print("best_bandwidth:", best_bandwidth)
# print("generating final kde estimate file")
# kde = KernelDensity(bandwidth=best_bandwidth)
kde.fit(proxy_vector)
print("kde fit done")

sort = np.argsort(proxy_scores)
statistics_sorted = statistics[sort]
predicates_sorted = predicates[sort]
ground_truth = statistics[predicates].mean()
proxy_scores_sorted=proxy_scores[sort]

density_estimates=parallel_score_samples(kde,proxy_scores_sorted.reshape(-1,1))
print("estimates computed!")

with open('density_estimates.pickle', 'wb') as handle:
    pickle.dump(density_estimates, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('density_estimates.pickle', 'rb') as handle:
    b = pickle.load(handle)


n_components = 20  # You can adjust this based on your data

# Create and fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=n_components)
gmm.fit(proxy_vector)
