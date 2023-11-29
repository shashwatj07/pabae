import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.neighbors import KernelDensity
import multiprocessing
import pickle

HOME = "/scratch/sharma96/ab/abae/data/data/"


# def parallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
#     with multiprocessing.Pool(thread_count) as p:
#         return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

class Records:
    def __init__(self, k, proxy_scores, statistics, predicates, density_estimates=None):
        assert(proxy_scores.shape == predicates.shape == statistics.shape)
        self.k = k
        self.proxy_scores = proxy_scores
        self.statistics = statistics
        self.predicates = predicates
        self.density_estimates=density_estimates
        # proxy_range=max(self.proxy_scores)-min(self.proxy_scores)
        if density_estimates is not None:
                    
                proxy_vector=self.proxy_scores.reshape(-1,1)
                self.kde = KernelDensity(kernel='gaussian', bandwidth=0.1)  # You can adjust the kernel and bandwidth
                self.kde.fit(proxy_vector)
                print("kde fit done!")
        
        self.sort = np.argsort(proxy_scores)
        self.statistics_sorted = statistics[self.sort]
        self.predicates_sorted = predicates[self.sort]
        self.ground_truth = statistics[predicates].mean()
        self.proxy_scores_sorted=proxy_scores[self.sort]

        # self.density_estimates=parallel_score_samples(self.kde,self.proxy_scores_sorted.reshape(-1,1))
        # print("estimates computed!")

        self.p = np.sum(self.predicates) / len(self.proxy_scores)
        self.sigma = np.std(self.statistics[self.predicates])
        self.strata = np.array_split(np.arange(self.sort.shape[0]), self.k)
        self.ps = np.zeros(self.k)
        self.sigmas = np.zeros(self.k)
        self.ms = np.zeros(self.k)
        for i in range(self.k):
            stratum = self.strata[i]
            _statistics = self.statistics_sorted[stratum].copy()
            _predicates = self.predicates_sorted[stratum].copy()
            self.ps[i] = np.sum(_predicates) / len(_predicates)
            self.sigmas[i] = np.std(_statistics[_predicates])
            self.ms[i] = np.mean(_statistics[_predicates])

    def sample(self, n, k=None):
        if k is None:
            sample_idxs = np.random.choice(self.sort.shape[0], n, replace=False)
        else:
            strata = np.array_split(np.arange(self.sort.shape[0]), self.k)[k]
            sample_idxs = np.random.choice(strata, n, replace=False)

        statistics = self.statistics_sorted[sample_idxs].copy()
        predicates = self.predicates_sorted[sample_idxs].copy()
        return statistics, predicates

    def sample_adaptive(self, n, k=None):
        if k is None:
            sample_idxs = np.random.choice(self.sort.shape[0], n, replace=False)
        else:
            # strata = np.array_split(np.arange(self.sort.shape[0]), self.k)[k]
            # print(len(self.strata))
            sample_idxs = np.random.choice(self.strata[k], n, replace=False)
        statistics = self.statistics_sorted[sample_idxs].copy()
        predicates = self.predicates_sorted[sample_idxs].copy()
        self.strata[k]= np.setdiff1d(self.strata[k], sample_idxs) #np.delete(self.strata[k][sample_idxs])
        return statistics, predicates
    
    def sample_adaptive_pattern(self, n, k=None):
        if k is None:
            # sample_idxs = np.random.choice(self.sort.shape[0], n, replace=False)
            print("won't work")
            raise Exception
        else:
            # strata = np.array_split(np.arange(self.sort.shape[0]), self.k)[k]
            # print(len(self.strata))
            sample_idxs = np.random.choice(self.strata[k], n, replace=False)
        statistics = self.statistics_sorted[sample_idxs].copy()
        predicates = self.predicates_sorted[sample_idxs].copy()
        
        range_size=30
        ranges = np.arange(-range_size, range_size + 1)
        idxs=np.where(np.isin(self.sort,sample_idxs))[0]

        # Broadcast to create a 2D array where each row is the range around an element
        to_remove = idxs[:, None] + ranges
        
        # Flatten the 2D array and remove duplicates
        to_remove = to_remove.flatten()
        to_remove=to_remove[to_remove<len(self.sort)]
        to_remove=self.sort[to_remove]
        for i in range(self.k):
            self.strata[i]= np.setdiff1d(self.strata[i], to_remove) #np.delete(self.strata[k][sample_idxs])
        return statistics, predicates
    
    def sample_kde(self, n, k=None):
        # samples=self.kde.sample(n).reshape(-1)
        sample_idxs=np.random.choice(self.sort.shape[0], n, replace=False)
        # Calculate absolute differences between 'samples' and 'scores' using broadcasting
        # absolute_differences = np.abs(self.proxy_scores_sorted - samples[:, np.newaxis])

        # # Find the index with the minimum absolute difference for each 'sample'
        # sample_idxs = np.argmin(absolute_differences, axis=1)

        # sample_idxs=np.zeros(n).astype(np.int64)
        # for i in range(len(samples)):
            # absolute_difference_id = np.argmin(np.abs(self.proxy_scores_sorted - samples[i]))
            # sample_idxs[i]=absolute_difference_id

        # print(sample_idxs)
        # sample_idxs=self.proxy_scores_sorted[closest_indices]

        statistics = self.statistics_sorted[sample_idxs]
        predicates = self.predicates_sorted[sample_idxs]
        return sample_idxs, statistics, predicates
    
    def summary(self):
        num_records = len(self.proxy_scores)
        table = tabulate(
            [
                ["NUM_RECORDS", num_records],
                ["K", self.k],
                ["P_S", np.round(self.ps.sum() / self.k, 5)],
                ["SIGMA_S", np.round(self.sigma, 5)],
                ["P_K", np.round(self.ps, 5)],
                ["SIGMA_K", np.round(self.sigmas, 5)],
                ["M_K", np.round(self.ms, 5)],
            ],
            headers=["Key", "Value"]
        )
        print(table)
        
    
class JacksonRecords(Records):
    def __init__(self, k):
        self.name = "jackson"
        # df = pd.read_csv(HOME + f"{self.name}.csv")
        df=pd.read_csv(HOME + "nightstreet_proxy_score_clip.csv")
        proxy_scores = df["proxy"].to_numpy()
        statistics = df["statistic"].to_numpy()
        predicates = df["predicate"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class JacksonRedLightRecords(Records):
    def __init__(self, k):
        self.name = "jackson_red_light"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class AmazonOfficeSuppliesRecords(Records):
    def __init__(self, k):
        self.name = "amazon_office"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class MovieFacesV2Records(Records):
    def __init__(self, k):
        self.name = "moviefacesv2"
        # proxy_scores = np.load("/future/u/jtguibas/aggpred/data/movie-faces-proxy-score-v3.npy")[:, 0]
        # predicates = np.load("/future/u/jtguibas/aggpred/data/movie-faces-predicates-v2.npy")
        # statistics = np.load("/future/u/jtguibas/aggpred/data/movie-faces-statistics-v2.npy")
        
        df=pd.read_csv(HOME+"amazon-all.csv")
        proxy_scores = df["Proxy"].to_numpy()
        statistics = df["Statistic"].to_numpy()
        predicates = df["Predicate"].to_numpy().astype(int)
        super().__init__(k, proxy_scores, statistics, predicates)

        
        
        
class CelebARecords(Records):
    def __init__(self, k):
        self.name = "CelebARecords"
        # df = pd.read_csv(HOME + f"{self.name}.csv")
        df=pd.read_csv(HOME+"celeba_all.csv")
        proxy_scores = df["Proxy"].to_numpy()
        statistics = df["Statistic"].to_numpy()
        predicates = df["Predicate"].to_numpy().astype(int)
        with open(HOME + 'density_estimates_celeba.pickle', 'rb') as handle:
            density_estimates = pickle.load(handle)
        super().__init__(k, proxy_scores, statistics, predicates, density_estimates)    
        

class Trec05PRecords(Records):
    def __init__(self, k):
        self.name = "trec05p"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistic"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
class TaipeiRecords(Records):
    def __init__(self, k):
        self.name = "taipei"
        # df = pd.read_csv(HOME + f"{self.name}.csv")
        # proxy_scores = df["proxy_scores"].to_numpy()
        # statistics = df["statistics"].to_numpy()
        # predicates = df["predicates"].to_numpy()
        # super().__init__(k, proxy_scores, statistics, predicates)
        
        df=pd.read_csv(HOME + "taipei_proxy_score_clip.csv")
        proxy_scores = df["proxy"].to_numpy()
        statistics = df["statistic"].to_numpy()
        predicates = df["predicate"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class SyntheticRecords(Records):
    def __init__(self, k, alpha=0.1, beta=0.5, N=1000000):
        self.name = "synthetic"
        rng = np.random.RandomState(3212142)
        proxy_scores = rng.beta(alpha, beta, size=N)
        statistics = rng.normal(10, 3, N)
        predicates = rng.binomial(n=1, p=proxy_scores).astype(bool)
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class SyntheticControlRecords(Records):
    def __init__(self, k, ps, sigmas, ms, N=1000000):
        self.name = "synthetic_control"
        rng = np.random.RandomState(3212142)
        strata_size = N // k
        proxy_scores = []
        statistics = np.concatenate([rng.normal(ms[i], sigmas[i], N // k) for i in range(k)])
        predicates = []
        for i in range(k):
            a = rng.binomial(n=1, p=[ps[i]]*strata_size)
            c = rng.binomial(n=strata_size, p=a).astype(bool)
            proxy_scores.append(a)
            predicates.append(c)
        proxy_scores = np.arange(N)
        predicates = np.concatenate(predicates)
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class SyntheticComplexPredicatesRecords(Records):
    def __init__(self, k, version="opt"):
        self.name = f"synthetic_complex_predicates_{version}"
        N = 1000000
        rng = np.random.RandomState(3212142)

        proxy_scores_a = rng.beta(0.4, 1, N)
        proxy_scores_b = rng.beta(0.2, 1, N)

        proxy_scores_gt = proxy_scores_a * proxy_scores_b

        if version == "opt":
            proxy_scores = proxy_scores_gt.copy()
        elif version == "left":
            proxy_scores = proxy_scores_a
        elif version == "right":
            proxy_scores = proxy_scores_b
        else:
            raise NotImplementedError

        statistics = rng.normal(10, 3, N)
        predicates = np.zeros(N)
        predicates = rng.binomial(n=1, p=proxy_scores_gt)

        predicates = predicates.astype(bool)
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class JacksonRedLightMultProxyRecords(Records):
    def __init__(self, k):
        self.name = "jackson_red_light_mult"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class JacksonRedLightCarProxyRecords(Records):
    def __init__(self, k):
        self.name = "jackson_red_light_car_proxy"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
        
        
class JacksonRedLightLightProxyRecords(Records):
    def __init__(self, k):
        self.name = "jackson_red_light_light_proxy"
        df = pd.read_csv(HOME + f"{self.name}.csv")
        proxy_scores = df["proxy_scores"].to_numpy()
        statistics = df["statistics"].to_numpy()
        predicates = df["predicates"].to_numpy()
        super().__init__(k, proxy_scores, statistics, predicates)
