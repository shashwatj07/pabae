import ray
import numpy as np
import scipy.stats as st
import scipy.optimize
import mystic as my
from tqdm import tqdm
from copy import deepcopy
import math


def _execute_ours(db, n1, n2, return_samples=False, sample_reuse=True):
    p = np.zeros(db.k)
    s = np.zeros(db.k)
    resovoir = []
    sampled=np.array([n1 for _ in range(db.k)]).astype(np.float64)
    
    for i in range(db.k):
        statistics, predicates = db.sample(n1, i)
        resovoir.append([statistics, predicates])
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
            
    weights = np.sqrt(p) * s
    norm = 1 if np.sum(weights) == 0 else np.sum(weights)
    weights = weights / norm
    allocation = np.floor(n2 * weights).astype(np.int64)
    sampled+=allocation.astype(np.float64)
    # print(allocation)

    m = np.zeros(db.k)
    for i in range(db.k):
        statistics, predicates = db.sample(allocation[i], i)
        resovoir[i][0] = np.concatenate([statistics, resovoir[i][0]])
        resovoir[i][1] = np.concatenate([predicates, resovoir[i][1]])
        if sample_reuse:
            statistics = resovoir[i][0]
            predicates = resovoir[i][1]
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 0:
            m[i] = np.mean(statistics[predicates])
    
    norm = 1 if p.sum() == 0 else p.sum()
    if return_samples:
        return np.sum(m * p / norm), resovoir
    else:
        return np.sum(m * p / norm), sampled/(n1*db.k + n2)#(allocation+n1).astype(np.float32)/(n1*db.k+n2)

def _execute_ours_max(db, n1, n2, return_samples=False, sample_reuse=True):
    p = np.zeros(db.k)
    s = np.zeros(db.k)
    resovoir = []
    
    for i in range(db.k):
        statistics, predicates = db.sample(n1, i)
        resovoir.append([statistics, predicates])
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
            
    weights = np.sqrt(p) * s
    norm = 1 if np.sum(weights) == 0 else np.sum(weights)
    weights = weights / norm
    allocation = np.floor(n2 * weights).astype(np.int64)
    sampled =np.array([n1 for _ in range(db.k)]).astype(np.float64)
    # print("mean: {}, pred_prabab: {}".format(m, p))

    m = np.zeros(db.k)
    # pbar = tqdm(total=n2)
    N=n1*db.k + n2

    while np.sum(allocation) > 0:
        i=np.argmax(allocation)
        sampled[i]+=1.0
        statistics, predicates = db.sample(1, i)
        # pbar.update(1)
        resovoir[i][0] = np.concatenate([statistics, resovoir[i][0]])
        resovoir[i][1] = np.concatenate([predicates, resovoir[i][1]])
        n2-=1
        if sample_reuse:
            statistics = resovoir[i][0]
            predicates = resovoir[i][1]
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 0:
            m[i] = np.mean(statistics[predicates])
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
        # updating allocation using strata formula
        weights = np.sqrt(p) * s
        norm = 1 if np.sum(weights) == 0 else np.sum(weights)
        weights = weights / norm
        allocation = np.floor(n2 * weights).astype(np.int64)
        # print("mean: {}, pred_prabab: {}".format(m, p))

    # pbar.close()
    norm = 1 if p.sum() == 0 else p.sum()
    if return_samples:
        return np.sum(m * p / norm), resovoir
    else:
        return np.sum(m * p / norm), sampled/(N)
    
def _execute_ours_adaptive(db, n1, n2, return_samples=False, sample_reuse=True):
    p = np.zeros(db.k)
    s = np.zeros(db.k)
    m = np.zeros(db.k)
    resovoir = []
    initial_N=n1*db.k + n2
    for i in range(db.k):
        statistics, predicates = db.sample_adaptive(n1, i)
        resovoir.append([statistics, predicates])
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
            
    weights = np.sqrt(p) * s
    norm = 1 if np.sum(weights) == 0 else np.sum(weights)
    weights = weights / norm
    sampled=np.array([n1 for _ in range(db.k)]).astype(np.float64)

    N=n2
    while np.sum(weights) ==0:
        if N<db.k:
            norm = 1.0 if p.sum() == 0 else p.sum()
            if return_samples:
                return np.sum(m * p / norm), resovoir
            else:
                return np.sum(m * p / norm), sampled/initial_N
        for i in range(db.k):
            try:
                    
                statistics, predicates = db.sample_adaptive(1, i)
                # resovoir.append([statistics, predicates])
                resovoir[i][0] = np.concatenate([statistics, resovoir[i][0]])
                resovoir[i][1] = np.concatenate([predicates, resovoir[i][1]])
                
                statistics = resovoir[i][0]
                predicates = resovoir[i][1]
                if len(predicates) > 0:
                    p[i] = np.mean(predicates)
                if np.sum(predicates) > 1:
                    s[i] = np.std(statistics[predicates], ddof=1)
                N-=1
                sampled[i]+=1.0
            except:
                continue
                
        weights = np.sqrt(p) * s
        norm = 1 if np.sum(weights) == 0 else np.sum(weights)
        weights = weights / norm
        # sampled+=1.0
        # print(np.sum(weights))

    # random_uniforms=np.random.uniform(size=N)
    strata_idxs=np.arange(0,db.k,1)
    for i in range(N):
        # cdf = np.cumsum(weights)
        # index = np.searchsorted(cdf, random_uniforms[i], side='left')
        while True:
            probs=weights[strata_idxs]
            norm=1 if np.sum(probs) == 0 else np.sum(probs)
            probs/=norm
            if np.sum(probs)==0:
                index=np.random.choice(strata_idxs)
            else:
                index=np.random.choice(strata_idxs, p=probs)
            if len(db.strata[index])==0:
                id_=np.where(strata_idxs==index)[0][0]
                strata_idxs=np.delete(strata_idxs,id_)
            else:
                break
        # print(index)
        # print(cdf)
        # print(random_uniforms[i])
        statistics, predicates = db.sample_adaptive(1,index)
        sampled[index]+=1.0
        resovoir[index][0] = np.concatenate([statistics, resovoir[index][0]])
        resovoir[index][1] = np.concatenate([predicates, resovoir[index][1]])
        statistics = resovoir[index][0]
        predicates = resovoir[index][1]
            
        if len(predicates) > 0:
            p[index] = np.mean(predicates)
        if np.sum(predicates) > 0:
            m[index] = np.mean(statistics[predicates])
        if np.sum(predicates) > 1:
            s[index] = np.std(statistics[predicates], ddof=1)
            
        weights = np.sqrt(p) * s
        norm = 1 if np.sum(weights) == 0 else np.sum(weights)
        weights = weights / norm
        
    norm = 1.0 if p.sum() == 0 else p.sum()
    if return_samples:
        return np.sum(m * p / norm), resovoir
    else:
        return np.sum(m * p / norm), sampled/initial_N
    
def _execute_ours_adaptive_pattern(db, n1, n2, return_samples=False, sample_reuse=True):
    p = np.zeros(db.k)
    s = np.zeros(db.k)
    m = np.zeros(db.k)
    resovoir = []
    initial_N=n1*db.k + n2
    for i in range(db.k):
        statistics, predicates = db.sample_adaptive_pattern(n1, i)
        resovoir.append([statistics, predicates])
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
            
    weights = np.sqrt(p) * s
    norm = 1 if np.sum(weights) == 0 else np.sum(weights)
    weights = weights / norm
    sampled=np.array([n1 for _ in range(db.k)]).astype(np.float64)

    N=n2
    while np.sum(weights) ==0:
        if N<db.k:
            norm = 1.0 if p.sum() == 0 else p.sum()
            if return_samples:
                return np.sum(m * p / norm), resovoir
            else:
                return np.sum(m * p / norm), sampled/initial_N
        for i in range(db.k):
            try:
                    
                statistics, predicates = db.sample_adaptive_pattern(1, i)
                # resovoir.append([statistics, predicates])
                resovoir[i][0] = np.concatenate([statistics, resovoir[i][0]])
                resovoir[i][1] = np.concatenate([predicates, resovoir[i][1]])
                
                statistics = resovoir[i][0]
                predicates = resovoir[i][1]
                if len(predicates) > 0:
                    p[i] = np.mean(predicates)
                if np.sum(predicates) > 1:
                    s[i] = np.std(statistics[predicates], ddof=1)
                N-=1
                sampled[i]+=1.0
            except:
                continue
                
        weights = np.sqrt(p) * s
        norm = 1 if np.sum(weights) == 0 else np.sum(weights)
        weights = weights / norm
        # sampled+=1.0
        # print(np.sum(weights))

    # random_uniforms=np.random.uniform(size=N)
    strata_idxs=np.arange(0,db.k,1)
    for i in range(N):
        # cdf = np.cumsum(weights)
        # index = np.searchsorted(cdf, random_uniforms[i], side='left')
        while True:
            probs=weights[strata_idxs]
            norm=1 if np.sum(probs) == 0 else np.sum(probs)
            probs/=norm
            if np.sum(probs)==0:
                index=np.random.choice(strata_idxs)
            else:
                index=np.random.choice(strata_idxs, p=probs)
            if len(db.strata[index])==0:
                id_=np.where(strata_idxs==index)[0][0]
                strata_idxs=np.delete(strata_idxs,id_)
            else:
                break
        # print(index)
        # print(cdf)
        # print(random_uniforms[i])
        statistics, predicates = db.sample_adaptive_pattern(1,index)
        sampled[index]+=1.0
        resovoir[index][0] = np.concatenate([statistics, resovoir[index][0]])
        resovoir[index][1] = np.concatenate([predicates, resovoir[index][1]])
        statistics = resovoir[index][0]
        predicates = resovoir[index][1]
            
        if len(predicates) > 0:
            p[index] = np.mean(predicates)
        if np.sum(predicates) > 0:
            m[index] = np.mean(statistics[predicates])
        if np.sum(predicates) > 1:
            s[index] = np.std(statistics[predicates], ddof=1)
            
        weights = np.sqrt(p) * s
        norm = 1 if np.sum(weights) == 0 else np.sum(weights)
        weights = weights / norm
        
    norm = 1.0 if p.sum() == 0 else p.sum()
    if return_samples:
        return np.sum(m * p / norm), resovoir
    else:
        return np.sum(m * p / norm), sampled/initial_N
    
def _execute_ours_eps_greedy(db, n1, n2, return_samples=False, sample_reuse=True):
    p = np.zeros(db.k)
    s = np.zeros(db.k)
    resovoir = []
    m = np.zeros(db.k)
    

    for i in range(db.k):
        statistics, predicates = db.sample(1, i)
        resovoir.append([statistics, predicates])
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
        if np.sum(predicates) > 0:
            m[i] = np.mean(statistics[predicates])
    
    N=n2+(n1-1)*db.k
    step_size=1/N
    eps=1.0
    weights = np.sqrt(p) * s
    norm = 1 if np.sum(weights) == 0 else np.sum(weights)
    weights = weights / norm
    allocation = np.floor(N * weights).astype(np.int64)
    sampled =np.array([1.0 for _ in range(db.k)])
    
    while N>0:
        # print(allocation)
        if np.random.uniform() < eps:
            i=int(np.random.choice(db.k, 1).item())
        else:
            i=np.argmax(allocation)
        sampled[i]+=1
        statistics, predicates = db.sample(1, i)
        eps-=step_size
        # pbar.update(1)
        resovoir[i][0] = np.concatenate([statistics, resovoir[i][0]])
        resovoir[i][1] = np.concatenate([predicates, resovoir[i][1]])
        N-=1
        if sample_reuse:
            statistics = resovoir[i][0]
            predicates = resovoir[i][1]
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 0:
            m[i] = np.mean(statistics[predicates])
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
        # updating allocation using strata formula
        weights = np.sqrt(p) * s
        norm = 1 if np.sum(weights) == 0 else np.sum(weights)
        weights = weights / norm
        allocation = np.floor(N * weights).astype(np.float32)
        # print("mean: {}, pred_prabab: {}".format(m, p))
    # print(sampled)
    # pbar.close()
    norm = 1 if p.sum() == 0 else p.sum()
    if return_samples:
        return np.sum(m * p / norm), resovoir
    else:
        return np.sum(m * p / norm), sampled/(n2+n1*db.k)
    
def _execute_ours_eps_greedy2(db, n1, n2, return_samples=False, sample_reuse=True):
    p = np.zeros(db.k)
    s = np.zeros(db.k)
    resovoir = []
    m = np.zeros(db.k)
    

    for i in range(db.k):
        statistics, predicates = db.sample(1, i)
        resovoir.append([statistics, predicates])
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
        if np.sum(predicates) > 0:
            m[i] = np.mean(statistics[predicates])
    
    N=(n1-1)*db.k
    step_size=1/N
    eps=1.0
    weights = np.sqrt(p) * s
    norm = 1 if np.sum(weights) == 0 else np.sum(weights)
    weights = weights / norm
    allocation = np.floor(N * weights).astype(np.int64)
    sampled =np.array([1.0 for _ in range(db.k)])
    
    while N>0:
        # print(allocation)
        if np.random.uniform() < eps:
            i=int(np.random.choice(db.k, 1).item())
        else:
            i=np.argmax(allocation)
        sampled[i]+=1
        statistics, predicates = db.sample(1, i)
        eps-=step_size
        # pbar.update(1)
        resovoir[i][0] = np.concatenate([statistics, resovoir[i][0]])
        resovoir[i][1] = np.concatenate([predicates, resovoir[i][1]])
        N-=1
        if sample_reuse:
            statistics = resovoir[i][0]
            predicates = resovoir[i][1]
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 0:
            m[i] = np.mean(statistics[predicates])
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
        # updating allocation using strata formula
        weights = np.sqrt(p) * s
        norm = 1 if np.sum(weights) == 0 else np.sum(weights)
        weights = weights / norm
        allocation = np.floor(N * weights).astype(np.float32)
        # print("mean: {}, pred_prabab: {}".format(m, p))
    # print(sampled)
    # pbar.close()

    weights = np.sqrt(p) * s
    norm = 1 if np.sum(weights) == 0 else np.sum(weights)
    weights = weights / norm
    allocation = np.floor(n2 * weights).astype(np.int64)
    # print(allocation)
    sampled=allocation.astype(np.float64)
    m = np.zeros(db.k)
    for i in range(db.k):
        statistics, predicates = db.sample(allocation[i], i)
        resovoir[i][0] = np.concatenate([statistics, resovoir[i][0]])
        resovoir[i][1] = np.concatenate([predicates, resovoir[i][1]])
        # if sample_reuse:
        #     statistics = resovoir[i][0]
        #     predicates = resovoir[i][1]
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 0:
            m[i] = np.mean(statistics[predicates])

    norm = 1 if p.sum() == 0 else p.sum()
    if return_samples:
        return np.sum(m * p / norm), resovoir
    else:
        return np.sum(m * p / norm), sampled/(n2)



def _execute_ours_importance(db, n1, n2, return_samples=False, sample_reuse=True):
    sample_idxs, statistics, predicates=db.sample_kde(n1*db.k + n2)
    # print("got samples")
    # Q_xis=np.array([])
    P_tilda_xis=predicates
    
    # # splitted_idxs=np.array_split(sample_idxs,len(sample_idxs)/10)
    # for i in tqdm(sample_idxs):
    #     # print()
    #     print(db.kde.score_samples( db.proxy_scores_sorted[10:15].reshape(-1,1) ))
    
    # Q_xis=np.exp(db.density_estimates[sample_idxs])
    
    
    # splitted_idxs=np.array_split(sample_idxs,64)
    # print(len(splitted_idxs))
    # for i in tqdm(splitted_idxs):
    #     # print(db.proxy_scores_sorted[i].reshape(-1,1))
    #     Q_xis=np.concatenate((Q_xis,np.exp(db.kde.score_samples(db.proxy_scores_sorted[i].reshape(-1,1)))), axis=0)
    #     # print(Q_xis.shape)
    # print("got Qs")
    # w_tilda=P_tilda_xis/Q_xis
    w_tilda=predicates
    Z=np.sum(w_tilda)
    return np.sum(statistics*w_tilda)/Z, np.array([0,0,0,0,0])


def _execute_ours_ucb(db, n1, n2, return_samples=False, sample_reuse=True):
    p = np.zeros(db.k)
    s = np.zeros(db.k)
    resovoir = []
    
    for i in range(db.k):
        statistics, predicates = db.sample(1, i)
        resovoir.append([statistics, predicates])
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)

    N=n2+(n1-1)*db.k
    weights = np.sqrt(p) * s
    norm = 1 if np.sum(weights) == 0 else np.max(weights)
    weights = weights / norm
    allocation = np.floor(n2 * weights).astype(np.int64)
    
    m = np.zeros(db.k)      
    # ucb_arms = [0 for i in range(db.k)]
    # rewards=[0 for i in range(db.k)]
    sampled=np.array([1.0 for i in range(db.k)])

    strata_idxs=np.arange(0,db.k,1)
    for _ in range(N):
        # cdf = np.cumsum(weights)
        # index = np.searchsorted(cdf, random_uniforms[i], side='left')
        while True:
            probs=weights[strata_idxs]
            norm=1 if np.sum(probs) == 0 else np.sum(probs)
            probs/=norm
            if np.sum(probs)==0:
                i=np.random.choice(strata_idxs)
            else:
                i=np.random.choice(strata_idxs, p=probs)
            if len(db.strata[i])==0:
                id_=np.where(strata_idxs==i)[0][0]
                strata_idxs=np.delete(strata_idxs,id_)
            else:
                break

        
        sampled[i]+=1
        statistics, predicates = db.sample_adaptive(1, i)
        resovoir[i][0] = np.concatenate([statistics, resovoir[i][0]])
        resovoir[i][1] = np.concatenate([predicates, resovoir[i][1]])
        # N-=1
        if sample_reuse:
            statistics = resovoir[i][0]
            predicates = resovoir[i][1]
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 0:
            m[i] = np.mean(statistics[predicates])
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)
        
        # updating allocation using strata formula
        weights = p * s**2/(sampled**0.5)
        # norm = 1 if np.sum(weights) == 0 else np.max(weights)
        # weights = weights/(norm)
        weights += (2*np.log(np.sum(sampled)+1)/sampled)**0.5
        norm = 1 if np.sum(weights) == 0 else np.sum(weights)
        weights = weights / norm
        # allocation = np.floor(N * weights).astype(np.int64)

    norm = 1 if p.sum() == 0 else p.sum()
    if return_samples:
        return np.sum(m * p / norm), resovoir
    else:
        return np.sum(m * p / norm), sampled/(n2+n1*db.k)


def _execute_uniform(db, n1, n2, return_samples=False):
    n = (n1 * db.k) + n2
    statistics, predicates = db.sample(n)
    resovoir = [statistics, predicates]
    m = np.mean(statistics[predicates]) if np.sum(predicates) > 0 else 0
    if return_samples:
        return m, resovoir
    else:
        return m

@ray.remote(num_returns=2)
def execute_ours_adaptive(db, n1, n2, trials=5, sample_reuse=True):
    estimates = np.zeros(trials)
    allocations=np.array([[0.0 for _ in range(db.k)] for _ in range(trials)])
    for trial in tqdm(range(trials)):
        db_to_use=deepcopy(db)
        estimates[trial],allocations[trial] = _execute_ours_adaptive(
            db_to_use, n1, n2, sample_reuse=sample_reuse
        )
    return estimates, allocations

@ray.remote(num_returns=2)
def execute_ours_adaptive_pattern(db, n1, n2, trials=5, sample_reuse=True):
    estimates = np.zeros(trials)
    allocations=np.array([[0.0 for _ in range(db.k)] for _ in range(trials)])
    for trial in tqdm(range(trials)):
        db_to_use=deepcopy(db)
        estimates[trial],allocations[trial] = _execute_ours_adaptive_pattern(
            db_to_use, n1, n2, sample_reuse=sample_reuse
        )
    return estimates, allocations

@ray.remote(num_returns=2)
def execute_ours_ucb(db, n1, n2, trials=5, sample_reuse=True):
    estimates = np.zeros(trials)
    allocations=np.array([[0.0 for _ in range(db.k)] for _ in range(trials)])
    for trial in tqdm(range(trials)):
        db_to_use=deepcopy(db)
        estimates[trial],allocations[trial] = _execute_ours_ucb(
            db_to_use, n1, n2, sample_reuse=sample_reuse
        )
    return estimates, allocations

@ray.remote(num_returns=2)
def execute_ours(db, n1, n2, trials=5, sample_reuse=True):
    estimates = np.zeros(trials)
    allocations=np.array([[0.0 for _ in range(db.k)] for _ in range(trials)])
    for trial in tqdm(range(trials)):
        # db_to_use=deepcopy(db)
        estimates[trial],allocations[trial] = _execute_ours(
            db, n1, n2, sample_reuse=sample_reuse
        )
    return estimates, allocations


def execute_ours_importance(db, n1, n2, trials=5):
    estimates = np.zeros(trials)
    # allocations=np.array([[0.0 for _ in range(db.k)] for _ in range(trials)])
    for trial in tqdm(range(trials)):
        estimates[trial],_ = _execute_ours_importance(
            db, n1, n2
        )

    return estimates


@ray.remote
def execute_uniform(db, n1, n2, trials=5, sample_reuse=True):
    estimates = np.zeros(trials)
    for trial in range(trials):
        estimates[trial] = _execute_uniform(
            db, n1, n2
        )
    return estimates

@ray.remote
def execute_ours_with_ci(db, n1, n2, trials=5, bootstrap_trials=1000, confidence=0.95, sample_reuse=True):
    estimates = np.zeros(trials)
    lower_bounds = np.zeros(trials)
    upper_bounds = np.zeros(trials)
    
    for trial in range(trials):
        resamples = np.zeros(bootstrap_trials)
        estimate, resovoir = _execute_ours(
            db, n1, n2, return_samples=True, sample_reuse=sample_reuse
        )
        
        for b in range(bootstrap_trials):
            p = np.zeros(db.k)
            s = np.zeros(db.k)
            bootstrap_resovoir = []
            for j in range(db.k):
                idxs = np.arange(resovoir[j][0].shape[0])
                sample_idxs = np.random.choice(idxs, n1, replace=True)
                statistics = resovoir[j][0][sample_idxs]
                predicates = resovoir[j][1][sample_idxs]
                bootstrap_resovoir.append([statistics, predicates])
                if len(predicates) > 0:
                    p[j] = np.mean(predicates)
                if np.sum(predicates) > 1:
                    s[j] = np.std(statistics[predicates], ddof=1)
            
            weights = np.sqrt(p) * s
            norm = 1 if np.sum(weights) == 0 else np.sum(weights)
            weights = weights / norm
            allocation = np.floor(n2 * weights).astype(np.int64)

            m = np.zeros(db.k)
            for j in range(db.k):
                idxs = np.arange(resovoir[j][0].shape[0])
                if len(idxs) != 0:
                    sample_idxs = np.random.choice(idxs, allocation[j], replace=True)
                    statistics = resovoir[j][0][sample_idxs]
                    statistics = np.concatenate([statistics, bootstrap_resovoir[j][0]])
                    predicates = resovoir[j][1][sample_idxs]
                    predicates = np.concatenate([predicates, bootstrap_resovoir[j][1]])
                else:
                    statistics = bootstrap_resovoir[j][0]
                    predicates = bootstrap_resovoir[j][1]
                if len(predicates) > 0:
                    p[j] = np.mean(predicates)
                if np.sum(predicates) > 0:
                    m[j] = np.mean(statistics[predicates])
                    
            norm = 1 if p.sum() == 0 else p.sum()
            resamples[b] = np.sum(m * p / norm)

        lower_percentile = ((1 - confidence) / 2) * 100
        lower_bound = np.percentile(resamples, lower_percentile)
        upper_percentile = 100 - lower_percentile
        upper_bound = np.percentile(resamples, upper_percentile)
        
        estimates[trial] = estimate
        lower_bounds[trial] = lower_bound
        upper_bounds[trial] = upper_bound
        
    return estimates, lower_bounds, upper_bounds


@ray.remote
def execute_ours_adaptive_with_ci(db, n1, n2, trials=1, bootstrap_trials=10000, confidence=0.95, sample_reuse=True):
    estimates = np.zeros(trials)
    lower_bounds = np.zeros(trials)
    upper_bounds = np.zeros(trials)
    
    for trial in range(trials):
        resamples = np.zeros(bootstrap_trials)
        db_cp=deepcopy(db)
        estimate, resovoir = _execute_ours_adaptive(
            db_cp, n1, n2, return_samples=True, sample_reuse=sample_reuse
        )
        
        for b in tqdm(range(bootstrap_trials),desc="bootstrap_trials"):
            p = np.zeros(db.k)
            s = np.zeros(db.k)
            m = np.zeros(db.k)
            bootstrap_resovoir = []
            for j in range(db.k):
                idxs = np.arange(resovoir[j][0].shape[0])
                sample_idxs = np.random.choice(idxs, len(idxs), replace=True)
                statistics = resovoir[j][0][sample_idxs]
                predicates = resovoir[j][1][sample_idxs]
                bootstrap_resovoir.append([statistics, predicates])
                if len(predicates) > 0:
                    p[j] = np.mean(predicates)
                if np.sum(predicates) > 1:
                    s[j] = np.std(statistics[predicates], ddof=1)
            
            # weights = np.sqrt(p) * s
            # norm = 1 if np.sum(weights) == 0 else np.sum(weights)
            # weights = weights / norm
            # allocation = np.floor(n2 * weights).astype(np.int64)

            
            # for j in range(db.k):
                # idxs = np.arange(resovoir[j][0].shape[0])
                # if len(idxs) != 0:
                #     sample_idxs = np.random.choice(idxs, allocation[j], replace=True)
                #     statistics = resovoir[j][0][sample_idxs]
                #     statistics = np.concatenate([statistics, bootstrap_resovoir[j][0]])
                #     predicates = resovoir[j][1][sample_idxs]
                #     predicates = np.concatenate([predicates, bootstrap_resovoir[j][1]])
                # else:
                #     statistics = bootstrap_resovoir[j][0]
                #     predicates = bootstrap_resovoir[j][1]
                # if len(predicates) > 0:
                #     p[j] = np.mean(predicates)
                if np.sum(predicates) > 0:
                    m[j] = np.mean(statistics[predicates])
                    
            norm = 1 if p.sum() == 0 else p.sum()
            resamples[b] = np.sum(m * p / norm)

        lower_percentile = ((1 - confidence) / 2) * 100
        lower_bound = np.percentile(resamples, lower_percentile)
        upper_percentile = 100 - lower_percentile
        upper_bound = np.percentile(resamples, upper_percentile)
        # print(resamples)
        # print(lower_percentile,lower_bound,upper_percentile,upper_bound)


        estimates[trial] = estimate
        lower_bounds[trial] = lower_bound
        upper_bounds[trial] = upper_bound
        
    return estimates, lower_bounds, upper_bounds


@ray.remote
def execute_uniform_with_ci(db, n1, n2, trials=5, confidence=0.95):
    estimates = np.zeros(trials)
    lower_bounds = np.zeros(trials)
    upper_bounds = np.zeros(trials)
    
    for trial in range(trials):
        estimate, resovoir = _execute_uniform(db, n1, n2, return_samples=True)
        statistics = resovoir[0]
        predicates = resovoir[1]
        s = np.std(statistics[predicates], ddof=1) if np.sum(predicates) > 1 else 0
        n = np.sum(predicates)
        halfwidth = abs(st.norm.ppf((1 - confidence) / 2) * s / np.sqrt(n))
        
        estimates[trial] = estimate
        lower_bounds[trial] = estimate - halfwidth
        upper_bounds[trial] = estimate + halfwidth
        
    return estimates, lower_bounds, upper_bounds


def predict_performance(db):
    p = np.zeros(db.k)
    s = np.zeros(db.k)
    strata = np.array_split(np.arange(db.sort.shape[0]), db.k)
    
    for i in range(db.k):
        stratum = strata[i]
        statistics = db.statistics_sorted[stratum].copy()
        predicates = db.predicates_sorted[stratum].copy()
        if len(predicates) > 0:
            p[i] = np.mean(predicates)
        if np.sum(predicates) > 1:
            s[i] = np.std(statistics[predicates], ddof=1)

    return (np.sum(np.sqrt(p) * s) ** 2) / (np.sum(p) ** 2)


def _execute_group_by_ours_v2(dbs, n1, n2, version="optimal"):
    ps = np.zeros([len(dbs), dbs[0].k])
    sigmas = np.zeros([len(dbs), dbs[0].k])
    
    s1_samples = []
    for g in range(len(dbs)):
        temp = []
        for k in range(dbs[0].k):
            statistics, predicates = dbs[g].sample(n1, k)
            temp.append([statistics.copy(), predicates.copy()])
            if len(predicates) > 0:
                ps[g, k] = np.sum(predicates) / len(predicates)
            if np.sum(predicates) > 1:
                sigmas[g, k] = np.std(statistics[predicates], ddof=1)
        s1_samples.append(temp)
    
    cs = np.zeros([len(dbs)])
    for g in range(len(dbs)):
        norm = 1 if ps[g].sum() == 0 else ps[g].sum()
        cs[g] = ((np.sqrt(ps[g]) * sigmas[g]).sum() ** 2) / (norm ** 2)
    cs = np.eye(len(dbs)) * cs
    
    if version == "optimal":
        def objective(x):
            return max(cs @ (1 / x))

        def constraint(x):
            return np.sum(x) - 1
        
        bounds = [(0, 1)]*len(dbs)
        guess = np.array([1]*len(dbs)) / len(dbs)

        @my.penalty.quadratic_inequality(constraint)
        def penalty(x):
            return 0.0
    
        solver = my.solvers.fmin
        mon = my.monitors.Null()
        kwds = dict(disp=False, full_output=True, itermon=mon,
                    args=(),  xtol=1e-8, ftol=1e-8, maxfun=10000, maxiter=5000)
        result = solver(objective, guess, bounds=bounds, penalty=penalty, **kwds)
        meta_weights = result[0]
        meta_weights /= meta_weights.sum()
    else:
        meta_weights = np.array([1]*len(dbs)) / len(dbs)
        
    mus = np.zeros([len(dbs), dbs[0].k])
    for g in range(len(dbs)):
        weights = np.sqrt(ps[g]) * sigmas[g]
        weights = np.nan_to_num(weights) 
        weights[weights < 0] = 0
        weights /= norm
        
        allocation = (meta_weights[g] * n2 * weights).astype(np.int32)
        allocation[allocation < 0] = 0
        
        for k in range(dbs[0].k):
            statistics, predicates = dbs[g].sample(allocation[k], k)
            statistics = np.concatenate([statistics, s1_samples[g][k][0]])
            predicates = np.concatenate([predicates, s1_samples[g][k][1]])
            if len(predicates) > 0 and predicates.sum() > 0:
                mus[g, k] = statistics[predicates].mean()
                ps[g, k] = predicates.sum() / len(predicates)

    norm = ps.sum(axis=1)
    return (mus * (ps.T / norm.T).T).sum(axis=1)

@ray.remote
def execute_group_by_ours_v2(dbs, n1, n2, trials=5, version="optimal"):
    estimates = np.zeros([trials, len(dbs)])
    for trial in range(trials):
        estimates[trial] = _execute_group_by_ours_v2(dbs, n1, n2, version)
    return estimates


def _execute_group_by_uniform_v2(dbs, n1, n2):
    n = len(dbs)*dbs[0].k*n1 + n2
    answers = np.zeros(len(dbs))
    for idx, db in enumerate(dbs):
        statistics, predicates = db.sample(n // len(dbs))
        answers[idx] = statistics[predicates].mean()
    return answers

@ray.remote
def execute_group_by_uniform_v2(dbs, n1, n2, trials):
    estimates = np.zeros([trials, len(dbs)])
    for trial in range(trials):
        estimates[trial] = _execute_group_by_uniform_v2(dbs, n1, n2)
    return estimates