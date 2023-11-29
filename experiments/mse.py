import ray
import pandas as pd
import scipy.optimize
import abae
import os
import numpy as np
from tqdm.autonotebook import tqdm
from copy import deepcopy

ray.init(num_cpus=int(0.75*os.cpu_count()), object_store_memory= 10* 1024 * 1024 * 1024)
K = 5
C = 0.5
TRIALS = 100

dbs = [
    abae.JacksonRecords(k=K),
    # abae.TaipeiRecords(k=K),
    #  abae.CelebARecords(k=K),
    # abae.MovieFacesV2Records(k=K)
    # abae.Trec05PRecords(k=K),
    # abae.AmazonOfficeSuppliesRecords(k=K)
]
# methods=["ada", "abae", "ucb"]
methods=["ada_ci"]
for db in tqdm(dbs):
    ns = np.arange(500, 5500, 500)
    n1s = (ns*C/db.k).astype(np.int64)
    n2s = (ns*(1-C)).astype(np.int64)
    for method in methods:
            
        if method=="ada":
            ours = [None]*len(ns)
            ours_allocations=[None]*len(ns)
            db_ours=deepcopy(db)
            for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
                db_ours=deepcopy(db)
                ours[idx], ours_allocations[idx]= abae.execute_ours_adaptive.remote(db_ours, n1, n2, trials=TRIALS, sample_reuse=True) 
            ours = np.array(ray.get(ours))
            ours_allocations = np.array(ray.get(ours_allocations))
            print(ours_allocations.shape)
            ours_allocations=np.mean(ours_allocations,axis=1)
            print(ours_allocations.shape)
            print("our allocation:", ours_allocations)
            weights = np.sqrt(db.ps) * db.sigmas
            norm = 1 if np.sum(weights) == 0 else np.sum(weights)
            weights = weights / norm
            allocation = np.array([weights for i in range(len(ns))])
            print("optimal allocation:", allocation)
            print("diff:", np.linalg.norm(allocation-ours_allocations,ord=1,axis=1))

        elif method=="ada_ci":
            ours = [None]*len(ns)
            # ours_allocations=[None]*len(ns)
            lb=[None]*len(ns)
            ub = [None]*len(ns)
            db_ours=deepcopy(db)
            for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
                db_ours=deepcopy(db)
                a= abae.execute_ours_adaptive_with_ci.remote(db_ours, n1, n2, trials=TRIALS, sample_reuse=True)
                a=ray.get(a)
                ours[idx], lb[idx], ub[idx]=a 
            # ours = np.array(ray.get(ours))
            # lb = np.array(ray.get(lb))
            # ub = np.array(ray.get(ub))
            # ours_allocations = np.array(ray.get(ours_allocations))
            # print(ours_allocations.shape)
            # ours_allocations=np.mean(ours_allocations,axis=1)
            # print(ours_allocations.shape)
            # print("our allocation:", ours_allocations)
            # weights = np.sqrt(db.ps) * db.sigmas
            # norm = 1 if np.sum(weights) == 0 else np.sum(weights)
            # weights = weights / norm
            # allocation = np.array([weights for i in range(len(ns))])
            # print("optimal allocation:", allocation)
            # print("diff:", np.linalg.norm(allocation-ours_allocations,ord=1,axis=1))

        elif method=="cv":
            ours = [None]*len(ns)
            ours_allocations=[None]*len(ns)
            db_ours=deepcopy(db)
            for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
                db_ours=deepcopy(db)
                ours[idx], ours_allocations[idx]= abae.execute_ours_adaptive.remote(db_ours, n1, n2, trials=TRIALS, sample_reuse=True) 
            ours = np.array(ray.get(ours))
            ours_allocations = np.array(ray.get(ours_allocations))
            print(ours_allocations.shape)
            ours_allocations=np.mean(ours_allocations,axis=1)
            print(ours_allocations.shape)
            print("our allocation:", ours_allocations)
            weights = np.sqrt(db.ps) * db.sigmas
            norm = 1 if np.sum(weights) == 0 else np.sum(weights)
            weights = weights / norm
            allocation = np.array([weights for i in range(len(ns))])
            print("optimal allocation:", allocation)
            print("diff:", np.linalg.norm(allocation-ours_allocations,ord=1,axis=1))
        elif method=="ada_pattern":
            ours = [None]*len(ns)
            ours_allocations=[None]*len(ns)
            db_ours=deepcopy(db)
            for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
                db_ours=deepcopy(db)
                ours[idx], ours_allocations[idx]= abae.execute_ours_adaptive_pattern.remote(db_ours, n1, n2, trials=TRIALS, sample_reuse=True) 
            ours = np.array(ray.get(ours))
            ours_allocations = np.array(ray.get(ours_allocations))
            print(ours_allocations.shape)
            ours_allocations=np.mean(ours_allocations,axis=1)
            print(ours_allocations.shape)
            print("our allocation:", ours_allocations)
            weights = np.sqrt(db.ps) * db.sigmas
            norm = 1 if np.sum(weights) == 0 else np.sum(weights)
            weights = weights / norm
            allocation = np.array([weights for i in range(len(ns))])
            print("optimal allocation:", allocation)
            print("diff:", np.linalg.norm(allocation-ours_allocations,ord=1,axis=1))
        elif method=="ucb":
            ours = [None]*len(ns)
            ours_allocations=[None]*len(ns)
            db_ours=deepcopy(db)
            for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
                db_ours=deepcopy(db)
                ours[idx], ours_allocations[idx]= abae.execute_ours_ucb.remote(db_ours, n1, n2, trials=TRIALS, sample_reuse=True) 
            ours = np.array(ray.get(ours))
            ours_allocations = np.array(ray.get(ours_allocations))
            print(ours_allocations.shape)
            ours_allocations=np.mean(ours_allocations,axis=1)
            print(ours_allocations.shape)
            print("our allocation:", ours_allocations)
            weights = np.sqrt(db.ps) * db.sigmas
            norm = 1 if np.sum(weights) == 0 else np.sum(weights)
            weights = weights / norm
            allocation = np.array([weights for i in range(len(ns))])
            print("optimal allocation:", allocation)
            print("diff:", np.linalg.norm(allocation-ours_allocations,ord=1,axis=1))
        elif method=="abae":
            ours = [None]*len(ns)
            ours_allocations=[None]*len(ns)
            # db_ours=deepcopy(db)
            for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
                # db_ours=deepcopy(db)
                ours[idx], ours_allocations[idx]= abae.execute_ours.remote(db, n1, n2, trials=TRIALS, sample_reuse=True) 
            ours = np.array(ray.get(ours))
            ours_allocations = np.array(ray.get(ours_allocations))
            print(ours_allocations.shape)
            ours_allocations=np.mean(ours_allocations,axis=1)
            print(ours_allocations.shape)
            print("our allocation:", ours_allocations)
            weights = np.sqrt(db.ps) * db.sigmas
            norm = 1 if np.sum(weights) == 0 else np.sum(weights)
            weights = weights / norm
            allocation = np.array([weights for i in range(len(ns))])
            print("optimal allocation:", allocation)
            print("diff:", np.linalg.norm(allocation-ours_allocations,ord=1,axis=1))
        elif method == "imp":
            ours=[None]*(len(ns))
            for idx, (n1, n2) in tqdm(enumerate(zip(n1s, n2s))):
                ours[idx]= abae.execute_ours_importance(db, n1, n2, trials=TRIALS) 
            ours = np.array(ours)
        
        else:
            print("method not implemented")
            exit(0)
        uniform = [None]*len(ns)
        for idx, (n1, n2) in enumerate(zip(n1s, n2s)):
            uniform[idx] = abae.execute_uniform.remote(db, n1, n2, trials=TRIALS) 
        uniform = np.array(ray.get(uniform))
        
        
        results = {}
        results[str({"K": K, "C": C, "TRIALS": TRIALS})] = [""]*len(uniform[0])
        results[f"truth"] = [db.ground_truth]*len(uniform[0])
        for i in range(len(ns)):
            results[f"uniform_{ns[i]}"] = uniform[i]
        for i in range(len(ns)):
            results[f"ours_{ns[i]}"] = ours[i]
        if "_ci" in method:
                
            for i in range(len(ns)):
                results[f"lb_{ns[i]}"] = lb[i]
            for i in range(len(ns)):
                results[f"ub_{ns[i]}"] = ub[i]

        df = pd.DataFrame.from_dict(results)
        df.to_csv(f"./results/mse/{db.name}_mse_{method}.csv")
    
ray.shutdown()
