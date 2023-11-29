# PABae

# Requirements

### pip install

Install the requirements with `pip install ray scipy mystic numpy pandas tabulate tqdm matplotlib`. 


### conda install

Run `conda env create -f env.yml`.

# Reproducing Experiments

Note that we have precomputed all the oracle labels.

The sampling routines are present in `pabae/algorithms.py`

We run `experiments/mse.py` for all our experiments.

The image, text, and video folders contain our scripts used for multiple tasks including but not limited to exploratory analysis, specialised model training, computing proxies etc.

In case of queries, please contact authors [Ashutosh Sharma](https://github.com/ashutoshuiuc), [Aakriti](https://github.com/Aakriti28) and [Shashwat Jaiswal](https://github.com/shashwatj07).

Special thanks to the authors of [ABae](https://arxiv.org/abs/2108.06313) for the basic framework and overall idea. 
