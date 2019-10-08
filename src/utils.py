"""
simulate_scr.py

Simulation code for conducting spatial capture-recapture experiments in silico.

"""
import argparse
import cProfile
import csv
from functools import partial
import itertools
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import rasterio
from scipy import stats
from scipy.optimize import minimize
# from scipy.special import binom
from skimage import graph
import sys
import time
import tqdm
# from rasterio.plot import show
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr

def find_lcp(raster, alpha2, raster_cell_size=1):
    """
    Computes least cost path lengths between every pair of cells in a raster.
    """
    def get_cell_cost(x):
        return np.exp(x*alpha2)
    get_all_cell_costs = np.vectorize(get_cell_cost)
    cost_raster = get_all_cell_costs(raster)
    lcp_graph = graph.MCP_Geometric(cost_raster, fully_connected=True)
    
    lcp_distances = np.zeros((raster.shape[0], raster.shape[1], raster.shape[0], raster.shape[1]))
    # pool = mp.Pool(mp.cpu_count())
    for x_start in range(raster.shape[0]):
        # results = pool.starmap(lcp_graph.find_costs, [(x_start, y_start) for y_start in range(raster.shape[1])])
        # print(len(results))
        # assert 2==3, 'break break'
        for y_start in range(raster.shape[1]):
            distances = lcp_graph.find_costs([(x_start, y_start)])[0]
            for x_end in range(raster.shape[0]):
                for y_end in range(raster.shape[1]):
                    lcp_distances[x_start, y_start, x_end, y_end] = distances[x_end, y_end]*raster_cell_size
    return lcp_distances

def find_lcp_to_pts(raster, alpha2, ref_pts, raster_cell_size=1):
    """
    Computes least cost path lengths from every cell in a raster to each of a list of reference points.
    """
    def get_cell_cost(x):
        return np.exp(x*alpha2)
    get_all_cell_costs = np.vectorize(get_cell_cost)
    cost_raster = get_all_cell_costs(raster)
    lcp_graph = graph.MCP_Geometric(cost_raster, fully_connected=True)

    # distances_temp = lcp_graph.find_costs([[0,0]])[0]
    # print(distances_temp[0,1]*0.1)
    # cells = []
    # dist_vals = []
    # for x in range(raster.shape[0]):
    #     for y in range(raster.shape[1]):
    #         if distances_temp[x,y]*0.1 < 0.5:
    #             dist_vals.append(distances_temp[x,y]*0.1)
    #             cells.append((x,y))
    # print(dist_vals)
    # print(cells)
    # assert 1==3, 'break, just break'

    lcp_distances = np.zeros((len(ref_pts), raster.shape[0], raster.shape[1]))
    for rpidx in range(len(ref_pts)):
        rp = ref_pts[rpidx]
        rp_int = (int(np.floor(rp[0])), int(np.floor(rp[1])))
        distances = lcp_graph.find_costs([rp_int])[0]
        # print(distances[0, 0]*raster_cell_size)
        lcp_distances[rpidx, ...] = distances*raster_cell_size

    # assert 2==3, 'break here checking lcp dists are same'
    return lcp_distances

# def simulate_spatial_capture_recapture_old(ac, tl, K, lcp_dist, alpha0, alpha1):
#     n_individuals = len(ac)
#     n_traps = len(tl)
#     detections = np.ndarray((n_individuals, n_traps))
#     for ni in range(len(ac)):
#         s = ac[ni]
#         sx = int(np.floor(s[0]))
#         sy = int(np.floor(s[1]))
#         # dists = []
#         for nt in range(len(tl)):
#             t = tl[nt]
#             tx = int(np.floor(t[0]))
#             ty = int(np.floor(t[1]))
#             dist = lcp_dist[sx, sy, tx, ty]
#             # dists.append(dist)
#             prob_cap = (1/(1+np.exp(alpha0)))*np.exp(-alpha1*dist*dist)
#             n_det = np.random.binomial(K, prob_cap)
#             detections[ni, nt] = n_det
#         # print(dists)
        
#     detections = detections[~np.all(detections==0, axis=1)]    
#     return(detections)

# def binom_logpmf(N, k, p):
#     prob = binom(int(N), int(k))*math.pow(p,k)*math.pow((1-p),(N-k))
#     # binom_rv = stats.binom(N,p)
#     # assert prob == binom_rv.pmf(k), (prob, binom_rv.pmf(k))
#     if prob == 0: # log(0) = -inf, which will make any sum of logs = -inf
#         return -sys.maxsize - 1
#     return np.log(prob)

# def scr_fit_function_old(params, *args):
#     alpha0 = params[0]
#     alpha1 = params[1]
#     alpha2 = params[2]
#     n0 = params[3]

#     raster = args[0]
#     raster_cell_size = args[1]
#     tl = args[2]
#     cap_hists = args[3]
#     K = args[4]

#     ## estimate the least cost path distances using current parameter values
#     timer = time.time()
#     est_lcp_distances = find_lcp(raster, alpha2, raster_cell_size=raster_cell_size)
#     # print('LCP COMPUTATION: %0.4f'%(time.time() - timer))

#     ## compute the capture probability p(i, x, y) at each trap i of individuals from pixel (x,y)
#     timer = time.time()
#     est_prob_cap = np.zeros((len(tl), raster.shape[0], raster.shape[1]))
#     for nt in range(len(tl)):
#         t = tl[nt]
#         tx = int(np.floor(t[0]))
#         ty = int(np.floor(t[1]))
#         for cx in range(raster.shape[0]):
#             for cy in range(raster.shape[1]):
#                 dist = est_lcp_distances[tx, ty, cx, cy]
#                 est_prob_cap[nt, cx, cy] = (1/(1+np.exp(alpha0)))*np.exp(-alpha1*dist*dist)
#     # print('CAPTURE PROBABILITY COMPUTATION: %0.4f'%(time.time() - timer))

#     ## compute conditional likelihood of each individual's capture history + undetected individuals' capture history
#     timer = time.time()
#     print('\tComputing conditional likelihoods for each individual capture history')
#     conditional_likelihoods = np.ndarray((len(cap_hists)+1, raster.shape[0], raster.shape[1]))
#     for ni in tqdm.tqdm(range(len(cap_hists))):
#         i_cap_hist = cap_hists[ni]
#         # timer = time.time()
#         tiled_i_cap_hist = np.tile(np.expand_dims(np.tile(np.expand_dims(i_cap_hist, axis=1), (1,40)), axis=2), (1,40))
#         probs = stats.binom.pmf(tiled_i_cap_hist, K, est_prob_cap)
#         zero_mask = probs == 0.0
#         log_probs = np.log(probs, where=np.invert(zero_mask))
#         log_probs[zero_mask] = -sys.maxsize - 1
#         log_cond_lik_sums = np.sum(log_probs, axis=0)
#         conditional_likelihoods[ni,...] = np.exp(log_cond_lik_sums)
#     i_cap_hist = np.squeeze(np.zeros((len(tl), 1)))
#     tiled_i_cap_hist = np.tile(np.expand_dims(np.tile(np.expand_dims(i_cap_hist, axis=1), (1,40)), axis=2), (1,40))
#     probs = stats.binom.pmf(tiled_i_cap_hist, K, est_prob_cap)
#     zero_mask = probs == 0.0
#     log_probs = np.log(probs, where=np.invert(zero_mask))
#     log_probs[zero_mask] = -sys.maxsize - 1
#     log_cond_lik_sum = np.sum(log_probs, axis=0)
#     conditional_likelihoods[-1, ...] = np.exp(log_cond_lik_sum)
#     # print('CONDITIONAL LIKELIHOOD COMPUTATION: %0.4f'%(time.time() - timer))

#     ## compute marginal likelihood of each individual's capture history + undetected individuals' capture history
#     timer = time.time()
#     print('\tComputing marginal likelihood of each individual capture history')
#     marginal_likelihoods = np.ndarray((1, len(cap_hists)+1))
#     for ni in tqdm.tqdm(range(len(cap_hists))):
#         marginal_likelihoods[0, ni] = sum([(1/float(raster.size))*conditional_likelihoods[ni, cx, cy] for cx in range(raster.shape[0]) for cy in range(raster.shape[1])])
#     marginal_likelihoods[0, -1] = sum([(1/float(raster.size))*conditional_likelihoods[-1, cx, cy] for cx in range(raster.shape[0]) for cy in range(raster.shape[1])])
#     # print('MARGINAL LIKELIHOOD COMPUTATION: %0.4f'%(time.time() - timer))

#     ## compute log likelihood
#     timer = time.time()
#     nv = np.ones((1, len(cap_hists)+1))
#     nv[0, -1] = n0
#     part1 = math.lgamma(len(cap_hists)+n0+1) - math.lgamma(n0+1)
#     part2 = sum(np.multiply(nv[0,], np.log(marginal_likelihoods[0,])))
#     # print('LOG LIKELIHOOD COMPUTATION: %0.4f'%(time.time() - timer))
#     return(-1*(part1 + part2))

def compute_cond_lik_ind(est_prob_cap, K, num_traps, raster, ind_cap_hist):
    """
    Computes the likelihood of an individual's capture history conditional on their activity center location.
    """
    broadcast_i_cap_hist = np.broadcast_to(ind_cap_hist[..., np.newaxis, np.newaxis], (num_traps, raster.shape[0], raster.shape[1]))
    probs = stats.binom.pmf(broadcast_i_cap_hist, K, est_prob_cap)
    zero_mask = probs == 0.0
    log_probs = np.log(probs, where=np.invert(zero_mask))
    log_probs[zero_mask] = -sys.maxsize - 1
    log_cond_lik_sums = np.sum(log_probs, axis=0)
    return np.exp(log_cond_lik_sums)


def compute_scr_neg_log_likelihood(params, *args, parallelize=True):
    """
    Computes the log-likelihood of observed capture histories, given parameter values.
    """
    # timer = time.time()

    alpha0 = params[0]
    alpha1 = np.exp(params[1])  # pass log(alpha1) in as parameter
    alpha2 = np.exp(params[2])  # pass log(alpha2) in as parameter
    n0 = np.exp(params[3])      # pass log(n0) in as parameter

    raster = args[0]
    raster_cell_size = args[1]
    tl = args[2]
    cap_hists = args[3]
    K = args[4]

    num_traps = len(tl)
    num_detected_ind = len(cap_hists)
    
    # ------------------------------------------------------------------------------------------#
    # estimate the least cost path distances using current parameter values
    # ------------------------------------------------------------------------------------------#
    est_lcp_distances = find_lcp_to_pts(raster, alpha2, tl, raster_cell_size=raster_cell_size)

    # ------------------------------------------------------------------------------------------#
    # compute the capture probability p(i, x, y) at each trap i of individuals from pixel (x,y)
    # ------------------------------------------------------------------------------------------#
    est_prob_cap = (1/(1+np.exp(alpha0)))*np.exp(-alpha1*(est_lcp_distances**2))

    # ------------------------------------------------------------------------------------------#
    # compute conditional likelihood of each individual's capture history + undetected individuals' capture history
    # ------------------------------------------------------------------------------------------#    
    conditional_likelihoods = np.ndarray((num_detected_ind+1, raster.shape[0], raster.shape[1]))
    if parallelize:
        func = partial(compute_cond_lik_ind, est_prob_cap, K, num_traps, raster)
        pool = mp.Pool(min(mp.cpu_count(), 4))
        conditional_likelihoods[0:num_detected_ind,...] = np.array(pool.map(func, cap_hists))
        pool.close()
        pool.join()
        i_cap_hist = np.squeeze(np.zeros((len(tl), 1)))
        conditional_likelihoods[-1, ...] = compute_cond_lik_ind(est_prob_cap, K, num_traps, raster, i_cap_hist)
    else:
        for ni in range(num_detected_ind):
            i_cap_hist = cap_hists[ni]
            conditional_likelihoods[ni,...] = compute_cond_lik_ind(est_prob_cap, K, num_traps, raster, i_cap_hist)
        i_cap_hist = np.squeeze(np.zeros((len(tl), 1)))
        conditional_likelihoods[-1, ...] = compute_cond_lik_ind(est_prob_cap, K, num_traps, raster, i_cap_hist)        

    # ------------------------------------------------------------------------------------------#
    # compute marginal likelihood of each individual's capture history + undetected individuals' capture history
    # ------------------------------------------------------------------------------------------#
    marginal_likelihoods = np.ndarray((1, num_detected_ind+1))
    for ni in range(num_detected_ind):
        marginal_likelihoods[0, ni] = sum([(1/float(raster.size))*conditional_likelihoods[ni, cx, cy] for cx in range(raster.shape[0]) for cy in range(raster.shape[1])])
    marginal_likelihoods[0, -1] = sum([(1/float(raster.size))*conditional_likelihoods[-1, cx, cy] for cx in range(raster.shape[0]) for cy in range(raster.shape[1])])
    
    
    ## compute log likelihood
    nv = np.ones((1, len(cap_hists)+1))
    nv[0, -1] = n0
    part1 = math.lgamma(len(cap_hists)+n0+1) - math.lgamma(n0+1)
    part2 = sum(np.multiply(nv[0,], np.log(marginal_likelihoods[0,])))

    # print('Computing log likelihood took %0.4f seconds'%(time.time() - timer))
    
    return(-1*(part1 + part2))

def callback(x):
    print(x)
