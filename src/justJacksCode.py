from simulate_scr import *
from run_scr_estimation import *
import argparse
import csv
import itertools
# import multiprocessing as mp
import numpy as np
# import os
import pickle
import rasterio
# from scipy import stats
# from scipy.optimize import minimize
# from scipy.special import binom
# from skimage import graph
# import sys
import time
# import tqdm

from visualize import *
from utils import find_lcp_to_pts

def lambda_ipp_log_linear(x, beta0, beta1):
    """
    Rate function for an inhomogeneous poisson process, in which
    the intensity is a log-linear function of x.
    lambda = exp(beta0 + beta1*x)
    """
    return np.exp(beta0 + beta1*x)

def simulate_ipp(N, raster, n_real=1, seed=1111):
    """
    Simulates a spatial inhomogeneous Poisson process over 2D raster.
    Simulation is done using the Lewis and Schedler thinning algorithm.
    Args:
    N           - number of point events to simulate per realizition
    raster      - raster over which to simulate the IPP
    n_real      - number of realizations of the IPP to simulate
    seed        - random seed for the simulation
    Returns:
    points      - list of n_real lists; each sublist contains N tuples (x, y) specifying spatial
                  points for one realization of the IPP.
    """
    np.random.seed(seed)

    n_pixels = np.prod(raster.shape)
    beta0 = np.log(float(N)/n_pixels)
    beta1 = -3
    min_rast_val = np.amin(raster)
    max_rast_val = np.amax(raster)
    lambda_max = max(lambda_ipp_log_linear(min_rast_val, beta0, beta1), lambda_ipp_log_linear(max_rast_val, beta0, beta1))

    points = [] # list points in each realization of the IPP
    for rno in range(n_real):
        counter = 0
        rpoints = [] # list of points for a single IPP realization
        while counter < N:
            x_coord = np.random.uniform(0, raster.shape[0], 1)[0]
            y_coord = np.random.uniform(0, raster.shape[1], 1)[0]
            z_val = raster[int(np.floor(x_coord)), int(np.floor(y_coord))]
            lambda_xy = lambda_ipp_log_linear(z_val, beta0, beta1)
            coin = np.random.uniform(0, 1, 1)[0]
            if coin < lambda_xy/lambda_max:
                counter += 1
                rpoints.append((x_coord, y_coord))
        points.append(rpoints)
    return points

def simulate_capture_histories(ac, tl, K, lcp_dist, alpha0, alpha1, n_real=50, seed=1111):
    """
    Simulates a capture history for a simulated spatial capture-recapture study.
    Assumes multiple individuals can be detected at each detector in a given sampling
    occasion--but multiple detections of the same individual at the same trap in a
    single sampling occasion are indistinguishable.
    Args:
    ac          - list of tuples (x,y) storing activity center locations
    tl          - list of tuples (x,y) storing trap locations
    K           - number of sampling occasions
    lcp_dist    - least cost paths between each trap and every raster pixel
    alpha0      - capture probability parameter
    alpha1      - home range parameter
    n_real      - number of realizations of the capture history to simulate
    seed        - random seed for the simulation
    Returns:
    detections  - matrix of the number of times each individual was detected at
                  each detector
    """
    n_individuals = len(ac)
    n_traps = len(tl)
    detections = []
    for rno in range(n_real):
        rdetections = np.ndarray((n_individuals, n_traps))
        for ni in range(len(ac)):
            s = ac[ni]
            sx = int(np.floor(s[0]))
            sy = int(np.floor(s[1]))
            for nt in range(len(tl)):
                dist = lcp_dist[nt, sx, sy]
                prob_cap = (1/(1+np.exp(alpha0)))*np.exp(-alpha1*dist*dist)
                n_det = np.random.binomial(K, prob_cap)
                rdetections[ni, nt] = n_det
        rdetections = rdetections[~np.all(rdetections==0, axis=1)]
        detections.append(rdetections)
    return(detections)

def create_grid_layout(n, m, raster, buffer=0.1):
    n_pix_h, n_pix_w = raster.shape
    grid_xmin = n_pix_h*buffer
    grid_xmax = n_pix_h*(1-buffer)
    grid_ymin = n_pix_w*buffer
    grid_ymax = n_pix_w*(1-buffer)

    grid_x = np.linspace(grid_xmin, grid_xmax, n)
    grid_y = np.linspace(grid_ymin, grid_ymax, m)
    grid_loc = list(itertools.product(grid_x, grid_y))

    return grid_loc

from simulate_scr import *
from run_scr_estimation import *

def main():
    #from simulate_scr
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True, help='Path to configuration file for SCR simulation settings.')

    args = parser.parse_args()
    config_file_string = args.config_file.split('/')[-1].split('.csv')[0]

    # Read in configuration parameters
    config_params = {}
    with open(args.config_file, 'r') as cf:
        paramreader = csv.reader(cf)
        for line in paramreader:
            if line[0] in ['landscape_raster_file', 'trap_config']:
                config_params[line[0]] = line[1]
            elif line[0] in ['ALPHA0', 'ALPHA1', 'ALPHA2', 'raster_cell_size']:
                config_params[line[0]] = float(line[1])
            else:
                config_params[line[0]] = int(line[1])

    # Load landscape
    landscape_raster = rasterio.open(config_params['landscape_raster_file'])
    landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    landscape_raster_name = config_params['landscape_raster_file'].split('/')[-1].split('.tif')[0]

    # Simulate ground truth activity centers
    timer = time.time()
    ac_realizations = simulate_ipp(config_params['N'], landscape_ndarr, n_real=config_params['n_ac_realizations'],
                                   seed=config_params['activity_centers_seed'])
    print('Simulated activity centers for %d realizations in %0.2f seconds' % (
        config_params['n_ac_realizations'], time.time() - timer))
    for acridx in range(config_params['n_ac_realizations']):
        # visualize_activity_centers(landscape_ndarr, ac_realizations[acridx])
        with open('../data/simulation/activity_centers/' + config_file_string + '_' + str(acridx) + '.csv', 'w') as f:
            acwriter = csv.writer(f)
            for p in ac_realizations[acridx]:
                acwriter.writerow([p[0], p[1]])

    # Simulate grid of trap locations
    if config_params['trap_config'] == 'grid':
        timer = time.time()
        trap_loc = create_grid_layout(config_params['n_traps_x'], config_params['n_traps_y'], landscape_ndarr)
        print('Simulated trap locations in %0.2f seconds' % (time.time() - timer))
        with open('../data/simulation/trap_locations/' + landscape_raster_name + '_grid_' + str(
                config_params['n_traps_x']) + '_' + str(config_params['n_traps_y']) + '.csv', 'w') as f:
            tlwriter = csv.writer(f)
            for p in trap_loc:
                tlwriter.writerow([p[0], p[1]])
        # visualize_trap_layout(landscape_ndarr, trap_loc)

    # Simulate capture histories
    timer = time.time()
    lcp_distances = find_lcp_to_pts(landscape_ndarr, config_params['ALPHA2'], trap_loc,
                                    raster_cell_size=config_params['raster_cell_size'])
    print('Computed ground truth least cost paths to each trap in %0.2f seconds' % (time.time() - timer))
    timer = time.time()
    for acridx in range(config_params['n_ac_realizations']):
        capture_histories = simulate_capture_histories(ac_realizations[acridx], trap_loc, config_params['caphist_realization_no'], lcp_distances,
                                                       config_params['ALPHA0'], config_params['ALPHA1'],
                                                       seed=config_params['capture_histories_seed'])
        with open('../data/simulation/capture_histories/' + config_file_string + '_' + str(acridx) + '.pkl', 'wb') as f:
            pickle.dump(capture_histories, f)
    print('Simulated and saved spatial capture-recapture histories in %0.2f seconds' % (time.time() - timer))

    #from scr_estimation
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', type=str, required=True,
    #                     help='Path to configuration file for SCR simulation settings.')
    # parser.add_argument('--ac_realization_no', type=int, required=True,
    #                     help='Integer index for activity center realization number.')
    # parser.add_argument('--caphist_realization_no', type=int, required=True,
    #                     help='Integer index for capture history realization number.')
    # args = parser.parse_args()
    # config_file_name = args.config_file.split('/')[-1].split('.csv')[0]

    #
    # # Read in configuration parameters
    # config_params = {}
    # with open(args.config_file, 'r') as cf:
    #     paramreader = csv.reader(cf)
    #     for line in paramreader:
    #         if line[0] in ['landscape_raster_file', 'trap_config']:
    #             config_params[line[0]] = line[1]
    #         elif line[0] in ['ALPHA0', 'ALPHA1', 'ALPHA2', 'raster_cell_size']:
    #             config_params[line[0]] = float(line[1])
    #         else:
    #             config_params[line[0]] = int(line[1])

    # Load into variables
    log_file_name = '../logs/%s_%d_%d.log' % (config_file_string, config_params['n_ac_realizations'], config_params['caphist_realization_no'])
    landscape_raster_file = config_params['landscape_raster_file']
    landscape_raster_name = landscape_raster_file.split('/')[-1].split('.tif')[0]
    raster_cell_size = config_params['raster_cell_size']
    trap_loc_path = '../data/simulation/trap_locations/%s_%s_%d_%d.csv' % (
        landscape_raster_name, config_params['trap_config'], config_params['n_traps_x'], config_params['n_traps_y'])
    ALPHA0 = config_params['ALPHA0']
    ALPHA1 = config_params['ALPHA1']
    ALPHA2 = config_params['ALPHA2']
    N = config_params['N']
    K = config_params['caphist_realization_no']

    #sys.stdout = open(log_file_name, 'w')
    print(
        'Ground truth parameter values:\nALPHA0: %d\nALPHA1: %0.4f\nALPHA2: %0.2f\nN: %d' % (ALPHA0, ALPHA1, ALPHA2, N))

    landscape_raster = rasterio.open(landscape_raster_file)
    print('width', landscape_raster.width)
    print('height', landscape_raster.height)
    landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    trap_loc = [(eval(line[0]), eval(line[1])) for line in csv.reader(open(trap_loc_path, 'r'))]

    # # Compute least cost paths through this landscape
    # lcp_distances = find_lcp_to_pts(landscape_ndarr, ALPHA2, trap_loc, raster_cell_size=raster_cell_size)

    # Load capture history
    capture_history_path = '../data/simulation/capture_histories/%s_%d.pkl' % (config_file_string, config_params['n_ac_realizations']-1)
    capture_histories = pickle.load(open(capture_history_path, 'rb'))
    detections = capture_histories[config_params['caphist_realization_no']]

    # Minimize
    param_init = [ALPHA0, np.log(ALPHA1), np.log(ALPHA2), np.log(N - len(detections))]
    print('Initial parameter values:\nalpha0: %d\nalpha1: %0.4f\nalpha2: %0.2f\nN: %d' % (ALPHA0, ALPHA1, ALPHA2, N))
    scr_args = (landscape_ndarr, raster_cell_size, trap_loc, detections, 10)

    initial_neg_log_likelihood = compute_scr_neg_log_likelihood(param_init, *scr_args)
    print('Initial negative log likelihood: %0.6f' % initial_neg_log_likelihood)

    timer = time.time()
    print('Starting minimization...')
    result = minimize(compute_scr_neg_log_likelihood, x0=param_init, args=scr_args, method='Nelder-Mead',
                      options={'maxiter': 100, 'disp': True})
    print('Stopped minimizing after %0.4f seconds' % (time.time() - timer))

    alpha0_hat = result.x[0]
    alpha1_hat = np.exp(result.x[1])
    alpha2_hat = np.exp(result.x[2])
    N_hat = len(detections) + np.exp(result.x[3])
    neg_log_likelihood = result.fun
    num_iter = result.nit
    num_fev = result.nfev
    status = result.status
    print('MLE parameter values:\nalpha0_hat: %d\nalpha1_hat: %0.4f\nalpha2_hat: %0.2f\nN_hat: %d' % (
        alpha0_hat, alpha1_hat, alpha2_hat, N_hat))
    print('Minimization status code: %s' % status)
    print('Final negative log likelihood: %0.6f' % neg_log_likelihood)

if __name__ == '__main__':
    main()
