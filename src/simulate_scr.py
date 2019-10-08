"""
Script for simulating a populations of individuals over a landscape, 
generating regular trap layouts, and simulating data from spatial 
capture-recapture studies.

The landscape raster, ground truth parameter values and other settings
used are read from a configuration file in /data/simulation/config.
"""
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

def simulate_capture_histories(ac, tl, K, lcp_dist, alpha0, alpha1, n_real=1, seed=1111):
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
            # dists = []
            for nt in range(len(tl)):
                # t = tl[nt]
                # tx = int(np.floor(t[0]))
                # ty = int(np.floor(t[1]))
                dist = lcp_dist[nt, sx, sy]
                # dists.append(dist)
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

def main():
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
    ac_realizations = simulate_ipp(config_params['N'], landscape_ndarr, n_real=config_params['n_ac_realizations'], seed=config_params['activity_centers_seed'])
    print('Simulated activity centers for %d realizations in %0.2f seconds'%(config_params['n_ac_realizations'], time.time() - timer))
    for acridx in range(config_params['n_ac_realizations']):
        # visualize_activity_centers(landscape_ndarr, ac_realizations[acridx])
        with open('../data/simulation/activity_centers/'+config_file_string+'_'+str(acridx)+'.csv', 'w') as f:
            acwriter = csv.writer(f)
            for p in ac_realizations[acridx]:
                acwriter.writerow([p[0], p[1]])
        
    # Simulate grid of trap locations
    if config_params['trap_config'] == 'grid':
        timer = time.time()
        trap_loc = create_grid_layout(config_params['n_traps_x'], config_params['n_traps_y'], landscape_ndarr)
        print('Simulated trap locations in %0.2f seconds'%(time.time() - timer))
        with open('../data/simulation/trap_locations/'+landscape_raster_name+'_grid_'+str(config_params['n_traps_x'])+'_'+str(config_params['n_traps_y'])+'.csv', 'w') as f:
            tlwriter = csv.writer(f)
            for p in trap_loc:
                tlwriter.writerow([p[0], p[1]])
        # visualize_trap_layout(landscape_ndarr, trap_loc)

    # Simulate capture histories
    timer = time.time()
    lcp_distances = find_lcp_to_pts(landscape_ndarr, config_params['ALPHA2'], trap_loc, raster_cell_size=config_params['raster_cell_size'])
    print('Computed ground truth least cost paths to each trap in %0.2f seconds'%(time.time() - timer))
    timer = time.time()
    capture_histories = simulate_capture_histories(ac_realizations[0], trap_loc, 10, lcp_distances, config_params['ALPHA0'], config_params['ALPHA1'], seed=config_params['capture_histories_seed'])
    print('Simulated spatial capture-recapture histories in %0.2f seconds'%(time.time() - timer))
    with open('../data/simulation/capture_histories/'+config_file_string+'.pkl', 'wb') as f:
        pickle.dump(capture_histories, f)


if __name__ == '__main__':
    main()