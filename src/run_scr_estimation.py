import argparse
import csv
import numpy as np
import pickle
import rasterio
from scipy.optimize import minimize
import sys
import time


from utils import find_lcp_to_pts, compute_scr_neg_log_likelihood, callback, compute_expected_n, compute_expected_c
from visualize import visualize_trap_layout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Path to configuration file for SCR simulation settings.')
    parser.add_argument('--ac_realization_no', type=int, required=True, help='Integer index for activity center realization number.')
    parser.add_argument('--caphist_realization_no', type=int, required=True, help='Integer index for capture history realization number.')
    args = parser.parse_args()
    config_file_name = args.config_file.split('/')[-1].split('.csv')[0]
    log_file_name = '../logs/%s_%d_%d.log'%(config_file_name, args.ac_realization_no, args.caphist_realization_no)

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
    
    # Load into variables
    landscape_raster_file = config_params['landscape_raster_file']
    landscape_raster_name = landscape_raster_file.split('/')[-1].split('.tif')[0]
    raster_cell_size = config_params['raster_cell_size']
    trap_loc_path = '../data/simulation/trap_locations/%s_%s_%d_%d.csv'%(landscape_raster_name, config_params['trap_config'], config_params['n_traps_x'], config_params['n_traps_y'])
    ALPHA0 = config_params['ALPHA0']
    ALPHA1 = config_params['ALPHA1']
    ALPHA2 = config_params['ALPHA2']
    N = config_params['N']
    K = config_params['K']

    sys.stdout = open(log_file_name, 'w')
    print('Ground truth parameter values:\nALPHA0: %d\nALPHA1: %0.4f\nALPHA2: %0.2f\nN: %d'%(ALPHA0, ALPHA1, ALPHA2, N))

    landscape_raster = rasterio.open(landscape_raster_file)
    landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    trap_loc = [(eval(line[0]), eval(line[1])) for line in csv.reader(open(trap_loc_path, 'r'))]

    # # Compute least cost paths through this landscape
    # lcp_distances = find_lcp_to_pts(landscape_ndarr, ALPHA2, trap_loc, raster_cell_size=raster_cell_size)
    
    # Load capture history
    capture_history_path = '../data/simulation/capture_histories/%s_%d.pkl'%(config_file_name, args.ac_realization_no)
    capture_histories = pickle.load(open(capture_history_path, 'rb'))
    detections = capture_histories[args.caphist_realization_no]
    
    # Minimize
    param_init = [ALPHA0, np.log(ALPHA1), np.log(ALPHA2), np.log(N-len(detections))]
    print('Initial parameter values:\nalpha0: %d\nalpha1: %0.4f\nalpha2: %0.2f\nN: %d'%(ALPHA0, ALPHA1, ALPHA2, N))
    scr_args = (landscape_ndarr, raster_cell_size, trap_loc, detections, 10)

    initial_neg_log_likelihood = compute_scr_neg_log_likelihood(param_init, *scr_args)
    print('Initial negative log likelihood: %0.6f'%initial_neg_log_likelihood)
    
    timer = time.time()
    print('Starting minimization...')
    result = minimize(compute_scr_neg_log_likelihood, x0 = param_init, args = scr_args, method='Nelder-Mead', options={'maxiter': 100, 'disp': True})
    print('Stopped minimizing after %0.4f seconds'%(time.time() - timer))

    alpha0_hat = result.x[0]
    alpha1_hat = np.exp(result.x[1])
    alpha2_hat = np.exp(result.x[2])
    N_hat = len(detections) + np.exp(result.x[3])
    neg_log_likelihood = result.fun
    num_iter = result.nit
    num_fev = result.nfev
    status = result.status
    print('MLE parameter values:\nalpha0_hat: %d\nalpha1_hat: %0.4f\nalpha2_hat: %0.2f\nN_hat: %d'%(alpha0_hat, alpha1_hat, alpha2_hat, N_hat))
    print('Minimization status code: %s'%status)
    print('Final negative log likelihood: %0.6f'%neg_log_likelihood)

    # # < CONS BIO DATA > #
    # landscape_raster_file = '../data/simulation/covariate_rasters/lowfrag_covariate.tif'
    # landscape_raster_name = landscape_raster_file.split('/')[-1].split('.tif')[0]
    # landscape_raster = rasterio.open(landscape_raster_file)
    # landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    
    # ## set ground truth landscape parameters
    # ALPHA0 = 2
    # ALPHA1 = 1/(2*0.3851879*0.3851879)
    # ALPHA2 = 2.25
    # N = 100
    # print('Ground truth:\nalpha0: %d\nalpha1: %0.4f\nalpha2: %0.2f\nN: %d'%(ALPHA0, ALPHA1, ALPHA2, N))
    # print('------------------------------')

    # ## load trap locations
    # trap_loc_path = '../data/simulation/trap_locations/lowfrag_covariate_grid_14_14_consbio.csv'
    # trap_loc = [(10*(eval(line[0])-0.5), 10*(eval(line[1])-0.5)) for line in csv.reader(open(trap_loc_path, 'r'))]
    # # sort the order
    # trap_loc = sorted(trap_loc, key=lambda element:(element[1], element[1]))

    # ## compute least cost paths through this landscape
    # lcp_distances = find_lcp_to_pts(landscape_ndarr, ALPHA2, trap_loc, raster_cell_size=0.1)

    # ## load capture history
    # capture_history_path = '../data/simulation/capture_histories/lowfrag_N100_a2225_consbio.csv'
    # detections = np.array([row for row in csv.reader(open(capture_history_path, 'r'))]).astype(np.float)
    # # < \CONS BIO DATA > #

    # # ------------------------------ #
    # #  < TEST CODE AREA >
    # est_density_uniform = 100/float(landscape_ndarr.size)*np.ones((landscape_ndarr.shape[0], landscape_ndarr.shape[1]))
    # e_n = compute_expected_n(landscape_ndarr, 0.1, trap_loc, ALPHA0, ALPHA1, ALPHA2, 10, est_density_uniform)
    # e_c = compute_expected_c(landscape_ndarr, 0.1, trap_loc, ALPHA0, ALPHA1, ALPHA2, 10, est_density_uniform)
    # e_r = e_c - e_n
    # print(e_c, e_n, e_r)
    # print(sum(sum(detections)), len(detections), sum(sum(detections)) - len(detections))
    # assert 2==3, 'end test code'
    # #  < \TEST CODE AREA >
    # # ------------------------------ #

    

if __name__ == '__main__':
    main()
    # cProfile.run('main()')