import csv
import numpy as np
import pickle
import rasterio
from scipy.optimize import minimize
import time


from utils import find_lcp_to_pts, compute_scr_neg_log_likelihood, callback
from visualize import visualize_trap_layout

def main():
    # print("Number of processors: ", mp.cpu_count())
    # print('------------------------------')
    
    landscape_raster_path = '../data/simulation/covariate_rasters/lowfrag_covariate.tif'
    landscape_raster_name = landscape_raster_path.split('/')[-1].split('.tif')[0]
    landscape_raster = rasterio.open(landscape_raster_path)
    landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    
    ## set ground truth landscape parameters
    ALPHA0 = 2
    ALPHA1 = 1/(2*0.3851879*0.3851879)
    ALPHA2 = 2.25
    N = 100
    print('Ground truth:\nalpha0: %d\nalpha1: %0.4f\nalpha2: %0.2f\nN: %d'%(ALPHA0, ALPHA1, ALPHA2, N))
    print('------------------------------')


    ## load trap locations
    # trap_loc_path = '../data/simulation/trap_locations/lowfrag_covariate_grid_14_14.csv'
    # trap_loc = [(eval(line[0]), eval(line[1])) for line in csv.reader(open(trap_loc_path, 'r'))]
    # --- CONS BIO DATA: ---#
    trap_loc_path = '../data/simulation/trap_locations/lowfrag_covariate_grid_14_14_consbio.csv'
    trap_loc = [(10*(eval(line[0])-0.5), 10*(eval(line[1])-0.5)) for line in csv.reader(open(trap_loc_path, 'r'))]
    # sort the order
    trap_loc = sorted(trap_loc, key=lambda element:(element[1], element[1]))
    trap_loc_orig = [(eval(line[0]), eval(line[1])) for line in csv.reader(open(trap_loc_path, 'r'))]
    

    ## compute least cost paths through this landscape
    lcp_distances = find_lcp_to_pts(landscape_ndarr, ALPHA2, trap_loc, raster_cell_size=0.1)
    
    ## load capture history
    # capture_history_path = '../data/simulation/capture_histories/lowfrag_N100_a2225.pkl'
    # capture_histories = pickle.load(open(capture_history_path, 'rb'))
    # detections = capture_histories[0]
    capture_history_path = '../data/simulation/capture_histories/lowfrag_N100_a2225_consbio.csv'
    detections = np.array([row for row in csv.reader(open(capture_history_path, 'r'))]).astype(np.float)
    
    ## minimize
    param_init = [ALPHA0, np.log(ALPHA1), np.log(ALPHA2), np.log(N-len(detections))]
    print('Initial parameters:\nalpha0: %d\nalpha1: %0.4f\nalpha2: %0.2f\nN: %d'%(ALPHA0, ALPHA1, ALPHA2, N))
    scr_args = (landscape_ndarr, 0.1, trap_loc, detections, 10)
    timer = time.time()
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

    print('MLE:\nalpha0: %d\nalpha1: %0.4f\nalpha2: %0.2f\nN: %d'%(alpha0_hat, alpha1_hat, alpha2_hat, N_hat))
    print('------------------------------')
    

if __name__ == '__main__':
    main()
    # cProfile.run('main()')