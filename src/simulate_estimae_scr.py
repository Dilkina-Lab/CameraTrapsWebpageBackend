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
from osgeo import gdal, osr


from visualize import *


def lambda_ipp_log_linear(x, beta0, beta1):
    return np.exp(beta0 + beta1*x)

def simulate_ipp(N, raster, n_real=1, seed=1111):

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

def create_file_layout(landscape_raster_file, trap_locations_file):
    ds = gdal.Open(landscape_raster_file)
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    wgs84_wkt = """
            GEOGCS["WGS 84",
                DATUM["WGS_1984",
                    SPHEROID["WGS 84",6378137,298.257223563,
                        AUTHORITY["EPSG","7030"]],
                    AUTHORITY["EPSG","6326"]],
                PRIMEM["Greenwich",0,
                    AUTHORITY["EPSG","8901"]],
                UNIT["degree",0.01745329251994328,
                    AUTHORITY["EPSG","9122"]],
                AUTHORITY["EPSG","4326"]] """

    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs, new_cs)

    # get the point to transform, pixel (0,0) in this case
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    # get the coordinates in lat long
    latlong = transform.TransformPoint(minx, miny)
    minlong = latlong[0]
    minlat = latlong[1]
    #print("long " + str(latlong[0]))
    #print("lat " + str(latlong[1]))
    latlong = transform.TransformPoint(maxx, maxy)
    #print("long " + str(latlong[0]))
    #print("lat " + str(latlong[1]))
    maxlong = latlong[0]
    maxlat = latlong[1]

    # get number of raster sells
    width = ds.RasterXSize
    height = ds.RasterYSize
    #print("raster sell size?? x" + str(width))
    #print("raster sell size?? y" + str(height))

    # divide to see size of raster cell
    longSize = maxlong - minlong
    latSize = maxlat - minlat
    longRosterSize = longSize / width
    latRosterSize = latSize / height
    #print("xSize " + str(longRosterSize) + " ySize " + str(latRosterSize))

    # open second file to get coordinates of points
    newChords = []
    i = 0
    f = open(trap_locations_file, "r")

    # cycle through and add to final list
    for x in f:
        if i == 0:
            i = 1
            l = x.split(",")
            number_of_traps = int(l[0])
            #print("config paramaters " + str(number_of_traps))
            continue
        i = 1
        l = x.split(",")
        latVar = l[1]
        longVar = l[2]
        latIndex = 0
        longIndex = 0
        currentLow = minlat
        currentHigh = minlat + latRosterSize
        # check latatude
        #print("\n" + latVar + " " + longVar)
        if float(latVar) < float(minlat) or float(latVar) > float(maxlat):
            print("does not work lat" + str(float(latVar)))
        else:
            while not (float(latVar) >= float(currentLow) and float(latVar) <= float(currentHigh)):
                currentLow += latRosterSize
                currentHigh += latRosterSize
                latIndex += 1

        # check longitude
        currentLow = minlong
        currentHigh = minlong + longRosterSize
        #print("long stuff " + str(longVar) + " " + str(currentLow) + " " + str(maxlong))
        if float(longVar) < float(minlong) or float(longVar) > float(maxlong):
            print("does not work long" + str(float(longVar)) + "min long " + str(float(minlong))  + "max long " + str(float(maxlong)))
        else:
            #print("current stuff " + str(longVar) + " " + str(currentLow) + " " + str(currentHigh))
            while not (float(longVar) >= float(currentLow) and float(longVar) <= float(currentHigh)):
                currentLow += longRosterSize
                currentHigh += longRosterSize
                longIndex += 1

        print(str(latIndex) + " " + str(longIndex))
        newChords.append([latIndex, longIndex])
    return newChords, latRosterSize

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
            if line[0] in ['landscape_raster_file', 'trap_config', 'trap_locations_file']:
                config_params[line[0]] = line[1]
            elif line[0] in ['ALPHA0', 'ALPHA1', 'ALPHA2', 'raster_cell_size']:
                config_params[line[0]] = float(line[1])
            else:
                config_params[line[0]] = int(line[1])

    # Load landscape
    landscape_raster = rasterio.open(config_params['landscape_raster_file'])
    landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    landscape_raster_name = config_params['landscape_raster_file'].split('/')[-1].split('.tif')[0]



    landscape_raster = rasterio.open(config_params['landscape_raster_file'])
    landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    print(landscape_ndarr.min(), landscape_ndarr.max())

    landscape_ndarr = (landscape_ndarr - landscape_ndarr.min()) / (landscape_ndarr.max() - landscape_ndarr.min())
    print(landscape_ndarr.min(), landscape_ndarr.max())

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
        config_params['log_file_name'] = '../data/simulation/trap_locations/'+landscape_raster_name+'_grid_'+str(config_params['n_traps_x'])+'_'+str(config_params['n_traps_y'])+'.csv'
        with open('../data/simulation/trap_locations/'+landscape_raster_name+'_grid_'+str(config_params['n_traps_x'])+'_'+str(config_params['n_traps_y'])+'.csv', 'w') as f:
            tlwriter = csv.writer(f)
            for p in trap_loc:
                tlwriter.writerow([p[0], p[1]])
        # visualize_trap_layout(landscape_ndarr, trap_loc)

    elif config_params['trap_config'] == 'file':
        trap_locations_name = config_params['trap_locations_file'].split('/')[-1].split('.csv')[0]
        timer = time.time()
        config_params['log_file_name'] = '../data/simulation/trap_locations/'+landscape_raster_name+'_file_'+trap_locations_name+'.csv'
        trap_loc = create_file_layout(config_params['landscape_raster_file'], config_params['trap_locations_file'])
        print('Simulated trap locations in %0.2f seconds'%(time.time() - timer))
        with open('../data/simulation/trap_locations/'+landscape_raster_name+'_file_'+trap_locations_name+'.csv', 'w') as f:
            tlwriter = csv.writer(f)
            for p in trap_loc[0]:
                tlwriter.writerow([p[0], p[1]])
        config_params['raster_cell_size'] = trap_loc[1]
        trap_loc = trap_loc[0]
        # visualize_trap_layout(landscape_ndarr, trap_loc)



    # Simulate capture histories
    timer = time.time()
    lcp_distances = find_lcp_to_pts(landscape_ndarr, config_params['ALPHA2'], trap_loc, raster_cell_size=config_params['raster_cell_size'])
    print('Computed ground truth least cost paths to each trap in %0.2f seconds'%(time.time() - timer))
    timer = time.time()
    for acridx in range(config_params['n_ac_realizations']):
        capture_histories = simulate_capture_histories(ac_realizations[acridx], trap_loc, 10, lcp_distances, config_params['ALPHA0'], config_params['ALPHA1'], seed=config_params['capture_histories_seed'])
        with open('../data/simulation/capture_histories/'+config_file_string+'_'+str(acridx)+'.pkl', 'wb') as f:
            pickle.dump(capture_histories, f)
    print('Simulated and saved spatial capture-recapture histories in %0.2f seconds'%(time.time() - timer))


    # Load into variables
    log_file_name = '../logs/%s_%d_%d.log' % (config_file_string, config_params['n_ac_realizations'], config_params['caphist_realization_no'])
    landscape_raster_file = config_params['landscape_raster_file']
    landscape_raster_name = landscape_raster_file.split('/')[-1].split('.tif')[0]
    raster_cell_size = config_params['raster_cell_size']
    trap_loc_path =  config_params['log_file_name']
    ALPHA0 = config_params['ALPHA0']
    ALPHA1 = config_params['ALPHA1']
    ALPHA2 = config_params['ALPHA2']
    N = config_params['N']
    K = config_params['caphist_realization_no']

    sys.stdout = open(log_file_name, 'w')
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
    capture_history_path = '../data/simulation/capture_histories/%s_%d.pkl' % (config_file_string, args.ac_realization_no)
    capture_histories = pickle.load(open(capture_history_path, 'rb'))
    detections = capture_histories[args.caphist_realization_no]

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