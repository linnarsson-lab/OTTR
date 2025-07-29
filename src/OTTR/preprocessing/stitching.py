import logging
import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from tifffile import imwrite, imread
from skimage.registration import phase_cross_correlation
from sklearn.neighbors import NearestNeighbors

from OTTR.preprocessing.utils import *

import skimage
from skimage.measure import regionprops

def find_adjacent_tiles(positions):
    nn = NearestNeighbors(n_neighbors=5,radius=1, metric='euclidean')
    nn.fit(positions)
    dists, indices = nn.kneighbors(positions, return_distance=True)
    
    adjacent_tiles = {}
    for i in range(len(indices)):
        dis = dists[i]
        idx = indices[i]
        in_range = idx[dis <= 1]
        
        adjacent_tiles[i] = [x for x in in_range if x != i]
    return adjacent_tiles

def determine_overlap(pair, positions, imgs, im_size, per_overlap=0.05):
    '''
    '''
    b0 = int(np.ceil(per_overlap * im_size))
    b1 = im_size - b0

    pos0 = positions[pair[0]]
    pos1 = positions[pair[1]]

    direction = ''

    if pos0[0] == pos1[0]:
        if pos0[1] > pos1[1]:
            im0 = imgs[pair[0]][:b0,:]
            im1 = imgs[pair[1]][b1:,:]
            direction = 'up'
        else:        
            im0 = imgs[pair[0]][b1:,:]
            im1 = imgs[pair[1]][:b0,:]
            direction = 'down'

    elif pos0[1] == pos1[1]:
        if pos0[0] > pos1[0]:
            im0 = imgs[pair[0]][:,:b0]
            im1 = imgs[pair[1]][:,b1:]
            direction = 'left'
        else:        
            im0 = imgs[pair[0]][:,b1:]
            im1 = imgs[pair[1]][:,:b0]
            direction = 'right'
            
    shift, _, _ = phase_cross_correlation(im0, im1)
        
    if direction == 'up':
        offset_coord = (int(shift[0]), int(-im_size + b0 + shift[1]))
    elif direction == 'down':
        offset_coord = (int(shift[0]), int(im_size - b0 + shift[1]))
    elif direction == 'left':
        offset_coord = (int(-im_size + b0 + shift[0]), int(shift[1]))
    else:
        offset_coord = (int(im_size - b0 + shift[0]), int(shift[1]))        
    return offset_coord

def stitch_images(fdir, outdir, FOVs_list, T, positions, overlapping_regions, rebin:int=None, overlap:float=0.05):
    '''
    '''
    imgs = [imread(os.path.join(fdir, pos_dir, T)) for pos_dir in FOVs_list]
    im_size = imgs[0].shape[0]
    edge = overlap * im_size

    N_y = np.max(positions[:,1])+1
    N_x = np.max(positions[:,0])+1
    
    ## Check how big the canvas should be
    if rebin == None:
        X = np.zeros((int(np.ceil((im_size * N_y*1.05)/2)*2), int(np.ceil((im_size * N_x)*1.05)/2)*2))
    else:
        X = np.zeros((int(np.ceil((im_size * N_y*1.05)/rebin)*rebin), int(np.ceil((im_size * N_x)*1.05)/rebin)*rebin))

    plotted_tiles = []
    plotted_coords = {}

    ## Pick image to start with
    pos_by_y = np.where(positions[:,0] == np.min(positions[:,0]))[0]
    pos_by_x = np.where(positions[:,1] == np.min(positions[pos_by_y,1]))[0]
    start_pos = list(set(pos_by_x) & set(pos_by_y))[0]
    y_0 = int((positions[start_pos][1] * im_size) - (positions[start_pos][1] * edge)+ 2*edge)
    x_0 = int((positions[start_pos][0] * im_size) - (positions[start_pos][0] * edge)+ 2*edge)

    ## Get endpoints
    y_1 = int(y_0+im_size)
    x_1 = int(x_0+im_size)

    X[y_0:y_1,x_0:x_1] = np.maximum(imgs[start_pos], X[y_0:y_1,x_0:x_1])
    plotted_tiles.append(start_pos)
    plotted_coords[start_pos] = [y_0, x_0]
    current_tiles = [start_pos]
    no_new_tiles = 0

    ## Plot the images on the canvas and resolve overlap
    while len(plotted_tiles) < len(positions):
        new_tiles = []
        for current_tile in current_tiles:
            next_tiles = overlapping_regions[current_tile]
            for tile in next_tiles:
                if not tile in plotted_tiles:
                    offset_coord = determine_overlap([current_tile,tile], positions, imgs, im_size, per_overlap=overlap)

                    y = plotted_coords[current_tile][0]
                    x = plotted_coords[current_tile][1]
                    y_off = y + offset_coord[1]
                    x_off = x + offset_coord[0]

                    X[y_off:y_off+im_size,x_off:x_off+im_size] = np.max(np.array([imgs[tile], 
                                                                                  X[y_off:y_off+im_size,x_off:x_off+im_size]]), 
                                                                  axis=0)
                    plotted_tiles.append(tile)
                    plotted_coords[tile] = [y_off, x_off]
                new_tiles.append(tile)
        current_tiles = np.unique(new_tiles)
    
    if rebin:
        h, w = X.shape
        X = rebin_im(X, [int(h/rebin),int(w/rebin)])

    imwrite(os.path.join(outdir, T), X.astype('uint16'))
    
    ## Check if it's the first image, in that case make a labelled image
    if T == 'Time00000.tif':
        fig, ax = plt.subplots(1,1)
        ax.imshow(X, 'gray')

        for pos, pos_dir in zip(positions, FOVs_list):
            if rebin:
                x_0 = (pos[0] * int(im_size/rebin)) + (0.5 * int(im_size/rebin))
                y_0 = (pos[1] * int(im_size/rebin)) + (0.5 * int(im_size/rebin))
            else:
                x_0 = (pos[0] * im_size) + (0.5 * im_size)
                y_0 = (pos[1] * im_size) + (0.5 * im_size)
            text = ax.text(x_0, y_0, pos_dir.split('_')[-1], color="w")
        plt.axis('off')
        plt.savefig(os.path.join('/', *fdir.split('/')) + '_FOVs.png')

    return 

def stitch_images_simple(fdir, outdir, FOVs_list, T, positions, rebin:int=None, overlap:float=0.1):
    '''
    '''
    imgs = [imread(os.path.join(fdir, pos_dir, T)) for pos_dir in FOVs_list]
    im_size = imgs[0].shape[0]
    edge = overlap * im_size
    im_min = np.array(imgs).min()

    N_y = np.max(positions[:,1])+1
    N_x = np.max(positions[:,0])+1

    ## Check how big the canvas should be
    if rebin == None:
        X = np.zeros((int(np.ceil((im_size * N_y*1.05)/16)*16), int(np.ceil((im_size * N_x)*1.05)/16)*16))
    else:
        X = np.zeros((int(np.ceil((im_size * N_y*1.05)/rebin)*rebin), int(np.ceil((im_size * N_x)*1.05)/rebin)*rebin))


    ## Plot the images on the canvas while taking into account the overlap
    for pos, im in zip(positions, imgs):
        if not pos[0] == 0:
            x_0 = int((pos[0] * im_size) - (pos[0] * edge))
        else:
            x_0 = pos[0]
        if not pos[1] == 0:
            y_0 = int((pos[1] * im_size) - (pos[1] * edge))
        else:    
            y_0 = pos[1]

        ## Get endpoints
        x_1 = int(x_0+im_size)
        y_1 = int(y_0+im_size)
        
        ## New tile
        cur_t = X[y_0:y_1,x_0:x_1]
        nonz = np.where(cur_t > 0)
        
        ## Add to existing image
        new_tile = im - im_min
        new_tile[nonz] = np.mean(np.array([cur_t[nonz].flatten(), new_tile[nonz].flatten()]), axis=0)
        X[y_0:y_1,x_0:x_1] = new_tile
                
    if rebin:
        h, w = X.shape
        X = rebin_im(X, [int(h/rebin),int(w/rebin)])

    imwrite(os.path.join(outdir, T), X.astype('uint16'))
    
    ## Check if it's the first image, in that case make a labelled image
    if T == 'Time00000.tif':
        fig, ax = plt.subplots(1,1)
        ax.imshow(X, 'gray')

        for pos, pos_dir in zip(positions, FOVs_list):
            if rebin:
                x_0 = (pos[0] * int(im_size/rebin)) + (0.5 * int(im_size/rebin))
                y_0 = (pos[1] * int(im_size/rebin)) + (0.5 * int(im_size/rebin))
            else:
                x_0 = (pos[0] * im_size) + (0.5 * im_size)
                y_0 = (pos[1] * im_size) + (0.5 * im_size)
            text = ax.text(x_0, y_0, pos_dir.split('_')[-1], color="w")
        plt.axis('off')
        plt.savefig(os.path.join('/', *fdir.split('/')) + '_FOVs.png')

    return 