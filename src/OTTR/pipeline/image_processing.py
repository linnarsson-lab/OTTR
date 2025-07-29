import logging
import pickle
import argparse
import yaml
import os
import sys
import time
import shutil
import numpy as np
from nd2reader import ND2Reader
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
from tqdm import tqdm
import multiprocessing as mp
import warnings
from tifffile import imwrite, imread

from OTTR.preprocessing.ND2_find_BG import ND2_find_BG
from OTTR.preprocessing.ND2_parser_FOVs import *
from OTTR.preprocessing.stitching import *
from OTTR.preprocessing.utils import *
from OTTR.pipeline import config

import skimage

def parse(config, fdir, file, outdir, rebin, max_p, verbose):
    try:
        parser = Nd2Parser_FOV(config, os.path.join(fdir, file), outdir, rebin=rebin, max_p=max_p, BG_correct=True, verbose=verbose)
        parser.Parse()
    except:
        return
    return

def update(q):
    # note: input comes from async `wrapMyFunc`
    pbar.update(1)

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

if __name__ == "__main__":
    fdir = sys.argv[1]
    config = config.load_config(fdir)
    outdir = os.path.join('/'.join(os.getcwd().split('/')[:-1]), 'processed', f'{fdir}_processed')
    files = os.listdir(fdir)
    logging.info(f"Parsing {fdir.split('/')[-1]}")

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    ## Parse the BG images
    logging.info(f'Exporting BG images')
    BG_parse = ND2_find_BG(file_dir=fdir, out_dir=outdir, rebin=config.params.binning, verbose=True)
    files = BG_parse.Parse()

    ## parse the remaining images
    logging.info(f'Exporting remaining images')
    pbar = tqdm(total=len(files))
    pbar.set_description(f'Processing {len(files)} files')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mp.get_context().Pool(mp.cpu_count(), maxtasksperchild=10) as pool:
            for file in files:
                pool.apply_async(parse, args=(config, os.getcwd(), file, outdir, config.params.binning, True, False,), callback=update)
            pool.close()
            pool.join()
            pbar.close()  

    ## Stitching and exporting videos
    vid_dir = os.path.join(outdir, 'vids')
    vid_dir_pos = os.path.join(vid_dir, 'positions')
    vid_dir_stitched = os.path.join(vid_dir, 'stitched')
    stitch_dir = os.path.join(outdir, 'stitched')
    folders = [f for f in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, f))]
    regions = np.unique([x.split('_')[0] for x in folders if len(x.split('_'))>1])

    if not os.path.isdir(vid_dir):
        os.mkdir(vid_dir)
    if not os.path.isdir(stitch_dir):
        os.mkdir(stitch_dir)
    if not os.path.isdir(vid_dir_stitched):
        os.mkdir(vid_dir_stitched)

    logging.info(folders)

    ## Stitching
    for Region in regions:
        logging.info(f'Stitching images for {Region}')
        channels = np.unique([x for x in folders if x.split('_')[0] == Region])  
        for ch in channels:
            logging.info(f'Stitching: {ch}')
            fdir = os.path.join(outdir, ch)
            positions = np.loadtxt(os.path.join(fdir, 'positions.txt'), delimiter=',').astype(int)
            FOVs_list = np.loadtxt(os.path.join(fdir, 'FOV_list.txt'), str)
            dir_stitched = os.path.join(stitch_dir, ch)

            ## Check how much to rebin the stitched image
            ranges = [[0,16], [17,64], [65, 248]]
            rb_val = [None, 2 , 4]
            for k, v in zip(ranges, rb_val):
                if len(positions) >= k[0] & len(positions) <= k[1]:
                    rebin = v
            
            logging.info(f'Rebinning videos: {rebin}')

            ## Made directory for stitched images
            if not os.path.isdir(dir_stitched):
                os.mkdir(dir_stitched)

            ## Get timepoints to stitch & overlapping tiles
            timepoints = sorted([T for T  in os.listdir(os.path.join(fdir, FOVs_list[0])) if T not in os.listdir(dir_stitched)])
            adjacent_tiles = find_adjacent_tiles(positions)

            pbar = tqdm(total=len(timepoints))
            pbar.set_description(f'Stitching {ch} with overlap: {config.params.overlap}')
            with mp.get_context().Pool(mp.cpu_count(), maxtasksperchild=10) as pool:
                for T in timepoints:
                    pool.apply_async(stitch_images_simple, args=(fdir, dir_stitched, FOVs_list, T, positions, None, config.params.overlap,), callback=update)
                pool.close()
                pool.join()
                pbar.close() 

            ## Export as .mov
            logging.info(f'Exporting stiched video')
            f_out = os.path.join(vid_dir_stitched, f'{ch}.mov')
            timepoints = sorted([T for T  in os.listdir(os.path.join(fdir, FOVs_list[0]))])
            SCALE = (1.0, 1.0, 0.09, 0.09)
            if rebin:
                h,w = io.imread(os.path.join(dir_stitched, timepoints[0])).astype('int16').shape
                new_dim = [int(h/rebin),int(w/rebin)]
                logging.info(new_dim)
                timelapse = np.asarray(
                    [rebin_im(io.imread(os.path.join(dir_stitched, T)).astype('int16'), new_dim) for T in timepoints]
                )
            else:
                timelapse = np.asarray(
                    [io.imread(os.path.join(dir_stitched, T)).astype('int16') for T in timepoints]
                )

            size = timelapse[0].shape
            out = cv2.VideoWriter(f_out, cv2.VideoWriter_fourcc(*'mp4v'), config.params.FPS, (size[1], size[0]), False)
            for data in timelapse:
                data = data - np.min(data)
                q = np.quantile(data.flatten(), .999)   ## Simple normalization
                data = (data / q) * 255
                data = np.clip(data, 0, 255).astype('uint8')
                out.write(data)
            out.release()
            

    if not os.path.isdir(vid_dir_pos):
        os.mkdir(vid_dir_pos)

    for Region in regions:
        logging.info(f'Exporting videos for {Region}')
        channels = np.unique([x for x in folders if x.split('_')[0] == Region])
        channels = [x for x in channels if x.split('_')[-1] != 'overview']
        pos = os.listdir(os.path.join(outdir, channels[0]))
        pos = [p for p in pos if p.split('_')[0] == 'pos'] ## Remove the position text files

        ## Stack images to tif files
        for ch in channels:
            logging.info(f'Exporting channel: {ch}')
            for p in pos:
                fn = os.path.join(vid_dir_pos, f'{ch}_{p}.mov')

                timepoints = sorted(os.listdir(os.path.join(outdir, ch, p)))
                SCALE = (1.0, 1.0, 0.09, 0.09)
                timelapse = np.asarray(
                    [io.imread(os.path.join(os.path.join(outdir, ch, p), T)).astype('int16') for T in timepoints]
                )

                size = timelapse[0].shape
                out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), config.params.FPS, (size[1], size[0]), False)
                for data in timelapse:
                    data = data - np.min(data)
                    q = np.quantile(data.flatten(), .999)   ## Simple normalization
                    data = (data / q) * 255
                    data = np.clip(data, 0, 255).astype('uint8')
                    out.write(data)
                out.release()

    logging.info('Finished exporting videos')
