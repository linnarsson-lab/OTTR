## Import
import numpy as np
import logging
import time, os, sys
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage.io as io
import glob
import warnings
from tqdm import tqdm

from cellpose import utils, models, plot

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')


class CellPoseSegmenter:
    def __init__(self, 
    data_dir: str, 
    out_dir: str, 
    model: str = 'nuclei', 
    chunksize:int = 2048, 
    in_3D: bool = False, 
    gpu: bool = False, 
    diameter: int = None, 
    flow_thresh:int = 2.5, 
    verbose: bool = False,
    max_t: int = None):
        '''
        '''
        self.do_3D = in_3D
        self.gpu = gpu
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.mask_dir = os.path.join(out_dir, 'masks')
        self.model = model
        self.flow_thresh = flow_thresh
        self.diameter = diameter
        self.chunksize = chunksize
        self.max_t = max_t
        self.verbose = verbose

    def fit(self):
        '''
        '''
        ## Check if output directory exists
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        if not os.path.isdir(self.mask_dir):
            os.mkdir(self.mask_dir)

        if self.verbose:
            logging.info(f'Verbose is on')

        logging.info(f'Saving masks to {self.mask_dir}')

        model = models.Cellpose(gpu=True, model_type=self.model)
        channels = [0,0]

        if self.verbose:
            logging.info(f'GPU: {model.gpu}')

        if not self.max_t:
            images = sorted(glob.glob(os.path.join(self.data_dir, '*.tif')))
        else:
            images = sorted(glob.glob(os.path.join(self.data_dir, '*.tif')))[:self.max_t]

        ## Chunk if images are larger than chunksize
        s = io.imread(images[0]).shape
        if s[-1]  > self.chunksize or s[-2] > self.chunksize:
            logging.info(f'Large images, chunking before running cellpose, shape per timepoint: {s}')
            for f in tqdm(images):
                f_mask = os.path.join(self.mask_dir, f.split('/')[-1])
                if not os.path.exists(f_mask):
                    im = io.imread(f)

                    ## Check if data should be segmented in 3D
                    if self.do_3D:
                        im = im.reshape(im.shape[0], 1, im.shape[1], im.shape[2])

                        ##### --- FINISH 3D --- ####

                    else:
                        ## Setup grid and chunk data
                        im = im.reshape(1, im.shape[0], im.shape[1])
                        _, n_x, n_y = np.ceil(np.array(im.shape) / self.chunksize)
                        xy = np.mgrid[0:n_x*self.chunksize:self.chunksize, 0:n_y*self.chunksize:self.chunksize].reshape(2,-1).astype('int').T
                        FOVs = [im[:,pos[0]:min(pos[0]+chunksize, im.shape[1]), pos[1]:min(pos[1]+chunksize, im.shape[2])] for pos in xy]
                        
                        ## Run cellpose, save only masks
                        masks, _, _, _ = model.eval(FOVs, diameter=self.diameter, flow_threshold=0.4, channels=channels, do_3D=self.do_3D)

                        # Stitch back togeter
                        mask = np.zeros((im.shape[1], im.shape[2]))
                        mx = 0
                        for pos, m in zip(xy, masks):
                            m[m>0] = m[m>0] + mx
                            mx = np.max(m)
                            mask[pos[0]:min(pos[0]+chunksize, mask.shape[0]), pos[1]:min(pos[1]+chunksize, mask.shape[1])] = m
                        
                        ## Save to file
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            io.imsave(f_mask, mask.astype('int16'))
                        if self.verbose:
                            logging.info(f'Finished image {f}')

        else:
            if self.verbose:
                logging.info(f'small images, no chunking, shape per timepoint: {s}')
            ## 3D
            if self.do_3D:
                for f in tqdm(images):
                    f_mask = os.path.join(self.mask_dir, f.split('/')[-1])
                    if not os.path.exists(f_mask):

                        im = io.imread(f)
                        im = im.reshape(im.shape[0], 1, im.shape[1], im.shape[2])

                        ##### --- FINISH 3D --- ####

            ## 2D
            if not self.do_3D:
                if self.verbose:
                    logging.info(f'calling cellpose')
                # imgs = []
                for f in tqdm(images):
                    f_mask = os.path.join(self.mask_dir, f.split('/')[-1])
                    if not os.path.exists(f_mask):
                        ## Load images
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            im = io.imread(f)
                            im = im.reshape(1, im.shape[0], im.shape[1])
                        
                            ## Run cellpose, save only masks
                            try:
                                mask, _, _, _ = model.eval(im, diameter=self.diameter, flow_threshold=self.flow_thresh, channels=channels, do_3D=self.do_3D)
                            except Exception as e:
                                logging.info(e)
                                mask = np.zeros(shape=im.shape)

                            ## Save to file
                            io.imsave(f_mask, mask.astype('int16'))