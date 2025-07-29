## Import
import logging
import time, os, sys
import glob
import warnings
import multiprocessing as mp
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage import io
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_local, rank
from skimage.morphology import binary_closing, disk
from skimage import img_as_float
import cv2
from OTTR.pipeline import config

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

def segment(f, mask_dir, block_size, diameter, threshold_rel, config):
    f_mask = os.path.join(mask_dir, f.split('/')[-1])
    if not os.path.exists(f_mask):
        ## Load images
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = io.imread(f)
            bg_mask = image == 0

            ## Manage background from unimaged
            vals = image.flatten()
            vals = vals[vals>0]
            bg_val = np.quantile(vals, .05)
            image[np.where(image < bg_val)] = bg_val

            ## Segment image
            try:
                image[bg_mask] = rank.maximum(image, disk(int(block_size/2)))[bg_mask]
                image = image-threshold_local(image, block_size=block_size)  ## correct for local background
                image[bg_mask] = np.median(image[bg_mask])
                ## Scale Image
                image = np.clip(image, 0, np.quantile(image,.999))
                image = (image - image.min()) / (image.max() - image.min())
                im = rank.mean(image, disk(5))

                # Find the image maximum dilation and filter foreground/background
                otsu = threshold_otsu(im[~bg_mask])

                # Comparison between image_max and im to find the coordinates of local maxima
                coordinates = peak_local_max(im, footprint= np.ones([diameter,diameter], dtype=bool), min_distance=diameter, threshold_rel=threshold_rel,labels=im>otsu)    

                mask = np.zeros(im.shape, dtype=bool)
                mask[tuple(coordinates.T)] = True
                markers, _ = ndi.label(mask)
                distances = ndi.morphology.distance_transform_edt(markers)
                mask_t = binary_closing(im > otsu)
                labels = watershed(-distances, markers, mask=mask_t)
                sizes = np.array([[k, v] for k,v in Counter(labels.flatten()).items()])
                invalid = sizes[(sizes[:,1] < 20)|(sizes[:,1] > 2500),0]
                x = np.isin(labels, invalid)
                labels[x] = 0
                d = {k:v for v,k in enumerate(np.unique(labels))}
                labels = np.array([d[i] for i in labels.flatten()]).reshape(labels.shape)  
            except Exception as e:
                logging.info(e)
                labels = np.zeros(shape=im.shape)

            ## Save to file
            io.imsave(f_mask, labels)

            if f.split('/')[-1] == 'Time00000.tif*':
                image = io.imread(f)
                cnts = cv2.findContours(labels.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]

                ## Plot final image as example
                plt.figure(figsize=(24,24))
                plt.imshow(image, cmap='Greys_r')
                plt.axis('off')
                for c in cnts:
                    XY = c[:,0,:]
                    plt.plot(XY[:,0], XY[:,1], lw=.5, c='red')
                f_out = '_'.join([f.split('/')[-4], f.split('/')[-2]])
                plt.savefig(f'{config.paths.plot_dir}/segment/{f_out}.png', dpi=600, bbox_layout='tight')

class WatershedSegmenter:
    def __init__(self, 
    data_dir: str, 
    out_dir: str, 
    diameter:int=10, 
    # threshold_rel:float=.7, 
    threshold_rel:float=.4, 
    block_size:int=201, 
    verbose: bool = False,
    max_t: int = None,
    parallel: bool = False):
        '''
        '''
        self.config = config.load_config()
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.mask_dir = os.path.join(out_dir, 'masks')
        self.diameter = diameter
        self.threshold_rel = threshold_rel
        self.block_size = block_size
        self.max_t = max_t
        self.verbose = verbose
        self.parallel = parallel

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

        logging.info(f'Local maximum watershed segmentation')
        logging.info(f'Saving masks to {self.mask_dir}')

        if not self.max_t:
            images = sorted(glob.glob(os.path.join(self.data_dir, '*.tif*')))
        else:
            logging.info(f'Max timepoint: {self.max_t}')
            images = sorted(glob.glob(os.path.join(self.data_dir, '*.tif*')))[:self.max_t]

        if self.verbose:
            s = io.imread(images[0]).shape
            logging.info(f'Segmenting timepoints. Shape of images: {s}')


        if self.parallel:
            with mp.get_context().Pool(min(10,len(images)), maxtasksperchild=1) as pool:
                for f in images:
                    pool.apply_async(segment, args=(f, self.mask_dir, self.block_size, self.diameter, self.threshold_rel, self.config, ))
                pool.close()
                pool.join()
        else:
            First = True
            for f in tqdm(images):
                f_mask = os.path.join(self.mask_dir, f.split('/')[-1])
                if not os.path.exists(f_mask):
                    ## Load images
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        image = io.imread(f)
                        bg_mask = image == 0

                        ## Manage background from unimaged
                        vals = image.flatten()
                        vals = vals[vals>0]
                        bg_val = np.quantile(vals, .05)
                        image[np.where(image < bg_val)] = bg_val

                        ## Segment image
                        try:
                            image[bg_mask] = rank.maximum(image, disk(nt(block_size/2)))[bg_mask]
                            image = image-threshold_local(image, block_size=self.block_size)  ## correct for local background
                            image[bg_mask] = np.median(image[bg_mask])
                            ## Scale Image
                            mn = np.min(image)
                            image = np.clip(image, 0, np.quantile(image,.999))
                            image = (image - image.min()) / (image.max() - image.min())
                            im = rank.mean(image, disk(5))

                            # Find the image maximum dilation and filter foreground/background
                            otsu = threshold_otsu(im[~bg_mask])

                            # Comparison between image_max and im to find the coordinates of local maxima
                            coordinates = peak_local_max(im, footprint= np.ones([diameter,diameter], dtype=bool), min_distance=self.diameter, threshold_rel=self.threshold_rel,labels=im>otsu)

                            mask = np.zeros(im.shape, dtype=bool)
                            mask[tuple(coordinates.T)] = True
                            markers, _ = ndi.label(mask)
                            distances = ndi.morphology.distance_transform_edt(markers)
                            mask_t = binary_closing(im > otsu)
                            labels = watershed(-distances, markers, mask=mask_t)
                            sizes = np.array([[k, v] for k,v in Counter(labels.flatten()).items()])
                            invalid = sizes[(sizes[:,1] < 20)|(sizes[:,1] > 2500),0]
                            x = np.isin(labels, invalid)
                            labels[x] = 0
                            d = {k:v for v,k in enumerate(np.unique(labels))}
                            labels = np.array([d[i] for i in labels.flatten()]).reshape(labels.shape)

                        except Exception as e:
                            logging.info(e)
                            labels = np.zeros(shape=im.shape)

                        ## Save to file
                        io.imsave(f_mask, labels)

                        if First == True:
                            image = io.imread(f)
                            cnts = cv2.findContours(labels.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

                            ## Plot final image as example
                            plt.figure(figsize=(24,24))
                            plt.imshow(image, cmap='Greys_r')
                            plt.axis('off')
                            for c in cnts:
                                XY = c[:,0,:]
                                plt.plot(XY[:,0], XY[:,1], lw=.5, c='red')
                            f_out = '_'.join([f.split('/')[-4], f.split('/')[-2]])
                            plt.savefig(f'{self.config.paths.plot_dir}/segment/{f_out}.png', dpi=600, bbox_layout='tight')
                            First = False