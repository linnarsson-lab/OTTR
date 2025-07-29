import logging
import sys, getopt, os
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
import multiprocessing as mp
from tifffile import imread
import matplotlib.pyplot as plt

from OTTR.preprocessing.utils import *
from OTTR.segmentation.utils import *
from OTTR.pipeline import config

config = config.load_config()

if __name__ == "__main__":
    max_t = None
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hp:m:",["help","max_t="])
        logging.info(f"extract_tracks arguments: {len(opts)}")
    except getopt.GetoptError as err:
        logging.info('extract_tracks -m <max_t>')
        logging.info(str(err))
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","help"):
            logging.info('extract_tracks -m <max_t>')
            sys.exit()
        elif opt in ("-m", "--max_t"):
            max_t = int(arg)

    ## Get directories
    fdir = os.getcwd()

    if not os.path.isdir('/' + os.path.join(*fdir.split("/")[:-1], 'segmented')):
        time.sleep(np.random.rand(1)[0])
        if not os.path.isdir('/' + os.path.join(*fdir.split("/")[:-1], 'segmented')):
            os.mkdir('/' + os.path.join(*fdir.split("/")[:-1], 'segmented'))

    segmentdir = os.path.join(*fdir.split("/")[:-1], 'segmented', f'{fdir.split("/")[-1]}_stitched')
    segmentdir = f'/{segmentdir}'

    ims = imread(f"{fdir}/Time00000.tif*")
    vals = ims.flatten()
    counts, bins, bars = plt.hist(vals, bins=200, range=(0,np.max(vals)))
    min_intensity = bins[np.argmax(counts)] * 1.05
    del ims, vals

    if not os.path.isdir(segmentdir):
        time.sleep(np.random.rand(1)[0])
        if not os.path.isdir(segmentdir):
            os.mkdir(segmentdir)
            logging.info(f'Creating dir for segmentation masks: {segmentdir}')

    if not os.path.isdir(os.path.join(segmentdir,'masks')):
        os.mkdir(os.path.join(segmentdir,'masks'))
        logging.info('Creating mask dir')

    ## Call Segmenter
    logging.info(f'Segmenting with diameter {config.params.cell_diameter}')
    call_segmenter(fdir, segmentdir, max_t=max_t, method=config.params.segmentation_method, diameter=config.params.cell_diameter, verbose=True)

    ## convert label images to pickles
    if not os.path.isdir(os.path.join(segmentdir, 'cells')):
        os.mkdir(os.path.join(segmentdir, 'cells'))
        
    logging.info('Convert masks to pickles')
    logging.info(f'Only maintaining objects in range({config.params.size_lim}) with minimum intensity {min_intensity}')
    time_frames = sorted([x.split('.')[0] for x in os.listdir(fdir)])[:max_t]
    time_frames = [x.split('.')[0] for x in os.listdir(os.path.join(segmentdir, 'masks'))]

    # with mp.get_context().Pool(min(10,len(time_frames)), maxtasksperchild=1) as pool:
    with mp.get_context().Pool(min(10,len(time_frames)), maxtasksperchild=1) as pool:
        for t in time_frames:
            pool.apply_async(mask_to_pickle, args=(fdir, segmentdir, t, config.params.size_lim, min_intensity,))
        pool.close()
        pool.join()
