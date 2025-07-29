import logging
import sys, os
import pickle
import numpy as np
from skimage import io
from skimage.measure import regionprops

def mask_to_pickle(indir: str, outdir: str, t: str, size_lim: list = [0,4000], min_intensity:int = 200, verbose=False):
    '''
    '''
    if verbose:
        logging.info(f'processing timepoint {t}')
    ## Generate paths
    intensity_path = os.path.join(indir, f'{t}.tif')
    mask_path = os.path.join(outdir, 'masks', f'{t}.tif')
    pickle_dir = os.path.join(outdir, 'cells')
    pickle_path = os.path.join(outdir, 'cells', f'{t}.pkl')  

    if not os.path.isdir(pickle_dir):
        os.mkdir(pickle_dir)

    if not os.path.isfile(pickle_path):
        ## Read images
        im = io.imread(mask_path)
        im_int = io.imread(intensity_path)
        
        ## Generate regionproperties using skimage
        regions = regionprops(im, intensity_image=im_int)
        
        ## Create dictionary to save cell measurements
        cells = {'id':[], 'area': [], 'centroid':[], 'eccentricity':[], 'image':[]}

        ## Loop over object and filter invalids
        for reg in regions:
            if (reg.convex_area>size_lim[0]) & (reg.convex_area<size_lim[1]) & (np.mean(reg.intensity_image) > min_intensity):
                cells['id'].append(reg.label)
                cells['centroid'].append(reg.centroid)
                cells['area'].append(reg.convex_area)
                cells['eccentricity'].append(reg.eccentricity)
                cells['image'].append(reg.intensity_image)
        
        ## Save to file
        pickle.dump(cells, open(pickle_path, 'wb'))
        return
    
def call_segmenter(fdir, outdir, max_t:int=None, method:str='cellpose', diameter=None, verbose=False):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if method == 'cellpose':
        from OTTR.segmentation.CellPoseSegmenter import CellPoseSegmenter
        segmenter = CellPoseSegmenter(data_dir=fdir, max_t=max_t, out_dir=outdir, diameter=diameter, model='cyto', verbose=verbose)
        segmenter.fit()
    elif method == 'watershed':
        from OTTR.segmentation.WatershedSegmenter import WatershedSegmenter
        segmenter = WatershedSegmenter(data_dir=fdir, max_t=max_t, out_dir=outdir, diameter=diameter, verbose=verbose, parallel=True) ## False
        segmenter.fit()