## Import
import numpy as np
import logging
import time, os, sys
from urllib.parse import urlparse
import skimage.io as io
import matplotlib.pyplot as plt
from tifffile import imwrite, imread
import pickle as pkl
from tqdm import tqdm

from nd2reader import ND2Reader
from OTTR.preprocessing.utils import *

class Nd2Parser_FOV:
    def __init__(self, config, File: str, out_dir: str, BG_correct: bool = True, rebin: int = None, max_p: bool = False, verbose: bool = False):
        '''
        Parser for ND2-files. Takes 2D or 3D multichannel ND2 files (single scene) and exports them as 
        seperate tifs grouped first by fluorophore and then by timepoint

        Arg:
            File        path to ND2 file
            out_dir     path to export directory
        '''
        self.config = config
        self.File = File
        self.out_dir = out_dir
        self.EXP = None
        self.Time = None
        self.Region = None
        self.Channel = None
        self.BG_correct = BG_correct
        self.rebin = rebin
        self.max_p = max_p
        self.verbose = verbose

    def Parse(self):
        '''
        Calls the parser
        '''

        ## Extract identifiers from filename
        self.EXP, _, _, self.Time, self.Region = list(filter(None, self.File.split('/')[-1].split('_')))[:5]

        if self.Time[:4] != 'Time':
            return

        else:
            try:
                with ND2Reader(self.File) as nd2:
                    if self.verbose:
                        logging.info(f'Reading nd2 file with shape {nd2.sizes}')

                    # Collect metadata
                    all_metadata = nd2.parser._raw_metadata
                    parsed_metadata = nd2.parser._raw_metadata.get_parsed_metadata()
                    
                    # Collect FOV coords
                    x_data = np.array(all_metadata.x_data)
                    y_data = np.array(all_metadata.y_data)
                    z_data = np.array(all_metadata.z_data)
                    all_coords = np.hstack((z_data,x_data,y_data))

                    ## Extract import metadata
                    self.Channel = parsed_metadata['channels'][0]
                    h = parsed_metadata['height']
                    w = parsed_metadata['width']
                    pixel_microns = parsed_metadata['pixel_microns']
                    fields_of_view = parsed_metadata['fields_of_view']
                    tag_name = self.EXP + '_' + self.Region + '_' + self.Channel

                    if self.verbose:
                        logging.info(f'Processing Region {self.Region}, Time {self.Time} and channel {self.Channel}')

                    ## Check if Z-stack
                    if len(x_data)/len(fields_of_view) > 1:
                        nd2.bundle_axes = 'zyx'
                        nd2.iter_axes = 'v'
                        
                        ## Get stage locations
                        zstep = len(nd2.metadata['z_levels'])
                        X_pos = np.round(x_data[::zstep], 1)
                        Y_pos = np.round(y_data[::zstep], 1)
                        self.zstack = True
                        itr = fields_of_view
                    else:
                        nd2.bundle_axes = 'yx'
                        # nd2.iter_axes = 'v'

                        ## Get stage locations
                        X_pos = np.round(x_data, 1)
                        Y_pos = np.round(y_data, 1)
                        self.zstack = False
                        itr = fields_of_view

                    ## Generate file handles
                    channel_dir = os.path.join(self.out_dir, f'{self.Region}_{self.Channel}')
                    positions = [os.path.join(channel_dir, f'pos_{x}') for x in np.fromiter(range(len(nd2)), 'int').tolist()]

                    ## Check if out_dir exists
                    if not os.path.isdir(self.out_dir):
                        try:
                            os.mkdir(self.out_dir)
                        except:
                            pass
                    if not os.path.isdir(channel_dir):
                        try:
                            os.mkdir(channel_dir)
                        except:
                            pass
                    for i, f in enumerate(positions):
                        if not os.path.isdir(f):
                            try:
                                os.mkdir(f)
                            except:
                                pass

                    ## Check if position file exists
                    pos_file = os.path.join(channel_dir, 'positions.txt')
                    FOV_list = os.path.join(channel_dir, 'FOV_list.txt')
                    if not os.path.exists(pos_file):
                        if self.config.params.overlap == .1:
                            step = 1028
                        else:
                            step = 1076
                        cm = get_grid_from_coords(X_pos, Y_pos, step) ## Formerly 1076, 718
                        ## Check  grid
                        str_pos = ['_'.join([str(x[0]),str(x[1])]) for x in cm]
                        if len(str_pos) > len(np.unique(str_pos)):
                            logging.info(f'Spacing with distance: 718')
                            cm = get_grid_from_coords(X_pos, Y_pos, 718) ## Formerly 1076, 718
                        np.savetxt(pos_file, cm.astype(int), delimiter=",")
                    
                    if not os.path.exists(FOV_list):
                        with open(FOV_list, "w") as outfile:
                            FOVs = [f'pos_{x}' for x in np.fromiter(fields_of_view, 'int').tolist()]
                            outfile.write("\n".join(FOVs))

                    ## Return if already processed (check last position)
                    if os.path.exists(os.path.join(positions[-1], f'{self.Time}.tif')):
                        return

                    ## Export the images
                    if self.verbose:
                        tbar = tqdm(total=len(itr))
                    for fov in itr:
                        im = nd2[fov]
                        f_out = os.path.join(positions[fov], f'{self.Time}.tif')
                        if not os.path.exists(f_out):
                            if self.rebin:
                                if self.zstack:
                                    im = np.array([rebin_im(x, [int(h/self.rebin),int(w/self.rebin)]) for x in im])
                                else:
                                    im = rebin_im(im, [int(h/self.rebin),int(w/self.rebin)])
                                if self.max_p & self.zstack:
                                    if self.zstack == True:
                                        im = im.max(0)

                            ## Do Background correction
                            if self.BG_correct:
                                f_BG = os.path.join(self.out_dir, 'BG', f'{self.Channel}.tif')
                                if os.path.isfile(f_BG):
                                    im_bg = imread(f_BG)
                                    im = BG_correction(im, im_bg)
                                    if self.verbose:
                                        logging.info(f'Corrected background')
                                else:
                                    if self.verbose:
                                        logging.info('Background not found!')

                            ## Remove excessive values
                            q = np.quantile(im, 0.999)
                            imwrite(f_out, im.clip(0,q).astype('uint16'))
                        if self.verbose:
                            tbar.update(1)
                    if self.verbose:
                        tbar.close()

                    return
            
            except Exception as e:
                logging.info(f'{self.File}')
                logging.info(e)
                return
