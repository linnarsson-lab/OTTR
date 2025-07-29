## Import
import numpy as np
import logging
import time, os, sys
import matplotlib.pyplot as plt
from tifffile import imwrite
import glob

from nd2reader import ND2Reader
from OTTR.preprocessing.utils import *

class ND2_find_BG:
    def __init__(self, file_dir: str, out_dir: str, rebin: int = None, verbose: bool = False, to_zarr=False):
        '''
        Parser for ND2-files. Takes 2D or 3D multichannel ND2 files (single scene) and exports them as 
        seperate tifs grouped first by fluorophore and then by timepoint

        Arg:
            File        path to ND2 file
            out_dir     path to export directory
        '''
        self.file_dir = file_dir
        self.out_dir = out_dir
        self.rebin = rebin
        self.verbose = verbose
        self.to_zarr = to_zarr

    def Parse(self):
        '''
        Calls the parser
        '''
        # file_list = os.listdir(self.file_dir)
        file_list = glob.glob(os.path.join(self.file_dir, '*.nd2'))
        others = []     ## List of remaining files

        if self.verbose:
            logging.info(f'Checking {len(file_list)} files')

        self.out_dir = os.path.join(self.out_dir, 'BG')
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
               
        for file in file_list:
            _, _, _, Time, _ = list(filter(None, file.split('/')[-1].split('_')))[:5]
            if Time[:4] != 'Time':
                with ND2Reader(file) as nd2:

                    if len(nd2.metadata['fields_of_view']) > 1:
                        logging.info(f'Found multiple positions, saving seperately')
                        parsed_metadata = nd2.parser._raw_metadata.get_parsed_metadata()
                        channel = parsed_metadata['channels'][0]

                        for fov in nd2.metadata['fields_of_view']:
                            im = nd2[fov]
                            f_out = os.path.join(self.out_dir, f'{channel}_Region_0000{fov}.tif')
                            imwrite(f_out, im)

                    elif len(nd2.metadata['z_levels']) > 1:
                        parsed_metadata = nd2.parser._raw_metadata.get_parsed_metadata()
                        channel = parsed_metadata['channels'][0]
                        h = parsed_metadata['height']
                        w = parsed_metadata['width']
                        z = len(parsed_metadata['z_levels'])
                        f_out = os.path.join(self.out_dir, f'{channel}.tif')

                        if not os.path.isfile(f_out):
                            if self.verbose:
                                logging.info(f'Found BG image: {file}')
                            ## Check if Z-stack
                            if 'z' in nd2.axes:
                                if self.rebin:
                                    im = np.array([rebin_im(im, [int(h/self.rebin),int(w/self.rebin)]) for im in nd2]).reshape(z,int(h/self.rebin),int(w/self.rebin))
                                im = im.mean(0)
                            else:
                                im = nd2[0]
                                if self.rebin:
                                    im = rebin_im(im, [int(h/self.rebin),int(w/self.rebin)])
                            imwrite(f_out, im.astype('uint16'))
            else:
                others.append(file)

        return others