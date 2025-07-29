"""Pipeline for cell tracking using btrack and importing results and metrics into shoji"""

import logging
import sys, os
import glob
import warnings
import numpy as np
from tqdm import tqdm
from tifffile import imread
import shoji
import btrack
from btrack.constants import BayesianUpdates

from OTTR.pipeline.utils import *
from OTTR.pipeline import config
from OTTR.preprocessing.utils import *
from OTTR.segmentation.utils import *
from OTTR.tools.track_measures import *

warnings.simplefilter("ignore")
logging.info('Start processing')

FEATURES = [
    "area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "solidity",
    "eccentricity"
]

## Find all relevant directories

CONFIG_FILE = f"{OTTR.__file__}/.cell_config.json"

exp = sys.argv[1]
sd = sys.argv[2]

logging.info(sd)
d = sd.split('stitched')[0]

export_dir = '/' + '/'.join(sd.split('/')[:-1])

## Check if exp exists
db = shoji.connect()
if hasattr(db.OTTR.samples, exp):
    logging.info('Already exists in shoji')
else:
    logging.info('##############################')
    logging.info(f'## Starting with {exp}')
    logging.info(f'## Using subdir {sd}')
    logging.info('##############################')

    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)

    ## Load all sections
    mask_dir = f"{sd}/masks/"
    imdir = f"{d}/stitched/{sd.split('/')[-1].strip('_stitched')}"

    segmentation = [imread(f, dtype=np.int8)>0 for f in sorted(glob.glob(f"{mask_dir}*.tif"))]
    try:
        segmentation = np.array(segmentation).astype(np.int8)
    except Exception:
        logging.info('Not all arrays same size, adding padding')
        new_shape = np.max(np.array([x.shape for x in segmentation]),0)
        del segmentation

        segmentation = [padding(imread(f, dtype=np.int8)>0, new_shape[0], new_shape[1]) for f in sorted(glob.glob(f"{mask_dir}*.tif"))]
        segmentation = np.array(segmentation).astype(np.int8)
    dims = segmentation.shape

    logging.info(segmentation.shape)

    NCells = len(np.unique(imread(sorted(glob.glob(mask_dir + '*.tif'))[0])))
    NFrame = (dims[1]/1024) * (dims[2]/1024)

    objects = btrack.utils.segmentation_to_objects(
        segmentation,
        properties=tuple(FEATURES),
        num_workers=12,  # parallelise this
    )

    del segmentation

    ## initialise a tracker session using a context manager
    with btrack.BayesianTracker() as tracker:
        logging.info('Start tracking')

        # configure the tracker using a config file
        tracker.configure(CONFIG_FILE)
        tracker.tracking_updates = ["MOTION", "VISUAL"]
        tracker.features = FEATURES

        logging.info(f"NFrames: {NFrame}, NCells start: {NCells}")
        if NCells>1000:
            logging.info('Set tracking method to approximate')
            tracker.update_method = BayesianUpdates.APPROXIMATE
            tracker.max_search_radius = 100

        # append the objects to be tracked
        tracker.append(objects)

        # set the tracking volume
        # tracker.volume=((0, segmentation.shape[2]), (0, segmentation.shape[1]))
        tracker.volume=((0, dims[2]), (0, dims[1]))

        # track them (in interactive mode)
        tracker.track(step_size=100)

        # # generate hypotheses and run the global optimizer
        # tracker.optimize()
        tracker.optimize(options={"tm_lim": 60_000 * 100})

        # # get the tracks in a format for napari visualization
        # data, properties, graph = tracker.to_napari()

        # store the tracks
        tracks = tracker.tracks

        # store the configuration
        cfg = tracker.configuration

        # export the track data
        tracker.export(f"{export_dir}/{exp}_tracks.h5", obj_type="obj_type_1")

    lens = []

    for t in tracks:
        lens.append(len(t))
    divisions = 0
    for track in tracks:
        if track.ID != track.parent:
            divisions += 1

    logging.info(exp)
    logging.info(f'divisions; {divisions}')

    logging.info(f'Single frame: {np.sum(np.array(lens)==1)}')
    logging.info(f'>1 :{np.sum(np.array(lens)>1)}')
    logging.info(f'>10 :{np.sum(np.array(lens)>10)}')
    logging.info(f'>100 :{np.sum(np.array(lens)>100)}')
    logging.info(f'>200 :{np.sum(np.array(lens)>200)}')

    db.OTTR.samples[exp] = shoji.Workspace()
    ws = db.OTTR.samples[exp]

    d_measures = {'ID': [],
                'Label': [],
                'Start': [],
                'End': [],
                'T_start': [],
                'T_end': [],
                'D_max': [],
                'D_tot': [],
                'D_net': [],
                'Displacement': [],
                'Meandering': [],
                'Parent': [],
                'Root': []}

    t_measures = {'Coord': [],
                'Movement': [],
                'Dist': [],
                'Rel_angle': [],
                'Glob_angle': [],
                'Eccentricity': [],
                'Area': []
                }

    for track_df in tracks:
        if len(track_df) > 1:
            id = track_df.ID
            d_measures['ID'].append(id)
            d_measures['Label'].append(id)
            d_measures['Start'].append(np.array([track_df.y[0],track_df.x[0]]))
            d_measures['End'].append(np.array([track_df.y[-1],track_df.x[-1]]))
            d_measures['T_start'].append(np.min(track_df.t))
            d_measures['T_end'].append(np.max(track_df.t))
            d_measures['Parent'].append(track_df.parent)
            d_measures['Root'].append(track_df.root)

            track = np.array([track_df.y, track_df.x]).T
            d_measures['D_max'].append(d_max(track))
            d_measures['D_tot'].append(d_tot(track))
            d_measures['D_net'].append(d_net(track))
            d_measures['Displacement'].append(displacement(track))
            d_measures['Meandering'].append(meandering_index(track))

            t_measures['Coord'].append(track)
            t_measures['Movement'].append(track - track[0])
            t_measures['Dist'].append(dist(track))
            t_measures['Glob_angle'].append(global_turning_angle(track));
            t_measures['Eccentricity'].append(fill_gaps(np.array(track_df.properties['eccentricity'])))
            t_measures['Area'].append(fill_gaps(np.array(track_df.properties['area'])))

            rel = relative_turning_angle(speed(track))
            if len(rel) > 0:
                t_measures['Rel_angle'].append(rel);
            else:
                t_measures['Rel_angle'].append(np.array([0]))

    ws.cells = shoji.Dimension(shape=None)
    ws.yx = shoji.Dimension(shape=2)
    ws.Name = shoji.Tensor('string', (None,), inits = np.array([exp], dtype=object))
    ## Add metadata information
    str_attr = ['ID']
    NCells = len(d_measures['ID'])

    for i in str_attr:
        ws[i] = shoji.Tensor("string", ("cells",))

    for k in d_measures.keys():
        if k not in str_attr:
            v = np.array(d_measures[k])
            dt = str(v.dtype)
            if dt == 'int64':
                dt = 'uint32'
            if dt == 'float64':
                dt = 'float16'
            if len(v.shape) == 1:
                ws[k] = shoji.Tensor(dt, ("cells",))
            elif len(v.shape) == 2:
                ws[k] = shoji.Tensor(dt, ("cells", "yx",))

    ## Start adding cells
    logging.info('Saving cells to shoji')
    failed_cells = []
    for i in tqdm(range(NCells)):
        d = {}
        for k in d_measures:
            v = np.array(d_measures[k][i])
            dt = str(v.dtype)
            if dt == 'int64':
                dt = 'uint32'
            if dt == 'float64':
                dt = 'float16'

            if v.dtype.type == np.str_:
                d[k] = np.atleast_1d(v.astype('object'))
            elif len(v.shape) == 0:
                d[k] = np.atleast_1d(d_measures[k][i]).astype(dt)
            elif len(v.shape) >= 1:
                d[k] = np.atleast_2d(d_measures[k][i]).astype(dt)

        ws.cells.append(d)

    for attr in t_measures.keys():
        logging.info(f'Adding: {attr}')
        dt = str(t_measures[attr][0].dtype)
        if dt == 'int64':
            dt = 'uint16'
        if dt == 'float64':
            dt = 'float16'
        mxt = np.max(ws['T_end'][:]) + 1
        vals = np.zeros([len(t_measures[attr]),2,mxt], dtype=dt)

        for i in range(ws.cells.length):
            v = t_measures[attr][i]
            t = ws.T_start[i][0]
            vals[i,:,t:t+v.shape[0]] = v.T

        ws[attr] = shoji.Tensor(dt, ("cells", "yx", None,), inits = vals)

    ## Calculate movement inconsistency
    steps, residuals, inconsistency = Movement_inconsistency(ws, ws.T_start, ws.T_end)
    ws['Step_distance'] = shoji.Tensor('float16', ('cells', None,), inits = steps)
    ws['Step_residuals'] = shoji.Tensor('float16', ('cells', None,), inits = residuals)
    ws['Movement_inconsistency'] = shoji.Tensor('float16', ('cells',), inits = inconsistency)

    ws.cells = shoji.Dimension(shape=ws.cells.length)
    logging.info(failed_cells)

    time_tracked = np.array(ws.T_end[:]-ws.T_start[:]) / 4
    ws.Time_tracked = shoji.Tensor(str(time_tracked.dtype), ("cells",), inits=time_tracked)