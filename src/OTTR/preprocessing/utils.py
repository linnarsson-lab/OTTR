import numpy as np

def rebin_im(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1).astype(int)

def flat_field_BF(im, im_bg):
    '''
    Flat field correction for bright-field images
    '''
    return (im / im_bg) * np.max(im)

def BG_correction(im, im_bg):
    '''
    Correct background fluorescence for fluorescent images
    '''
    im_cor = im / im_bg
    im_cor = im_cor * im_bg.mean()
    return im_cor


def get_grid_from_coords(X_pos, Y_pos, step):
    '''
    Uses the X positions and Y positions to generate a grid for the images
    '''
    pos = []
    for dim in [X_pos, Y_pos]:
        if np.sum(dim < 0) > 0:
            dim = dim + abs(np.min(dim))
        coords = np.array([np.round(x/step, 0).astype(int) for x in dim])
        if np.min(coords) > 0:
            coords = coords - np.min(coords)
        elif np.min(coords) < 0:
            coords = coords + np.min(coords)
        pos.append(coords)
        
    coords = np.array([[x,y] for x,y in zip(pos[0], pos[1])])
    coords[:,0] = abs(coords[:,0] - coords.max(axis=0)[0])
    return coords.astype(int)