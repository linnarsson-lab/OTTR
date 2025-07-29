import numpy as np

def fill_gaps(sequence):
    for i in range(sequence.shape[0]):
        if np.isnan(sequence[i]):
            sequence[i] = sequence[i-1]
    return sequence

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = np.array.shape[0]
    w = np.array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

def rebin_im(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1).astype(int)

def flat_field_BF(im, im_bg):
    '''
    Flat field correction for bright-field images
    '''
    im_cor = (im / im_bg) * np.max(im)
    return im_cor

def BG_correction(im, im_bg):
    '''
    Correct background fluorescence for fluorescent images
    '''
    im_cor = im / im_bg
    im_cor = im_cor * im_bg.mean()
    im_cor = im_cor
    return im_cor


def get_grid_from_coords(X_pos, Y_pos, step):
    '''
    Uses the X positions and Y positions to generate a grid for the images
    '''
    pos = []
    for dim in [X_pos, Y_pos]:
        if np.sum(dim < 0) > 0:
            dim = dim + abs(np.min(dim))
        rn = np.max(dim) - np.min(dim)
        coords = np.array([np.round(x/step, 0).astype(int) for x in dim])
        if np.min(coords) > 0:
            coords = coords - np.min(coords)
        elif np.min(coords) < 0:
            coords = coords + np.min(coords)
        pos.append(coords)
        
    coords = np.array([[x,y] for x,y in zip(pos[0], pos[1])])
    coords[:,0] = abs(coords[:,0] - coords.max(axis=0)[0])
    return coords.astype(int)