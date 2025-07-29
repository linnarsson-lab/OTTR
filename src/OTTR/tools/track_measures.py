from tqdm import trange

# -*- coding: utf-8 -*-
"""
Based on and adapted from the work of Carl-Magnus Svensson
Adapted by: Camiel Mannens

Created on Wed Apr  5 15:18:46 2017

@author: Carl-Magnus Svensson
@affiliation: Research Group Applied Systems Biology, Leibniz Institute for 
Natural Product Research and Infection Biology – Hans Knöll Institute (HKI),
Beutenbergstrasse 11a, 07745 Jena, Germany.
@email: carl-magnus.svensson@leibniz-hki.de or cmgsvensson@gmail.com

Functions that calculates a number of cell migration properties. Full details 
on the tracks and the analysis can be found in the review "Svensson et al.,
Untangling cell tracks: quantifying cell migration by time lapse image data 
analysis, Cytomerty Pt A, 2017"

Copyright by Dr. Carl-Magnus Svensson

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology -
Hans Knöll Insitute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

Licence: BSD-3-Clause, see ./LICENSE or 
https://opensource.org/licenses/BSD-3-Clause for full details

"""
import numpy as np

def dist(tracks):
    '''
        The distance from the center as a function of time.
        
        Parameters:
        -----------------------------------------------------------------------
        tracks: array_type
            The tracks to be analysed, in dimension N_time_steps x 2
            
        Returns:
        -----------------------------------------------------------------------
        d: array_type
            The distance from the center as a function of time for each track.          
    '''
    d = np.diff(tracks, axis = 0)
    d = np.sqrt(d[:,0]**2 + d[:,1]**2)
    
    return d
    
def MSD(tracks):
    '''
        The mean squared distance (MSD) travelled as a function of time.
        
        Parameters:
        -----------------------------------------------------------------------
        tracks: array_type
            The tracks to be analysed, in dimension N_tracks x N_time_steps x 2
            
        Returns:
        -----------------------------------------------------------------------
        MSD: array_type
            The MSD as a function of time for each track.          
    '''
    MSD = np.mean((tracks[0,0] - tracks[-1,0])**2 + (tracks[0,1] - tracks[-1,1])**2, axis = 0)
    
    return MSD
    
def d_tot(tracks):
    '''
        The total distance travelled as a function of time.
        
        Parameters:
        -----------------------------------------------------------------------
        tracks: array_type
            The tracks to be analysed, in dimension N_tracks x N_time_steps x 2
            
        Returns:
        -----------------------------------------------------------------------
        d_tot: array_type
            The total distance travelled as a function of time for each track.          
    '''
    d = dist(tracks)
    d_tot = np.sum(d, axis = 0)
    
    return d_tot
    
def d_net(tracks):
    '''
        The net distance travelled.
        
        Parameters:
        -----------------------------------------------------------------------
        tracks: array_type
            The tracks to be analysed, in dimension N_tracks x N_time_steps x 2
            
        Returns:
        -----------------------------------------------------------------------
        d_tot: array_type
            The net distance travelled for each track.          
    '''
    d_net = np.sqrt((tracks[0,0] - tracks[-1,0])**2 + (tracks[0,1] - tracks[-1,1])**2)
    
    return d_net
    
def d_max(tracks):
    '''
        The maximum distance travelled.
        
        Parameters:
        -----------------------------------------------------------------------
        tracks: array_type
            The tracks to be analysed, in dimension N_tracks x N_time_steps x 2
            
        Returns:
        -----------------------------------------------------------------------
        d_max: array_type
            The maximum distance travelled for each track.          
    '''
    x0 = tracks[0]
    d = np.zeros([tracks.shape[0]-1])
    ii = 0
    for x in tracks[1:]:
        d[ii] = np.sqrt((x0[0] - x[0])**2 + (x0[1] - x[1])**2)
        ii += 1
        
    return np.max(d, axis = 0)

def displacement(tracks):
    '''
        The total displacement (2D).
        
        Parameters:
        -----------------------------------------------------------------------
        tracks: array_type
            The tracks to be analysed, in dimension N_tracks x N_time_steps x 2
            
        Returns:
        -----------------------------------------------------------------------
        displacement: array_type
            The total displacement per track.          
    '''
    return np.array([(tracks[0,0] - tracks[-1,0]), (tracks[0,1] - tracks[-1,1])])
    
def meandering_index(tracks):
    '''
        The meandering index of a number of tracks.
        
        Parameters:
        -----------------------------------------------------------------------
        tracks: array_type
            The tracks to be analysed, in dimension N_tracks x N_time_steps x 2
            
        Returns:
        -----------------------------------------------------------------------
        MI: array_type
            The meandering index for each track.          
    '''
    d_n = d_net(tracks)
    d_t = d_tot(tracks)
    MI = np.nan_to_num(d_n/d_t)
    
    return MI

def speed(tracks):
    '''
    '''
    
    v = np.diff(tracks, axis = 0)
    return v

def global_turning_angle(tracks):
    '''
        Calculates all global turning angles for a collection of tracks.
        
        Parameters:
        -----------------------------------------------------------------------
        tracks: array_type
            The tracks to be analysed, in dimension N_tracks x N_time_steps x 2
            
        Returns:
        -----------------------------------------------------------------------
        alpha: array_type
            The global turning angles.          
    '''
    alpha = np.nan_to_num(np.arctan((np.diff(tracks[:,1])/np.diff(tracks[:,0]))))
    
    return alpha
    
def relative_turning_angle(vel):
    '''
        Calculates all relative turning angles for a collection of tracks.
        
        Parameters:
        -----------------------------------------------------------------------
        vel: array_type
            The instantaneous velocities of a collection of tracks, in 
            dimension N_tracks x N_time_steps x 2
            
        Returns:
        -----------------------------------------------------------------------
        phi: array_type
            The relative turning angles.          
    '''
    v = vel
    v_shift = np.roll(v,1,axis = 0)
    l = np.sqrt(v[:,0]**2 + v[:,1]**2)
    l_shift = np.roll(l,1)
    phi = np.nan_to_num([np.arccos(np.inner(v1,v2)/(l1*l2)) for v1,v2,l1,l2 in zip(v[1:],v_shift[1:],l[1:],l_shift[1:])])
    
    return phi

def Movement_inconsistency(ws, start, end):
    '''
        Calculates the movement inconsistency by comparing movement distance of every 
        step to the average movement of a cell.
        
        Parameters:
        -----------------------------------------------------------------------
        Coords: array_type
            The track coordinates, in 
            dimension N_tracks x 2 x time steps
        start: array_type
            start point for every track
        end: array_type
            end point for every track
            
        Returns:
        -----------------------------------------------------------------------
        phi: array_type
            The relative turning angles.    
    '''
    Coords = ws.Coord[:]
    t = Coords.shape[-1]

    steps = Coords[:,:,1:] - Coords[:,:,:-1]
    div = np.sqrt(steps[:,0,:]**2 + steps[:,1,:]**2)
    div[div == np.inf] = 0
    residuals = np.zeros([steps.shape[0], steps.shape[2]], dtype='float16')
    steps_out = np.zeros([steps.shape[0], steps.shape[2]], dtype='float16')
    inconsistency = np.zeros(Coords.shape[0], dtype='float16')

    for i in trange(Coords.shape[0]):
        step = div[i,ws.T_start[i][0]:ws.T_end[i][0]]
        avg = np.mean(step)
        step = step / avg ## Scale to average movement of 1, because we are not interested in how much they move
        avg = np.mean(step)
        vals = abs(step - avg)
        residuals[i,ws.T_start[i][0]:ws.T_end[i][0]] = vals
        inconsistency[i] = np.mean(vals)
        steps_out[i,ws.T_start[i][0]:ws.T_end[i][0]] = step
    return steps_out, residuals, inconsistency