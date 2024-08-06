#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Contains functions for manipulating input fields and calculating the position of targets
'''

import numpy as np
from scipy.optimize import minimize

def ei(x):
    '''Complex exponential'''
    return np.exp(1j*x)


def calc_tilt(system, delay):
    '''Calculates the tilt for a given phase maximum phase delay'''
    return np.arcsin(delay / (system.k*system.A))        #delay must be in radians


def calc_correl(x1, x2):
    '''Correlation between two complex vectors x1 and x2'''
    return np.abs(np.conj(x1).T @ x2) / (np.linalg.norm(x1)*np.linalg.norm(x2))


def calc_tilt_phases(system, w, d=0):
    '''Calcultes the array of dipole phases to produce a focus tilt a distance w across the output plane.
    d corresponds to any extra shift from which may have already been applied, and so the angle is adjusted accordingly'''
    w_angle = np.arctan(w/(system.D + d))  #Shift twice memory effect distance
    max_phase = system.k*system.A*np.sin(w_angle)      #Convert to phase for dipole array

    phases = np.linspace(0,1,system.N_in)*max_phase      #Normalsied linear phase to apply to input array
    
    return phases


def calc_shift_phases(system,d,f):
    '''Calcultes the array of dipole phases to shift a focus, origianlly at distance f, a distance d behind the output plane.'''
    x = system.y_input - system.A/2
    phases = system.k*((x**2 + f**2)**.5 - (x**2 + (f+d)**2)**.5)
    
    return phases


def calc_d_extra(system,theta):
    '''Calculates the extra backward distance assuming the tilt occurs on a circle'''
    return system.D * (1 - np.cos(theta))


def calc_distances_1D(TM, system, max_scan=2, num_correls=100):
    '''Computes distance matrix between beads, along with corresponding correlations, using correlation in obtained TM rows.
    Max scan gives range of shifts in memory effects'''
    
    correls = np.zeros((len(TM),len(TM)-1,num_correls)) 

    max_shift = max_scan*system.L_me
    shifts = np.linspace(-max_shift, max_shift, num_correls)   #All shifts to apply

    for i,row in enumerate(TM):
        remainings = TM[np.arange(len(TM))!= i]

        for p,shift in enumerate(shifts):
            shifted_row = ei(calc_tilt_phases(system,shift)) * row

            for q,remaining in enumerate(remainings):
                correls[i,q,p] = calc_correl(shifted_row, remaining)
        
    #Find maximum corelations and convert tilt to displacement
    ds = -shifts[np.argmax(correls,axis=2)]
    
    #Take maximum correlation
    correls = np.max(correls, axis=2)
    
    return ds, correls


def calc_distances_2D(TM, system, max_tilt=2, max_shift=3, num_correls_tilt=100, num_correls_shift=100):
    '''Computes distance matrix between beads, along with corresponding correlations, using correlation in obtained TM rows.
    Max scan gives range of shifts in memory effects'''
    
    correls = np.zeros((len(TM),len(TM)-1, num_correls_tilt, num_correls_shift)) 

    #All tilts to apply
    max_tilt = max_tilt*system.L_me
    tilts = np.linspace(-max_tilt, max_tilt, num_correls_tilt) 
    
    #All shifts to apply
    max_shift = max_shift*system.L_me
    shifts = np.linspace(-max_shift, max_shift, num_correls_shift)

    for i,row in enumerate(TM):
        remainings = TM[np.arange(len(TM))!= i]
            
        for f, shift in enumerate(shifts):
            shifted_row = ei(calc_shift_phases(system,shift,system.D)) * row
            
            for p,tilt in enumerate(tilts):
                phases = calc_tilt_phases(system,tilt, d=shift)
                tilted_row = ei(phases) * shifted_row

                for q,remaining in enumerate(remainings):
                    correls[i,q,p,f] = calc_correl(tilted_row, remaining)
    
    #Reshape to find maximum
    correls = correls.reshape(len(TM),len(TM)-1, num_correls_tilt*num_correls_shift) 
    
    #Find maximum corelations and convert tilt to displacement for x and y direction
    max_inds = np.argmax(correls, axis=2)
    ds_x = -tilts[(max_inds/num_correls_shift).astype(int)]
    ds_y = -shifts[max_inds%num_correls_shift]
    
    #Take maximum correlation
    correls = np.max(correls,axis=2)
    
    return ds_x,ds_y, correls 


def sort(nums, correls, return_inds=0):
    '''Sorts numpy arrays of distances (nums) and correlations (correls) based on bead location'''
    #Sort inds
    sums = np.sum(nums>0,axis=-1)
    inds = np.flip(np.argsort(sums))

    #Sort collumns an rows
    nums = nums[inds]
    nums = nums[:,inds]
    correls = correls[inds]
    correls = correls[:,inds]

    if return_inds == 0:
        return nums, correls
    
    elif return_inds == 1:
        return nums, correls, inds

    
def produce_diag(matrix, diag_val=0):
    '''Introduce diagonal to rectangular matrix'''
    dim = np.max(matrix.shape)
    new_matrix = np.zeros((dim,dim))
    
    new_matrix[~np.eye(dim, dtype=bool)] = matrix.flatten()
    
    np.fill_diagonal(new_matrix, diag_val)
    
    return new_matrix


def stress_fn(x,D,C, alpha=6):
    ''' Calculate stess for use in mds
    
    x : vector of x coordinates
    D : Distance matrix
    C : Correlation matrix
    '''
    
    xj,xi = np.meshgrid(x,x)
    
    dissimilarity = C**alpha * (D - abs(xi - xj))**2
    
    np.fill_diagonal(dissimilarity, 0)     #Dont count diagonal in sum
    stress = np.sum(dissimilarity, axis=1) **.5
    stress = np.sum(stress)
    
    return stress


def mds(ds, correls, system, alpha=6, return_inds=0):
    '''Performs multidimentional scaling on inter-bead distances to produce distance map'''
    #Converting to square matricies
    distances = produce_diag(ds)
    correlations = produce_diag(correls, diag_val=1)

    #Sort by location
    if return_inds == 0:
        distances, correlations = sort(distances,correlations, return_inds=return_inds)
        distances = abs(distances)  #Algorithm uses magnitudes
    
    elif return_inds == 1:
        distances, correlations, inds = sort(distances,correlations, return_inds=return_inds)
        distances = abs(distances)  #Algorithm uses magnitudes

    initial_x = np.linspace(0,system.L_me, system.N_bead)

    #Perform minimisation
    result = minimize(stress_fn, initial_x, args=(distances, correlations,alpha), method='BFGS')
    
    if return_inds == 0:
        return result.x
    
    elif return_inds == 1:
        return result.x, inds

