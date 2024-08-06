#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Contains discrete dipole model for modelling complex scattering systems
'''

import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt

class DiscreteDipole():
    '''The Discrete Dipole class provides functionality to emulate a complex scattering system consisting of N non-absorbing scattering particles.
    
    Initialisation:
        The user should specify the number of scattering particles N, the systems height L and height to width ratio beta.
        Initialisation will randomly produce scatter positions along with other properties such as the thickness.
    
    Simulation:
        Upon declaring input and output dimensions, the forward and backward propagating transmission matricies are calculated.
        Properties are used to update these matricies if the system's input and output planes are adjusted.
        One may simulate the system for a given input field using the propagate function.
        The show function similarly propagates a given field but produces an image of the whole system.
        '''

    def __init__(self, N, L, beta, k=1):
        super().__init__()
        
        #Coordinates
        self._x_scat = np.random.uniform(0,L*beta,N)
        self._y_scat = np.random.uniform(0,L,N)
        self._x_input = None
        self._y_input = None
        self._x_output = None
        self._y_output = None
        
        #Transmission Matricies
        self.TM = None
        self.I_TM_back = None
        self.TM_b = None  #_b denotes a bead TM
        self.I_TM_back_b = None
        
        #System Configuration
        self._bead_config = None
        self.L = L      #Height of system
        self.e = L*beta #Thickness of system
        self.k = k      #Wavevector
    
    #Coordinates
    @property
    def x_scat(self):
        '''x-coordinates of scatterers'''
        return self._x_scat

    @x_scat.setter
    def x_scat(self, value):
        self._x_scat = value
        try:
            self.gen_TMs()
            self.gen_bead_TMs()
        except:
            pass
        
    @property
    def y_scat(self):
        '''y-coordinates of scatterers'''
        return self._y_scat

    @y_scat.setter
    def y_scat(self, value):
        self._y_scat = value
        try:
            self.gen_TMs()
            self.gen_bead_TMs()
        except:
            pass
        
    @property
    def x_input(self):
        '''x-coordinates of input points'''
        return self._x_input

    @x_input.setter
    def x_input(self, value):
        self._x_input = value
        try:
            self.gen_TMs()
            self.gen_bead_TMs()
        except:
            pass
        
    @property
    def y_input(self):
        '''y-coordinates of input points'''
        return self._y_input

    @y_input.setter
    def y_input(self, value):
        self._y_input = value
        try:
            self.gen_TMs()
            self.gen_bead_TMs()
        except:
            pass
        
    @property
    def x_output(self):
        '''x-coordinates of output points'''
        return self._x_output

    @x_output.setter
    def x_output(self, value):
        self._x_output = value
        try:
            self.gen_TMs()
            self.gen_bead_TMs()
        except:
            pass
        
    @property
    def y_output(self):
        '''y-coordinates of output points'''
        return self._y_output

    @y_output.setter
    def y_output(self, value):
        self._y_output = value
        try:
            self.gen_TMs()
            self.gen_bead_TMs()
        except:
            pass
     
    #System Configuration
    @property
    def N_in(self):
        '''Number of input points'''
        return len(self.x_input)
    
    @property
    def N_out(self):
        '''Number of output points'''
        return len(self.x_output)
    
    @property
    def N_scat(self):
        '''Number of scattering particle'''
        return len(self.x_scat)
    
    @property
    def D(self):
        '''Input to output distance'''
        return np.mean(self.x_output) - self.x_input[0]
    
    @property
    def A(self):
        '''Width of input plance'''
        return np.max(self.y_input) - np.min(self.y_input)
    
    @property
    def W(self):
        '''Width of output plane'''
        return np.max(self.y_output) - np.min(self.y_output)
    
    @property
    def L_sp(self):
        '''Characteristic speckle length'''
        return 2*np.pi*self.D / (self.k*self.W)

    @property
    def L_me(self):
        '''Characteristic memory effect length'''
        return self.D / (self.k*self.e)
    
    #Bead Configuration
    @property
    def bead_config(self):
        '''Configuration  of output targets'''
        return self._bead_config

    @bead_config.setter
    def bead_config(self, value):
        # You can add your custom function here
        self._bead_config = value
        
        self.gen_bead_TMs()
        
    @property
    def N_bead(self): 
        '''Number of output targets'''
        return np.count_nonzero(self.bead_config == 1)
    
    @N_bead.setter
    def N_bead(self, value):
        bead_config = np.zeros(self.N_out,dtype=int)
        bead_config[:value] = 1
        np.random.shuffle(bead_config)
            
        self.bead_config = bead_config
        
    @property
    def x_bead(self):
        '''x-coordinates of targets'''
        return self.x_output[self.bead_config.astype(bool)]
    
    @property
    def y_bead(self): 
        '''y-coordinates of targets'''
        return self.y_output[self.bead_config.astype(bool)]
    
    
    def gen_E0_var(self, x, y, x_input,y_input, E_in):
        '''Computes the variable incident field (due to dipoles at x_input and y_input)
        at all given x and y coordinates'''
        E0 = np.zeros(np.shape(x), dtype=complex)

        for i, xi in enumerate(x_input):
            d = np.sqrt((x-xi)**2+(y-y_input[i])**2)
            E0 += -hankel1(0,d)*E_in[i]
                
        return E0
                
    def E_at_N_var(self, E0):
        '''Computes overall field at all N dipoles using variable E0'''
        x ,x_ = np.meshgrid(self.x_scat,self.x_scat)
        y ,y_ = np.meshgrid(self.y_scat,self.y_scat)

        d = np.sqrt((x-x_)**2+(y-y_)**2)

        matrix = hankel1(0,d)
        np.fill_diagonal(matrix, 1)

        return np.linalg.solve(matrix,E0)

    def E_at_r_var(self, x,y, EN, E0):
        '''Generates field at given grid of x and y coordinates using variable E0'''
        E = E0

        for i,xi in enumerate(self.x_scat):
            d = np.sqrt((x-xi)**2+(y-self.y_scat[i])**2)
            E += -hankel1(0,d)*EN[i]

        return E

    def E_at_out(self, EN, x_input,y_input, x_output, y_output, E_in):
        '''Measures the output field  at N_out positions at output plane'''

        E0 = self.gen_E0_var(x_output, y_output, x_input,y_input, E_in)
        E_out = self.E_at_r_var(x_output ,y_output ,EN,E0)        

        return E_out

    def gen_TMs(self):
        '''Returns the TM for a given configuration of scatters with given input and output dimensions
        Function triggered by changing output/input coordinates'''  
        
        TM = np.zeros((self.N_out, self.N_in), dtype=complex)

        for p in range(len(self.y_input)):
            E_in = np.zeros(self.N_in, dtype=complex)
            E_in[p] = 1 + 0*1j
            
            E0_scat = self.gen_E0_var(self.x_scat, self.y_scat, self.x_input, self.y_input, E_in) 
            EN = self.E_at_N_var(E0_scat)
            
            TM[:,p] = self.E_at_out(EN, self.x_input,self.y_input, self.x_output,self.y_output, E_in)
            
        self.TM = TM
            
        I_TM_back = np.zeros((self.N_in, self.N_out), dtype=complex)
            
        for p in range(len(self.y_output)):
            E_in = np.zeros(self.N_out, dtype=complex)
            E_in[p] = 1 + 0*1j
            
            E0_scat = self.gen_E0_var(self.x_scat, self.y_scat, self.x_output, self.y_output, E_in)
            EN = self.E_at_N_var(E0_scat)
            
            I_TM_back[:,p] = self.E_at_out(EN, self.x_output,self.y_output, self.x_input,self.y_input, E_in)

        self.I_TM_back = abs(I_TM_back)**2
    
    def gen_bead_TMs(self, random_config=0):
        '''Generate forward and backward TMs connecting to the flourescent beads. Forward TM is complex, backward
        is a real, intensity TM. Function triggered by changing bead configuration or N_bead (gives random config)'''  
        #If user wants new random bead configuration, produce it
        if random_config == 1:
            bead_config = np.zeros(self.N_out,dtype=int)
            bead_config[:self.N_bead] = 1
            np.random.shuffle(bead_config)
            
            self.bead_config = bead_config
            
        if len(self.bead_config) != (len(self.x_output) or len(self.y_output)):
            print('Output dimensions do not match bead configuration. Please update')
        
        else:
            #Reducing down to the correct dimensionality for out NN (later see if NN can predict bead location)
            self.TM_b = np.delete(self.TM,np.where(self.bead_config==0),axis=0)
            self.I_TM_back_b = np.delete(self.I_TM_back,np.where(self.bead_config==0),axis=1) 
    
    
    def pad_TM(self):
        '''Pads a bead TM with zeros anywhere there is no bead. Can either input 1 TM or both'''
        
        #Forward TM
        padded_TM = np.zeros(self.TM.shape, dtype=type(self.TM.flat[0]))
        padded_TM[np.where(self.bead_config == 1)] = self.TM

        #Backward TM
        padded_I_TM_back = np.zeros(self.I_TM_back.shape, dtype=type(self.I_TM_back.flat[0]))
        padded_I_TM_back += self.bead_config
        padded_I_TM_back[np.where(padded_TM == 1)] = self.I_TM_back.flat
                
        return padded_TM, padded_I_TM_back
    
    def propagate(self, E_in, direction='f', x_input='none',y_input='none',x_output='none',y_output='none'):
        '''Produces the field E_in between given coordinates (x_input,y_input to x_output,y_output)'''
        
        if direction == 'f':   #Forward Propagation
            x_input = self.x_input
            y_input = self.y_input
            x_output = self.x_output
            y_output = self.y_output
            
        elif direction == 'b':  #Backward propagation
            x_input = self.x_output
            y_input = self.y_output
            x_output = self.x_input
            y_output = self.y_input
            
        elif direction == 'c':  #Custom propagation
            if x_input=='none' and y_input=='none' and x_output=='none' and y_output=='none':
                print('Please provide custom coordinates')
            else:
                pass
            
        
        E0_scat = self.gen_E0_var(self.x_scat, self.y_scat, x_input, y_input, E_in) 
        EN = self.E_at_N_var(E0_scat)
        E_out= self.E_at_out(EN, x_input,y_input, x_output,y_output, E_in)
        
        return E_out
    
    def show(self, E_in, N_points, direction = 'f', plot_intensity=0, x_scale=1/4, y_scale=1/2, cap=0, figsize=(10,10), plot_phase=0, image_filename=None):
        '''Produces phase and amplitude plots of system for a given input. 'f' for forward propagation, 'b' for back'''
        
        if direction == 'f':   #Forward Propagation
            x_input = self.x_input
            y_input = self.y_input
            x_output = self.x_output
            y_output = self.y_output
            
        elif direction == 'b':  #Backward propagation
            x_input = self.x_output
            y_input = self.y_output
            x_output = self.x_input
            y_output = self.y_input
        
        #Mins and maxes
        xmin = np.min(x_input) - x_scale*self.e
        xmax = np.max(x_output) + x_scale*self.e
        ymin = np.min(y_input) - y_scale*self.L
        ymax = np.max(y_input) + y_scale*self.L
        
        #Plotting coords
        x = np.linspace(xmin,xmax,N_points)
        y = np.linspace(ymin,ymax,N_points)
        xx ,yy = np.meshgrid(x,y)

        #Calculate field
        E0_scat = self.gen_E0_var(self.x_scat, self.y_scat, x_input, y_input, E_in) 
        EN = self.E_at_N_var(E0_scat)
        E0_grid = self.gen_E0_var(xx, yy, x_input, y_input, E_in)
        E = self.E_at_r_var(xx,yy,EN,E0_grid)

        #Convert to intensity
        A = abs(E)
        phase = np.angle(E)/(2*np.pi) + .5

        #Label appropriately
        if plot_intensity == 1:
            A = A**2
            name = 'Intensity'
            
        else:
            name = 'Amplitude'
        
        #Choose to cap plotted intensity
        if cap != 0:
            A /= cap
            
        else:
            I_out = abs(self.propagate(E_in, direction=direction))**2
            A /= np.max(I_out)
            

        #Produce coordinates for plotted elements
        x_plot = (self.x_scat - x_input[0] + x_scale*self.e)*(N_points)/(xmax-xmin)
        y_plot = (self.y_scat + y_scale*self.L)*(N_points)/(ymax-ymin)

        x_source = (x_input - x_input[0] + x_scale*self.e)*(N_points)/(xmax-xmin)
        y_source = (y_input + y_scale*self.L)*(N_points)/(ymax-ymin)
        
        if self.N_bead != 0:
            x_bead = (self.x_bead - x_input[0] + x_scale*self.e)*(N_points)/(xmax-xmin)
            y_bead = (self.y_bead + y_scale*self.L)*(N_points)/(ymax-ymin)

        
        #Scale figure
        if plot_phase == 0:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            shrink = .765
            
        elif plot_phase == 1:
            w,h = figsize
            fig, (ax,ax2) = plt.subplots(1, 2, figsize=(2*w,h))
            
            shrink = .81
        
        #Plot points
        ax.scatter(x_source,y_source, color='red', label='Point Sources', s=30)
        ax.scatter(x_plot,y_plot, color='white', label='Scattering Dipole', s=15)
        
        
        if self.N_bead != 0:
            ax.scatter(x_bead,y_bead, color='orange', label='Flourescent Particle', s=30)
        
        #Plot and scale field
        im = ax.imshow(A)
        cbar = fig.colorbar(im, ax=ax, shrink=shrink)
        cbar.set_label('Normalised ' + name, fontsize=figsize[0]*2, labelpad=15)
        im.set_clim(vmax=1)  #Cap at peak output intensity
        
        ax.set_axis_off()
        
        #Appropriately label phase axis
        if plot_phase == 1:   
            ax2.scatter(x_source,y_source, color='red', s=30)
            ax2.scatter(x_plot,y_plot, color='white', s=15)
            
            if self.N_bead != 0:
                ax2.scatter(x_bead,y_bead, color='orange', s=30)
            
            im2 = ax2.imshow(phase)
            
            max_phase = np.max(phase)
            cbar = fig.colorbar(im2, ax=ax2, shrink=shrink, ticks=[max_phase/4, max_phase/2, max_phase*3/4, max_phase])
            cbar.ax.set_yticklabels(['π/2', 'π', '3π/2', '2π'])
            cbar.set_label('Phase', fontsize=figsize[0]*2, labelpad=15)
            ax2.set_axis_off()
        
        #Arange figure
        if self.N_bead == 0:
            fig.legend(loc='lower center', bbox_to_anchor=(.43,.01), facecolor='#666699', ncol=3, fontsize=figsize[0]*2)
         
        else:
            fig.legend(loc='lower center', facecolor='#666699', ncol=3, fontsize=figsize[0]*2)
    
        fig.tight_layout()
        
        #Choose to save
        if image_filename != None:
            plt.savefig(image_filename, dpi=360)

            print('Saved plotted validation to {} \n'.format(image_filename))

