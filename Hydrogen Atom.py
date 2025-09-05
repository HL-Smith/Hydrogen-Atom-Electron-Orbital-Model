# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:30:18 2022

@author: harve
"""
#The Hydrogen Atom

import matplotlib as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import ipywidgets as widgets
import scipy as sp
from scipy import constants
from scipy import special 
from scipy import integrate

from IPython.display import display
from ipywidgets import interactive
import tkinter as tk



class HydrogenAtom:
    '''class defines the parameters (quantum numbers) needed to describe a hydrogenic (single-electron) atom:
    n = principle quantum  number
    l = angular (azimuthal) quantum number
    m = magnetic quantum number
    s = spin quantum number'''

#class attributes

   
    def __init__(self, n, r, l, m, s, mu, Z, theta, phi):
        self.n=n
        self.l=l
        self.m=m
        self.s=s
        self.theta=theta
        self.phi=phi
        self.r=r
        
        self.mu=mu #reduced mass
        self.Z=Z #atomic number
    
    def Observables(self, n, l, m, s):
        '''returns the energy and angular momentum of the electron in the hydrogen atom'''
        
        print('Energy = ', -13.6/n**2, 'eV')
        print('Orbital Angular Momentum = ', np.sqrt(l*(l+1))*constants.hbar, 'kgm^2s^-2')
        print('Spin Angular Momentum = ', np.sqrt(s*(s+1))*constants.hbar, 'kgm^2s^-2')
    
    def Radial(self,n,l,r):
        '''returns the radial function of the Hydrogen Atom'''
    
        C = np.sqrt((2.0/n)**3 * special.factorial(n-l-1) /(2.0*n*(special.factorial(n+l))**1))
        laguerre = special.assoc_laguerre(2.0*r/n,n-l-1,2*l+1)
        return r*C * np.exp(-r/n) * (2.0*r/n)**l * laguerre 
    
    
    
    def Sphericalharmonic(self, m, l, theta, phi):
        '''calculates the spherical harmonics'''
       
        return special.sph_harm(m, l, theta, phi).real


     
    def RadialGraph(self,n,l):
            '''returns a plot of the radial probability density of the Hydrogen Atom'''
            
            if n<2:
                rmax=10
            if 1<n<3:
                rmax=25
            if n>=3:
                rmax=60
    
            r=np.linspace(0,rmax,1000)
            
            C = np.sqrt((2/n)**3 * special.factorial(n-l-1) /(2.0*n*(special.factorial(n+l))**1))          
            laguerre = special.assoc_laguerre(2.0*r/n,n-l-1,2*l+1) 
            
            RadialProbDensity = (C * np.exp(-r/n) * (2.0*r/n)**l * laguerre)**2  
            RadialDistribution = (r**2)*(C * np.exp(-r/n) * (2.0*r/n)**l * laguerre)**2  
            
            fig, (ax1, ax2) = plt.subplots(2,1,constrained_layout=True)
            fig.suptitle("Electron radial probability density and distribution of a H atom: n="+str(n)+", l="+str(l)+", m="+str(m))
            
            ax1.plot(r, RadialProbDensity, color='orange', label = 'probability density')
            ax1.axvline(1, color='black', linestyle='--', label = 'r=$a_0$')
            ax1.set_xlabel('$r [a_0]$')
            ax1.set_ylabel('$(R_{nl}(r))^2$')
            ax1.legend(loc='upper right') #creates legend in upper left corner of plot
            
            
            ax2.plot(r, RadialDistribution, color='orange', label = 'probability distribution')
            ax2.axvline(1, color='black', linestyle='--', label = 'r=$a_0$')
            ax2.set_xlabel('$r [a_0]$')
            ax2.set_ylabel('$r^2 (R_{nl}(r))^2$')
            ax2.legend(loc='upper right') #creates legend in upper left corner of plot



    def SphericalharmonicGraph(self, m, l, theta, phi):
        '''returns a plot of the spherical harmonics'''
        
        phi, theta = np.meshgrid(phi, theta)
        Ylm = (special.sph_harm(m, l, theta, phi).real)
            
        x=np.sin(phi) * np.cos(theta) * abs(Ylm**2)
        y=np.sin(phi) * np.sin(theta) * abs(Ylm**2)
        z=np.cos(phi) * abs(Ylm**2)
        
        fig=plt.figure(figsize=(9,9))
        ax=fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='jet', alpha=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Angular probability density of a H atom: n="+str(n)+", l="+str(l)+", m="+str(m))
        plt.show()
        
        
    def Wavefunction(self,n,l,m):
        '''calculates the probability distribution of the wavefunction for the hydrogen atom, then uses a contour
           plot to visualize'''
            
        if n<2:
            rmax=10
        if 1<n<3:
            rmax=25
        if n>=3:
            rmax=40

        x=np.linspace(-rmax,rmax,1000)
        y=np.linspace(-rmax,rmax,1000)
        Z=0
        
        X,Y=np.meshgrid(x,y)
        
        r     = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arctan2(np.sqrt(X**2+Y**2), Z )
        phi   = np.arctan2(Y, X)
        
        
        W=(R.Radial(n, l, r)*R.Sphericalharmonic(m, l, theta, phi))**2
        
        print(np.shape(W))
        
        plt.figure(figsize=(9,9))
        plt.contourf(Y,X,W,200, cmap='Oranges')
        plt.title("Electron probability density of a H atom: n="+str(n)+", l="+str(l)+", m="+str(m))
        #plt.colorbar()
        plt.xlabel('$x [a_0]$')
        plt.ylabel('$y [a_0]$')
        
        
        
n=float(input('Enter an integer value: n = ')) #asks user to input values of n     
l=float(input('Enter an integer value: l = ')) #asks user to input values of m       
m=float(input('Enter an integer value: m = ')) #asks user to input values of Z       
s=float(input('Enter +0.5 for spin up electron: s = ')) #asks user to input values of s   
   

theta = np.linspace(0, np.pi*2,1000)
phi = np.linspace(0, np.pi,1000)        


R=HydrogenAtom(0,0,0,0,0,0,0,0,0)
R.SphericalharmonicGraph(m, l, theta, phi)
R.RadialGraph(n,l)
R.Wavefunction(n, l, m)
R.Observables(n, l, m, s)













