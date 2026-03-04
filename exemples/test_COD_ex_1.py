################### Test cases for C.O.D. ###################
############### Complex Orthogonal Decomposition ############

import pycod as pcod

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scifft

##################### Test Case n°1 #########################

#############################################################
# This test case is based on the study of 2D fluid sloshing #
# into a tank. First the sloshing is forced so that         #
# oscillations are periodic with no damping. Amplitudes are #
# arbitrary. Frequencies and wavelengths follows the disp-  #
# ersion relation of waves in a finite size fluid height    #
# (Airy wave relation).                                     #
#                                                           #
# We therefore look at a sum of standing waves              #                                                 
#                                                           #
# This code aims at demonstrating how C.O.D. works and how  #
# it is done.                                               #
#############################################################

#### Functions for the sloshing case ####

def airy(h,L=1000,m=1,g=9.806):
    """Frequency calculus based on Airy relation, data has to be in mm"""
    return np.sqrt(m*g/(4*np.pi*L*0.001)*np.tanh(m*np.pi*(h/L)))

#### Spatio-temporal signal construction ####
#### Sum of two modes ####

#Info

L=400 #width of the tank
h=100 #height of water in tank

m_1=1 #selected mode n°1
m_2=3 #selected mode n°2

lambda_1=2*L/(m_1) #wavelenght n°1
lambda_2=2*L/(m_2) #wavelenght n°2

f_1=airy(h,L,m_1)
f_2=airy(h,L,m_2)

omega_1 = 2*np.pi*f_1 #pulsation n°1
omega_2 = 2*np.pi*f_2 #pulsation n°2

A_1=15 #Amplitude of component n°1
A_2=4 #Amplitude of component n°2
eps_1=0 #0 stat. #1 travel.
eps_2=0

phi_1=0   #phase in rad
phi_2=0

#Generation

nb_period=40
T_max=nb_period*2*np.pi/omega_1
raff_t=1000
delta_t=T_max/(raff_t-1)
list_t=np.linspace(0,T_max,raff_t)

raff_x=250
delta_x=L/(raff_x-1)
list_x=np.linspace(-L/2, L/2, raff_x)

(nt,nx) = (raff_t,raff_x)

sig=np.zeros((nt,nx))

for i in range(len(list_t)):
    for j in range(len(list_x)):
        sig[i,j]=A_1*(np.sin(omega_1*list_t[i]+phi_1)*np.sin(2*np.pi/lambda_1*list_x[j])+eps_1*np.cos(omega_1*list_t[i]+phi_1)*np.cos(2*np.pi/lambda_1*list_x[j]))+A_2*(np.sin(omega_2*list_t[i]+phi_2)*np.sin(2*np.pi/lambda_2*list_x[j])+eps_2*np.cos(omega_2*list_t[i]+phi_2)*np.cos(2*np.pi/lambda_2*list_x[j]))
        
# Verification plots
x_test=0
plt.figure(figsize=(6,4), dpi=500)
plt.title("Time oscillations for x=x_test")
plt.plot(list_t,sig[:,x_test])
plt.ylabel('Amplitude [-]')
plt.xlabel('t (s)')
#plt.savefig('time_series.pdf', dpi=500, transparent=True)
print('Amplitude of the oscillation at x=x_test :',max(sig[:,x_test]))

t_test=18
plt.figure(dpi=500)
plt.title("Surface form for t=t_test")
plt.plot(list_x,sig[t_test,:])
plt.xlabel('x')
print('Amplitude of the surface at t=t_test :',max(sig[t_test,:]))

#### C.O.D procedure ####

## First step : Hilbert transform

# Hilbert transform

H_transfo=pcod.H_transform(sig,delta_t)

# General complex fft to compare

fft_temp=scifft.fftshift(scifft.fft(sig, axis=0),axes=0)
f_axis=np.linspace(-1/(2*delta_t),1/(2*delta_t),raff_t)

plt.figure(figsize=(6,4), dpi=500)
plt.title("FFT of the signal at x=x_test")
plt.plot(f_axis,2/len(list_t)*np.abs(fft_temp[:,x_test]))
plt.ylabel('Amplitude [-]')
plt.xlabel(r'$f$ (Hz)')
plt.xlim([0,10])
plt.savefig('fft.pdf', dpi=500, transparent=True)
print('Amplitude of the time spectrum at x=x_test :',max(2/len(list_t)*np.abs(fft_temp[:,x_test])))

## Second step : Complex Orthogonal Dec. in itself

[lamb,v,Q_d,lamb_max]=pcod.comp_ortho_dec(H_transfo)

## Third step : Results

#Eigenvalue spectrum
plt.figure(dpi=500)
plt.plot(lamb,'*') 
plt.xlabel('Component N°')
plt.yscale('log')
plt.ylabel('Energy of Components')
plt.title("Eigenvalues of C.O.D")
plt.savefig('energy.pdf', dpi=500, transparent=True)
plt.grid()

#Amplitudes
lamb_amp=pcod.amplitude(lamb, v) 
plt.figure(dpi=500)
plt.plot(lamb_amp,'*') 
plt.xlabel('Component N°')
plt.ylabel(r'Amplitude [-]')
plt.title("Amplitude of Components")
#plt.savefig('amp.pdf', dpi=500, transparent=True)
plt.grid()

#FFT of each component (C.O.C : complex orthogonal coordinates)

r = pcod.fourier_COC(Q_d,v,lamb,delta_t,plot=1)

# Which component of the C.O.D do you want? (n+1)
n_mod = 0

# Plots
plt.figure(dpi=500)
plt.plot(list_x, np.real(v[:, n_mod]),'b',label='real')
plt.plot(list_x, np.imag(v[:, n_mod]),'r',label='imag')
plt.xlabel('x')
plt.title(f"Spatial form of the C.O.D. Component n°{n_mod+1} (normalized)")

[s_form_re,s_form_im] = pcod.spatial_form(n_mod, lamb, v)

plt.figure(dpi=500)
plt.plot(list_x, s_form_re,'b',label='real')
plt.plot(list_x, s_form_im,'r',label='imag')
plt.legend()
plt.xlabel('x')
plt.ylabel('Amplitude [-]')
#plt.savefig('spa_form_1.pdf', dpi=500, transparent=True)
plt.title(f"Spatial form of C.O.D. Component n°{n_mod+1} (with dim.)")


print('\nTheoretical Amplitude :', A_1)
print('Spatial Amplitude from C.O.D. for component n°1:',lamb_amp[n_mod] )
print('Temporal amplitude from C.O.D. for component n°1:', np.sqrt(1/(raff_t)*np.sum(np.abs(Q_d)**2, axis=1)[n_mod])*max(v[:,n_mod]))  

pcod.test_norm(v, n_mod) #Test normalization of the desired mode

pcod.test_cross_ortho(v) #Test the cross-orthogonality of the modes

alpha=pcod.travelling_index(v[:,n_mod])
print('Theoretical Traveling Index for component n°1:',eps_1)
print('Traveling Index C.O.D for component n°1. :',alpha)

#################################################################

# Which component of the C.O.D do you want? (n+1)
n_mod = 1

# Plots
plt.figure(dpi=500)
plt.plot(list_x, np.real(v[:, n_mod]),'b',label='real')
plt.plot(list_x, np.imag(v[:, n_mod]),'r',label='imag')
plt.xlabel('x')
plt.title(f"Spatial form of the C.O.D. Component n°{n_mod+1} (normalized)")

[s_form_re,s_form_im] = pcod.spatial_form(n_mod, lamb, v)

plt.figure(dpi=500)
plt.plot(list_x, s_form_re,'b',label='real')
plt.plot(list_x, s_form_im,'r',label='imag')
plt.legend()
plt.xlabel('x')
plt.ylabel('Amplitude [-]')
#plt.savefig('spa_form_2.pdf', dpi=500, transparent=True)
plt.title(f"Spatial form of C.O.D. Component n°{n_mod+1} (with dim.)")


print('\nTheoretical Amplitude :', A_2)
print('Spatial Amplitude from C.O.D. for component n°2:',lamb_amp[n_mod])
print('Temporal amplitude from C.O.D. for component n°2:', np.sqrt(1/(raff_t)*np.sum(np.abs(Q_d)**2, axis=1)[n_mod])*max(v[:,n_mod]))  

pcod.test_norm(v, n_mod) #Test normalization of the desired mode


alpha=pcod.travelling_index(v[:,n_mod])
print('Theoretical Traveling Index for component n°2:',eps_2)
print('Traveling Index C.O.D for component n°2. :',alpha)

###################################################################
####################### GIF generation ############################

#pcod.make_wave_gif(sig, list_x, list_t, out_path="wave_1.gif", fps=20, frame_step=3, dpi=150, figsize=(9,3))

#pcod.make_superposed_profiles(sig, list_x, list_t, times_to_plot=None, n_curves=10,out_path="superposed_profiles.pdf",dpi=150, figsize=(8,4))



















