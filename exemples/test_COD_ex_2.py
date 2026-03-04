################### Test cases for C.O.D. ###################
############### Complex Orthogonal Decomposition ############

import pycod as pcod

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scifft
from scipy.signal import hilbert

#############################################################
# We study the case of C.O.D for a temporally decreasing    #
# standing wave.                                            #
#                                                           #
# This code aims at demonstrating how C.O.D. works and how  #
# it is done.                                               #
#############################################################

#### Spatio-temporal signal construction ####

#Info

L=400 #Domain size

lambda_1=300 #Wavelength

f_1=5 #Frequency

omega_1 = 2*np.pi*f_1

A_1=16
eps_1=0 #0 stat. #1 travel.

phi_1=0  #phase in rad

a=1 #damping

#Generation

nb_period=20
T_max=nb_period*2*np.pi/omega_1
raff_t=500
delta_t=T_max/(raff_t-1)
list_t=np.linspace(0,T_max,raff_t)

raff_x=1200
delta_x=L/(raff_x-1)
list_x=np.linspace(-L/2, L/2, raff_x)

(nt,nx) = (raff_t,raff_x)

sig=np.zeros((nt,nx))

for i in range(len(list_t)):
    for j in range(len(list_x)):
        sig[i,j]=A_1*np.exp(-a*list_t[i])*(np.sin(omega_1*list_t[i]+phi_1)*np.sin(2*np.pi/lambda_1*list_x[j])+eps_1*np.cos(omega_1*list_t[i]+phi_1)*np.cos(2*np.pi/lambda_1*list_x[j]))
        
# Verification plots
x_test=30
plt.figure(dpi=500)
plt.title("Time oscillations for x=x_test")
plt.plot(list_t,sig[:,x_test])
plt.xlabel('t (s)')
plt.ylabel('Amplitude [-]')
##plt.save('time_series_3.pdf', transparent=True)
print('Amplitude of the oscillation at x=x_test :',max(sig[:,x_test]))

t_test=18
plt.figure(dpi=500)
plt.title("Surface form for t=t_test")
plt.plot(list_x,sig[t_test,:])
plt.xlabel('x')
plt.ylabel('Amplitude [-]')
print('Amplitude of the surface at t=t_test :',max(sig[t_test,:]))

#### C.O.D procedure ####

## First step : Hilbert transform

# Hilbert transform

H_transfo=pcod.H_transform(sig,delta_t)

# General complex fft to compare

fft_temp=scifft.fftshift(scifft.fft(sig, axis=0),axes=0)
f_axis=np.linspace(-1/(2*delta_t),1/(2*delta_t),raff_t)

plt.figure(dpi=500)
plt.title("FFT of the signal at x=x_test")
plt.plot(f_axis,2/len(list_t)*np.abs(fft_temp[:,x_test]))
plt.ylabel('Amplitude [-]')
plt.xlabel(r'$f$ (Hz)')
plt.xlim([0,25])
plt.ylim([-0.1,3])
##plt.save('fft_3.pdf', transparent=True)
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
#plt.save('energy_3.pdf', transparent=True)
plt.title("Eigenvalues of C.O.D")


#Amplitudes
lamb_amp=pcod.amplitude(lamb, v) 
plt.figure(dpi=500)
plt.plot(lamb_amp,'*') 
plt.xlabel('N° du mode')
plt.ylabel('Amplitude [-]')
##plt.save('amp_3.pdf', transparent=True)
plt.title("Amplitude of Components")


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
#plt.save('spa_form_1_3.pdf', transparent=True)
plt.title(f"Spatial form of C.O.D. Component n°{n_mod+1} (with dim.)")

pcod.test_norm(v, n_mod) #Test normalization of the desired mode

pcod.test_cross_ortho(v) #Test the cross-orthogonality of the modes


alpha=pcod.travelling_index(v[:,n_mod])
print('Theoretical Traveling Index for component n°1:',eps_1)
print('Traveling Index C.O.D for component n°1. :',alpha)

#################################################################
# Comparison Approximation Hilbert T. versus Exact Hilbert T. ###

# Parameters
omega = omega_1         
gamma = a   # Damping coefficient, small compared to omega if you want good approximation
Amp=A_1     
t = list_t

# Temporal part of the signal: damped sine
f = Amp * np.exp(-gamma * t) * np.sin(omega * t)

# Compute analytical approximation of Hilbert transform
hilbert_approx = - Amp * np.exp(-gamma * t) * np.cos(omega * t)

# Compute exact Hilbert transform using scipy's analytic signal
analytic_signal = hilbert(f)
hilbert_exact = np.imag(analytic_signal)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(t, hilbert_exact, label='Exact Hilbert Transform', linewidth=2)
plt.plot(t, hilbert_approx, '--', label='Approximate Hilbert Transform', linewidth=2)
plt.xlabel('t (s)', fontsize=13)
plt.ylabel(r'$\tilde{s}(t,x)$', fontsize=13)
plt.title(r'Hilbert Transform of $e^{{-\gamma t}} \sin(\omega t)$')
plt.legend()
#plt.save('Hilbert.pdf', transparent=True)
#plt.show()

###################################################################
####################### GIF generation ############################

#pcod.make_wave_gif(sig, list_x, list_t, out_path="wave_2.gif", fps=20, frame_step=3, dpi=150, figsize=(9,3))

#pcod.make_superposed_profiles(sig, list_x, list_t, times_to_plot=None, n_curves=15,out_path="superposed_profiles.pdf",dpi=150, figsize=(8,4))
