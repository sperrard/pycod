################### Test cases for C.O.D. ###################
############### Complex Orthogonal Decomposition ############

import pycod as pcod

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scifft

#############################################################
# We study the case of C.O.D for a travelling wave with     #
# noise.                                                    #
#                                                           #
# This code aims at demonstrating how C.O.D. works and how  #
# it is done.                                               #
#############################################################

#### Spatio-temporal signal construction ####

#Info

L=400 #Domain size

lambda_1=200 #Wavelength

f_1=2 #Frequency

omega_1 = 2*np.pi*f_1

A_1=100
eps_1=0 #0 stat. #1 travel.
phi_1=0  #phase in rad

#Generation

nb_period=50
T_max=nb_period*2*np.pi/omega_1
raff_t=1000
delta_t=T_max/(raff_t-1)
list_t=np.linspace(0,T_max,raff_t)

raff_x=1200
delta_x=L/(raff_x-1)
list_x=np.linspace(0,L, raff_x)

(nt,nx) = (raff_t,raff_x)

sig=np.zeros((nt,nx))

for i in range(len(list_t)):
    for j in range(len(list_x)):
        sig[i,j]=A_1*(np.sin(omega_1*list_t[i]+phi_1)*np.sin(2*np.pi/lambda_1*list_x[j])+eps_1*np.cos(omega_1*list_t[i]+phi_1)*np.cos(2*np.pi/lambda_1*list_x[j]))
        
#Add noise ! 

sigma=0.5 #std deviation
Amp_noise=A_1*0.1 #amplitude of noise
        
sig_noisy = np.zeros((nt,nx))

for i in range(len(list_t)):
    for j in range(len(list_x)):
        sig_noisy[i,j]=sig[i,j]+Amp_noise*np.random.normal(loc=0.0, scale=sigma, size=1)[0]
        
# Verification plots
x_test=50
plt.figure(dpi=500)
plt.title("Time oscillations for x=x_test")
plt.plot(list_t,sig[:,x_test],label='clean')
plt.plot(list_t,sig_noisy[:,x_test],label='noisy')
plt.xlabel('t (s)')
plt.legend()
print('Amplitude of the oscillation at x=x_test :',max(sig[:,x_test]))

t_test=18
plt.figure(dpi=500)
plt.title("Surface form for t=t_test")
plt.plot(list_x,sig_noisy[t_test,:],label='clean')
plt.plot(list_x,sig_noisy[t_test,:],label='noisy')
plt.xlabel('x')
plt.legend()
print('Amplitude of the surface at t=t_test :',max(sig[t_test,:]))

#### C.O.D procedure ####

## First step : Hilbert transform

# Hilbert transform

H_transfo=pcod.H_transform(sig,delta_t)
H_transfo_noisy=pcod.H_transform(sig_noisy,delta_t)

# General complex fft to compare

fft_temp=scifft.fftshift(scifft.fft(sig, axis=0),axes=0)
fft_temp_noisy=scifft.fftshift(scifft.fft(sig_noisy, axis=0),axes=0)
f_axis=np.linspace(-1/(2*delta_t),1/(2*delta_t),raff_t)

plt.figure(dpi=500)
plt.title("FFT of the signal at x=x_test")
plt.plot(f_axis,2/len(list_t)*np.abs(fft_temp[:,x_test]),label='clean')
plt.plot(f_axis,2/len(list_t)*np.abs(fft_temp_noisy[:,x_test]),label='noisy')
plt.ylabel('A')
plt.xlabel(r'$f$ (Hz)')
plt.xlim([0,10])
plt.legend()

## Second step : Complex Orthogonal Dec. in itself

[lamb,v,Q_d,lamb_max]=pcod.comp_ortho_dec(H_transfo)
[lamb_noisy,v_noisy,Q_d_noisy,lamb_max_noisy]=pcod.comp_ortho_dec(H_transfo_noisy)

## Third step : Results

#Eigenvalue spectrum
plt.figure(dpi=500)
plt.plot(lamb,'*',label='clean') 
plt.plot(lamb_noisy,'*',label='noisy')
plt.xlabel('Component N°')
plt.yscale('log')
plt.ylabel('Energy of Components')
plt.title("Eigenvalues of C.O.D")
plt.legend()
plt.grid()

#Amplitudes
lamb_amp=pcod.amplitude(lamb, v) 
lamb_amp_noisy=pcod.amplitude(lamb_noisy, v_noisy) 
plt.figure(dpi=500)
plt.plot(lamb_amp,'*',label='clean') 
plt.plot(lamb_amp_noisy,'*',label='noisy')
plt.xlabel('N° du mode')
plt.ylabel(r'A')
plt.legend()
plt.title("Amplitude of Components")
plt.grid()

#FFT of each component (C.O.C : complex orthogonal coordinates)

r = pcod.fourier_COC(Q_d,v,lamb,delta_t,plot=1)
r_noisy = pcod.fourier_COC(Q_d_noisy,v_noisy,lamb_noisy,delta_t,plot=1)

#Which component of the C.O.D do you want ? (n+1)
n_mod=0
#Spatial form
[s_form_re,s_form_im] = pcod.spatial_form(n_mod, lamb, v)
[s_form_re_n,s_form_im_n] = pcod.spatial_form(n_mod, lamb_noisy, v_noisy)

plt.figure(dpi=500)
plt.plot(list_x, s_form_re,'b',label='real')
plt.plot(list_x, s_form_im,'r',label='imag')
plt.plot(list_x, s_form_re_n,'b--',label='real')
plt.plot(list_x, s_form_im_n,'r--',label='imag')
plt.legend()
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.title(f"Spatial form of C.O.D. Component n°{n_mod+1} (with dim.)")

print('\nTheoretical Amplitude :', A_1)
print('Spatial Amplitude from C.O.D. for component n°1:',lamb_amp[n_mod] )
print('Temporal amplitude from C.O.D. for component n°1:', np.sqrt(1/(raff_t)*np.sum(np.abs(Q_d)**2, axis=1)[n_mod])*max(v[:,n_mod]))  
print('Spatial Amplitude from C.O.D. for noisy:',lamb_amp_noisy[n_mod] )
print('Temporal amplitude from C.O.D. for noisy:', np.sqrt(1/(raff_t)*np.sum(np.abs(Q_d_noisy)**2, axis=1)[n_mod])*max(v_noisy[:,n_mod]))  

pcod.test_norm(v, n_mod) #Test normalization of the desired mode

pcod.test_cross_ortho(v) #Test the cross-orthogonality of the modes

pcod.test_norm(v_noisy, n_mod) #Test normalization of the desired mode

pcod.test_cross_ortho(v_noisy) #Test the cross-orthogonality of the modes

alpha=pcod.travelling_index(v[:,n_mod])
alpha_noisy=pcod.travelling_index(v_noisy[:,n_mod])
print('Traveling Index C.O.D, clean :',alpha)
print('Traveling Index C.O.D, noisy :',alpha_noisy)


####################### GIF generation ############################

pcod.make_wave_gif(sig_noisy, list_x, list_t, out_path="wave_5.gif", fps=10, frame_step=3, dpi=150, figsize=(9,3))
