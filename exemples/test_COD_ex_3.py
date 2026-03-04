################### Test cases for C.O.D. ###################
############### Complex Orthogonal Decomposition ############

import pycod as pcod

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as scifft

#############################################################
# We study the case of C.O.D applied to a standing wave     #
# with a modulation frequency behaviour                     #                                  #
#                                                           #
# This code aims at demonstrating how C.O.D. works and how  #
# it is done.                                               #
#############################################################

#### Spatio-temporal signal construction ####

# ---------- Parameters (feel free to change) ----------
L = 400              # spatial domain width (same scale as before)
raff_x = 250         # spatial samples
raff_t = 1000      # temporal samples (higher for good spectral resolution)
nt=raff_t
nx=raff_x

list_x = np.linspace(-L/2, L/2, raff_x)
T = 30.0             # total time (s)
list_t = np.linspace(0, T, raff_t)
delta_t = list_t[1] - list_t[0]

# Temporal modulation parameters
omega0 = 2.0 * np.pi * 1   # carrier rad/s (e.g. 1 Hz)
Omega = 2.0 * np.pi * 0.2    # modulation rad/s 
eps = 3                   # modulation index (epsilon)
A_1 = 2                      # amplitude

sig=np.zeros((nt,nx))

for i in range(len(list_t)):
    for j in range(len(list_x)):
        theta=omega0 * list_t[i] + eps * np.sin(Omega * list_t[i])
        sig[i,j]= A_1* (0.01*list_x[j])**3 * (np.sin(theta))

# Verification plots
x_test=0
plt.figure(dpi=500)
#plt.title("Time oscillations for x=x_test")
plt.plot(list_t,sig[:,x_test])
plt.xlabel('t (s)')
plt.ylabel('Amplitude [-]')
plt.savefig('time_signal_4.pdf', dpi=500, transparent=True)
print('Amplitude of the oscillation at x=x_test :',max(sig[:,x_test]))

t_test=18
plt.figure(dpi=500)
plt.title("Surface form for t=t_test")
plt.plot(list_x,sig[t_test,:])
plt.xlabel('x')
print('Amplitude of the surface at t=t_test :',max(sig[t_test,:]))

#### C.O.D procedure ####

## First step : Hilbert transform

# General complex fft
fft_temp=scifft.fftshift(scifft.fft(sig, axis=0),axes=0)
f_axis=np.linspace(-1/(2*delta_t),1/(2*delta_t),raff_t)

plt.figure(dpi=500)
plt.title("FFT of the signal at x=x_test")

# --- Overlay Jacobi–Anger theoretical peaks ---
from scipy.special import jv

# Define base and modulation frequencies
f0 = omega0 / (2 * np.pi)
Fm = Omega / (2 * np.pi)

# Choose which sidebands to show (n = -N ... N)
N = 15
n_vals = np.arange(-N, N+1)
f_sidebands = f0 + n_vals * Fm

# Theoretical amplitudes ∝ |J_n(eps)|
amp_spectrum = 2/len(list_t) * np.abs(fft_temp[:, x_test])
J = np.abs(jv(n_vals, eps))
J = J / np.max(J) * np.max(amp_spectrum)  # scale to match FFT height

plt.stem(f_sidebands, J, linefmt='C1-', markerfmt='C1o',
         basefmt=" ", label=r'Jacobi–Anger peaks ($|J_n(\epsilon)|$)')
plt.plot(f_axis,2/len(list_t)*np.abs(fft_temp[:,x_test]),'b',label='FFT')
plt.legend()
plt.ylabel('Amplitude [-]')
plt.xlabel(r'$f$ (Hz)')
plt.xlim([0,10])
#plt.savefig('fft_4.pdf', dpi=500, transparent=True)

# Hilbert transform

H_transform=pcod.H_transform(sig, delta_t)

## Second step : Complex Orthogonal Dec. in itself

[lamb,v,Q_d,lamb_max]=pcod.comp_ortho_dec(H_transform)

## Third step : Results

#Eigenvalue spectrum
plt.figure(dpi=500)
plt.plot(lamb,'*') 
plt.xlabel('Component N°')
plt.yscale('log')
plt.ylabel('Energy of Components')
#plt.title("Eigenvalues of C.O.D")
plt.savefig('lamb_4.pdf', dpi=500, transparent=True)



#Amplitudes
lamb_amp=[np.sqrt(lamb[i])*max(np.abs(v[:,i])) for i in range(len(lamb))] 
plt.figure(dpi=500)
plt.plot(lamb_amp,'*') 
plt.xlabel('N° du mode')
plt.ylabel(r'Amplitude [-]')
#plt.title("Amplitude of Components")
plt.savefig('amp_4.pdf', dpi=500, transparent=True)


#FFT of each component (C.O.C : complex orthogonal coordinates)

r = pcod.fourier_COC(Q_d,v,lamb,delta_t)

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
#plt.title(f"Spatial form of C.O.D. Component n°{n_mod+1} (with dim.)")
plt.savefig('spa_form_4.pdf', dpi=500, transparent=True)


print('Spatial Amplitude from C.O.D. for component n°1:',lamb_amp[n_mod] )
print('Temporal amplitude from C.O.D. for component n°1:', np.sqrt(1/(raff_t)*np.sum(np.abs(Q_d)**2, axis=1)[n_mod])*max(v[:,n_mod])) 

#Test norm of the spatial form
pcod.test_norm(v, n_mod) #Test normalization of the desired mode

pcod.test_cross_ortho(v) #Test the cross-orthogonality of the modes

alpha=pcod.travelling_index(v[:,n_mod])
print('Theoretical Traveling Index for component n°1:',0)
print('Traveling Index C.O.D for component n°1. :',alpha)

####################### GIF generation ############################

#pcod.make_wave_gif(sig, list_x, list_t, out_path="wave_3.gif", fps=25, frame_step=4, dpi=150, figsize=(9,3))

#pcod.make_superposed_profiles(sig, list_x, list_t, times_to_plot=None, n_curves=20,out_path="superposed_profiles.pdf",dpi=150, figsize=(8,4))

