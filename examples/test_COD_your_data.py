################### Test cases for C.O.D. ###################
############### Complex Orthogonal Decomposition ############

import pycod as pcod

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scifft
import csv 

################# C.O.D on your data ########################
#############################################################

#Info on you data

nt=1500
nx=40
delta_t=1/50 #1/fps
list_t=np.linspace(0,30-30/1500,1500)
raff_t=nt
raff_x=nx

############### Calling the data ############################

sig=np.zeros((nt,nx))

with open('amp_wave.csv', encoding='utf8') as File:
    reader = csv.reader(File, delimiter=';')
    data_list = list(reader)
    i=0
    for line in data_list[0:]:
            sig[i][:]=line
            i=i+1
        
File.close()

list_x=[]

with open('abscissa_wave.csv', encoding='utf8') as File:
    reader = csv.reader(File, delimiter=';')
    data_list = list(reader)
    for line in data_list[0:]:
        list_x.append(float(line[0]))
        
File.close()




# Verification plots
x_test=30
plt.figure(dpi=500)
plt.title("Time oscillations for x=x_test")
plt.plot(list_t,sig[:,x_test])
plt.xlabel('t (s)')
print('Amplitude of the oscillation at x=x_test :',max(sig[:,x_test]))

t_test=18
plt.figure(dpi=500)
plt.title("Surface form for t=t_test")
plt.plot(list_x,sig[t_test,:])
plt.xlabel('x')
print('Amplitude of the surface at t=t_test :',max(sig[t_test,:]))

#### C.O.D procedure ####

## First step : Hilbert transform

H_transfo=pcod.H_transform(sig, delta_t)

# General complex fft to compare
fft_temp=scifft.fftshift(scifft.fft(sig, axis=0),axes=0)
f_axis=np.linspace(-1/(2*delta_t),1/(2*delta_t),raff_t)

plt.figure(dpi=500)
plt.title("FFT of the signal at x=x_test")
plt.plot(f_axis,2/len(list_t)*np.abs(fft_temp[:,x_test]))
plt.ylabel('A')
plt.xlabel(r'$f$ (Hz)')
plt.xlim([0,10])
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
plt.grid()


#Amplitudes
lamb_amp=[np.sqrt(lamb[i])*max(np.abs(v[:,i])) for i in range(len(lamb))] 
plt.figure(dpi=500)
plt.plot(lamb_amp,'*') 
plt.xlabel('N° du mode')
plt.ylabel(r'A')
plt.title("Amplitude of Components")
plt.grid()

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
plt.ylabel('Amplitude')
plt.title(f"Spatial form of C.O.D. Component n°{n_mod+1} (with dim.)")

print('Spatial Amplitude from C.O.D. for component n°1:',lamb_amp[n_mod] )
print('Temporal amplitude from C.O.D. for component n°1:', np.sqrt(1/(raff_t)*np.sum(np.abs(Q_d)**2, axis=1)[n_mod])*max(v[:,n_mod]))  


pcod.test_norm(v, n_mod) #Test normalization of the desired mode

pcod.test_cross_ortho(v) #Test the cross-orthogonality of the modes


alpha=pcod.travelling_index(v[:,n_mod])
print('Traveling Index C.O.D for component n°1. :',alpha)

#################################################################

#Which component of the C.O.D do you want ? (n+1)
n_mod=1
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
plt.ylabel('Amplitude')
plt.title(f"Spatial form of C.O.D. Component n°{n_mod+1} (with dim.)")

print('Spatial Amplitude from C.O.D. for component n°2:',lamb_amp[n_mod] )
print('Temporal amplitude from C.O.D. for component n°2:', np.sqrt(1/(raff_t)*np.sum(np.abs(Q_d)**2, axis=1)[n_mod])*max(v[:,n_mod]))  

pcod.test_norm(v, n_mod) #Test normalization of the desired mode


alpha=pcod.travelling_index(v[:,n_mod])
print('Traveling Index C.O.D for component n°2. :',alpha)


###################################################################
####################### GIF generation ############################
#pcod.make_wave_gif(sig, list_x, list_t, out_path="wave_test.gif", fps=20, frame_step=3, dpi=150, figsize=(9,3))





















