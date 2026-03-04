############################################################### 

################ Packages #####################################

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as scifft
from scipy.signal import hilbert
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import pathlib
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize

###############################################################

############### Basis Functions ###############################

def H_transform(sig,delta_t,acc=0):
    """Generates the Hilbert transform of the matrix"""
    fft_temp=scifft.fftshift(scifft.fft(sig, axis=0),axes=0)
    nt, nx = sig.shape
    f_axis=np.linspace(-1/(2*delta_t),1/(2*delta_t),nt)

    # Hilbert transform

    fft_temp_prep=np.zeros((nt,nx),dtype=complex)

    fft_temp_prep[int(len(f_axis)/2):][:]=fft_temp[int(len(f_axis)/2):][:]
    fft_temp_prep[int(len(f_axis)/2)+1:][:]=2*fft_temp[int(len(f_axis)/2)+1:][:]

    H_transfo=scifft.ifft(fft_temp_prep, axis=0)
    
    if acc==1:
        # Compute analytic signal along axis=0 (time)
        H_transfo = hilbert(sig, axis=0)
        return H_transfo
    
    return H_transfo

def comp_ortho_dec(h_mat):
    """Performs the C.O.D. (solves the eigenvalue problem)"""
    nt, n_space = h_mat.shape
    Z = h_mat.T
    R = (1 / nt) * Z @ Z.conj().T  #Correlation matrix

    # R being hermitian, we can use "eigh" for more stability
    lamb, v = np.linalg.eigh(R)
    idx = lamb.argsort()[::-1]
    lamb = lamb[idx]
    v = v[:, idx]

    lamb = np.abs(np.real(lamb))  # to ensure that the result is real and positive
    Q_d = v.conj().T @ Z  #Projection

    return lamb, v, Q_d, max(lamb)


def travelling_index(v):
    """Calculus of the inverse of the condition number"""
    """Allows to know if waves are propagative or stationnary"""
    W=np.zeros((len(v),2))
    W[:,0]=np.real(v)
    W[:,1]=np.imag(v)
    
    alpha=1/np.linalg.cond(W)
    return alpha

def amplitude(lamb,v):
    """Returns the amplitude of all the modes"""
    return [np.sqrt(lamb[i])*max(np.abs(v[:,i])) for i in range(len(lamb))]

def spatial_form(n_mod,lamb,v):
    """Returns the spatial form of the desired component"""
    return [np.sqrt(lamb[n_mod]) * np.real(v[:,n_mod]),np.sqrt(lamb[n_mod]) * np.imag(v[:,n_mod])]

def test_norm(v,n_mod):
    """Test normalization of the desired mode"""
    integrand = np.matmul(np.conj(v[:,n_mod]),v[:,n_mod]) #has to be 1+0j in theory
    return print('Test Spatial Norm :', integrand)

def test_cross_ortho(v):
    """Test the cross-orthogonality of the modes"""
    #Full cross-orthogonality check
    orthogonality_matrix = np.conj(v.T) @ v

    # Identity matrix of the same size
    num_modes = v.shape[1]
    identity_matrix = np.eye(num_modes)

    # Tolerance for numerical errors
    tol = 1e-12

    # Check if the matrix is close to identity
    if np.allclose(orthogonality_matrix, identity_matrix, atol=tol):
        print("All spatial modes are properly orthonormal (cross-orthogonality OK).")
    else:
        print("Warning: Spatial modes are NOT perfectly orthonormal!")
        # Optionally, print the difference matrix
        print("Deviation from identity:\n", orthogonality_matrix - identity_matrix)
    return None

###################################################################

########## Functions for non-uniform (nu) spatial grid ############

def spatial_weights(list_x):
    """Computes the weights for the non-uniform case"""
    nx=len(list_x)
    w = np.zeros(nx)
    w[0] = (list_x[1] - list_x[0]) / 2
    w[-1] = (list_x[-1] - list_x[-2]) / 2
    w[1:-1] = (list_x[2:] - list_x[:-2]) / 2
    return w


def comp_ortho_dec_nu(h_mat, w):
    """
    Performs the weighted C.O.D. (solves the eigenvalue problem)
    accounting for non-uniform spatial grid.
    
    h_mat: complex analytic signal (nt x nx)
    w: spatial weights (nx,)
    """
    h_mat = np.asarray(h_mat)
    nt, nx = h_mat.shape
    if len(w) != nx:
        raise ValueError("w length must equal number of spatial points (nx)")

    Z = h_mat.T  # shape (nx, nt)

    sqrtw = np.sqrt(np.asarray(w))
    Z_w = sqrtw[:, None] * Z   # W^{1/2} * Z  -> shape (nx, nt)

    R = (1.0 / nt) * (Z_w @ Z_w.conj().T)  # nx x nx Hermitian

    # Eigendecomposition (sorted descending)
    lamb, u = np.linalg.eigh(R)
    idx = np.argsort(lamb)[::-1]
    lamb = np.abs(np.real(lamb[idx]))
    u = u[:, idx]

    # Convert back to spatial modes orthonormal under weighted inner product
    phi = (1.0 / sqrtw[:, None]) * u   # shape nx x n_modes

    # Re-normalize phi under weighted inner product
    norms = np.sqrt(np.real(np.sum((phi.conj() * w[:, None]) * phi, axis=0)))
    phi = phi / norms[None, :]

    # Temporal coefficients: a_j(t) = phi_j^H W Z  -> shape (n_modes, nt)
    Q_d = phi.conj().T @ (w[:, None] * Z)

    lamb_max = float(lamb[0]) if len(lamb) > 0 else 0.0
    return lamb, phi, Q_d, lamb_max

def travelling_index_nu(v, w):
    """
    Traveling index for a non-uniform spatial grid using condition number.
    v : complex spatial mode, shape (nx,)
    w : weights for each spatial point, shape (nx,)
    """
    # Apply weights
    v_re = np.sqrt(w) * np.real(v)
    v_im = np.sqrt(w) * np.imag(v)
    
    # Form 2-column matrix
    W = np.vstack([v_re, v_im]).T  # nx x 2
    
    # Compute the condition number
    alpha = 1 / np.linalg.cond(W)
    return alpha

def test_norm_nu(v, w, n_mod):
    """
    Check weighted norm of a spatial mode for non-uniform grid.
    v : array of shape (nx, n_modes)
    w : array of shape (nx,)
    n_mod : integer, 0-based index of mode
    """
    W_sqrt = np.sqrt(np.diag(w))      # sqrt of weights
    v_weighted = W_sqrt @ v           # apply sqrt(weights)

    integrand = np.matmul(np.conj(v_weighted[:,n_mod]),v_weighted[:,n_mod]) #has to be 1+0j in theory
    return print('Test Spatial Norm :', integrand)


def test_cross_ortho_nu(v, w):
    """Check weighted cross-orthogonality of spatial modes for non-uniform grid"""
    # Form the weighted inner product matrix
    W_sqrt = np.sqrt(np.diag(w))      # sqrt of weights
    v_weighted = W_sqrt @ v           # apply sqrt(weights)
    orth_matrix = np.conj(v_weighted.T) @ v_weighted  # Hermitian inner product

    num_modes = v.shape[1]
    identity_matrix = np.eye(num_modes)

    tol = 1e-12
    if np.allclose(orth_matrix, identity_matrix, atol=tol):
        print("All weighted spatial modes are properly orthonormal (cross-orthogonality OK).")
    else:
        print("Warning: Weighted spatial modes are NOT perfectly orthonormal!")
        print("Deviation from identity:\n", orth_matrix - identity_matrix)
    return None

########################## Plot and Videos ###################################

def fourier_COC(mat_coc,v,lamb,delta_t,plot=1):
    """Realizes the FFT of the "n_mod" first components of the C.O.D (the signals in Q_d)"""
    
    mat_coc=np.transpose(mat_coc)                    
    (nt,n_m) = np.shape(mat_coc)
    
    #Windowing
    
    win = np.hamming(nt)
    mat_coc = mat_coc*win[None,:,None]
    
    #Fourier transform
    
    fft_temp_mat=scifft.fft(mat_coc, axis=1)
    f_axis=np.linspace(-1/(2*delta_t),1/(2*delta_t),nt)
    
    mat=np.abs(np.mean(fft_temp_mat,axis=0)) #(1,nt,1) -> (nt,1)
    
    if plot==1:
        fig, ax = plt.subplots(dpi=500)
        plt.xlabel(r'$f$ (Hz)')            
        plt.xlim([0,10])
        plt.ylim([-0.1,8.1])
        plt.ylabel(r'Amplitude [-]')
        for i in range(0,5):
            ax.plot(f_axis,np.sqrt(np.pi)*max(np.abs(v[:,i]))/len(f_axis)*np.abs(mat[:,i]), linewidth=1,label="n={}".format(i+1))
        for i in range(5,10):
            ax.plot(f_axis,np.sqrt(np.pi)*max(np.abs(v[:,i]))/len(f_axis)*np.abs(mat[:,i]), linewidth=1,label='_nolegend_')
        plt.title('C.O.D. spectra of each component')
        
        plt.legend()
        #plt.savefig('spectre_cod.pdf', dpi=500, transparent=True)

    return None

def make_wave_gif(sig, list_x, list_t, out_path="wave.gif",
                  fps=15, frame_step=4, dpi=150, figsize=(8,3.5)):
    """
    Create and save a GIF showing the surface sig(t,x) as a line vs x.
      - sig : array nt x nx (time x space)
      - list_x : spatial coordinates (length nx)
      - list_t : time coordinates (length nt)
      - out_path : output filename (GIF)
      - fps : frames per second in GIF
      - frame_step : take every `frame_step`-th time sample as a frame
      - dpi, figsize : figure settings
    """
    nt, nx = sig.shape
    # choose frames (avoid creating a giant GIF)
    frame_indices = np.arange(0, nt, frame_step)
    n_frames = len(frame_indices)

    # figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    (line,) = ax.plot(list_x, sig[0, :], lw=2)
    ax.set_xlabel('x')
    ax.set_ylabel('Surface (A)')
    ax.set_title('Wave oscillation')
    # nice symmetric y-limits with a small margin
    smin, smax = np.min(sig), np.max(sig)
    yr = max(abs(smin), abs(smax))
    ax.set_ylim(-1.05 * yr, 1.05 * yr)
    ax.set_xlim(list_x[0], list_x[-1])
    time_text = ax.text(0.02, 0.92, '', transform=ax.transAxes)

    def init():
        line.set_ydata(np.zeros_like(list_x))
        time_text.set_text('')
        return (line, time_text)

    def update(frame_i):
        i = frame_indices[frame_i]
        y = sig[i, :]
        line.set_ydata(y)
        time_text.set_text(f"t = {list_t[i]:.3f} s  (frame {frame_i+1}/{n_frames})")
        return (line, time_text)

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   init_func=init, blit=True, interval=1000/fps)

    # ensure output directory exists
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save as GIF using Pillow
    print(f"Saving GIF to {out_path}  —  this may take a moment...")
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print("Done.")

def make_spatiotemporal_map(sig, list_x, list_t,
                            out_path="spatiotemporal.png",
                            cmap="RdBu_r", dpi=150,
                            figsize=(8,4)):
    """
    Create a spatio-temporal figure of the wave field sig(t,x):
      - Horizontal axis  : x
      - Vertical axis    : t
      - Color            : amplitude (sig)
    """

    sig = np.asarray(sig)
    nt, nx = sig.shape

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    extent = [list_x[0], list_x[-1], list_t[-1], list_t[0]]  
    # flipped vertically so t=0 is on top

    im = ax.imshow(sig, aspect='auto', extent=extent,
                   cmap=cmap, origin='upper')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Surface amplitude A")

    ax.set_xlabel("x")
    ax.set_ylabel("time t")
    ax.set_title("Spatio-temporal wave pattern")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    print(f"Saved spatio-temporal map to: {out_path}")
    
def make_superposed_profiles(
        sig, list_x, list_t,
        times_to_plot=None,
        n_curves=5,
        cmap_name="viridis",
        out_path="superposed_profiles.pdf",
        dpi=150, figsize=(8,4)
    ):
    """
    Plot several wave snapshots sig(t_i,x) with a colormap, a time colorbar,
    and a transparent background.
    """

    sig = np.asarray(sig)
    nt, nx = sig.shape

    # Select indices of times to plot
    if times_to_plot is None:
        idx = np.linspace(0, nt - 1, n_curves).astype(int)
    else:
        idx = [np.argmin(np.abs(list_t - t)) for t in times_to_plot]

    # Colormap + normalization over curve index
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=len(idx) - 1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Transparent background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Plot each snapshot
    for k, i in enumerate(idx):
        ax.plot(
            list_x, sig[i, :],
            lw=2,
            color=cmap(norm(k))
        )

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Amplitude [-]")
    #ax.set_title("Superposed wave snapshots")
    ax.grid(True, alpha=0.25)

    # Colorbar that maps index → actual time
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("time t [s]")

    # Replace tick labels with actual physical times
    ticks = np.linspace(0, len(idx)-1, min(len(idx), 6))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{list_t[idx[int(t)]]:.1f}" for t in ticks])

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)

    print(f"Saved superposed profile plot to {out_path}")