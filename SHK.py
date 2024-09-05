import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping


# Constants according to Perdelwitz et al. 2021
lam_K = 3933.66
lam_k1, lam_k2 = lam_K-2.25, lam_K+2.75
lam_H = 3968.47
lam_h1, lam_h2 = lam_H-1.5, lam_H+2.25
HW0 = 0.6
HW12 = 0.25
sigma_SB = 5.6703744191844314e-05

# TODO: change the path
Teff_grid = np.arange(2300, 7001, 100)
grid_dir = '../spe_grid/PFS_SHK_phoenix_grid/'

# Load the spectrum and blaze for order 1 and 2, then normalize the spectrum
def read_norm_spectrum(spe_f, blaze_f):
    data_spe = scipy.io.readsav(spe_f)
    wK, speK = data_spe['w'][1], data_spe['sp'][1]
    speKerr = np.sqrt(speK)
    wH, speH = data_spe['w'][2], data_spe['sp'][2]
    speHerr = np.sqrt(speH)

    data_blaze = scipy.io.readsav(blaze_f)
    blazeK = data_blaze['smnf'][1]
    blazeH = data_blaze['smnf'][2]

    speK = speK / blazeK
    speH = speH / blazeH
    speKerr = speKerr / blazeK
    speHerr = speHerr / blazeH

    return wK, speK, speKerr, wH, speH, speHerr

def value_from_grid(value, grid):
    idx = np.abs(grid - value).argmin()
    return grid[idx]

# Load the theoretical spectrum from the grid according to the Teff
def read_theoretical_spectrum(Teff):
    Teff_str = '%04d' % value_from_grid(Teff, Teff_grid)
    speth_f = grid_dir + f'{Teff_str}K_R130000_3800-4300A_normalized.txt'
    data = np.loadtxt(speth_f)
    w, spe_th, spe_flux = data[:,0], data[:,1], data[:,2]
    return w, spe_th, spe_flux

# Load the ThAr wavelength solution
def read_thar_wavelength(thar_f):
    data_thar = scipy.io.readsav(thar_f)
    w_tharK = data_thar['thar_ws'][1]
    w_tharH = data_thar['thar_ws'][2]
    return w_tharK, w_tharH

# Define the negative likelihood function using CCF to fit the redshift
def neg_like_ccf(z, w, spe_ms, w_th, spe_th):
    # using CCF as the negative likelihood
    w_shift = w/(1.0+z)  # shifting observed onto theoretical
    # interp the theoretical spectrum onto the shifted, observed wavelength
    spe_th_interp = np.interp(w_shift, w_th, spe_th)  
    corr = np.sum((spe_th_interp - spe_ms)**2)
    return corr

# Find the reshift of this observed spectrum using the model spec
def fit_z(w, spe_ms, w_th, spe_th, init_z=0.0001, niter=1000):
    # w, spe_ms = measured wavelength and flux
    # w_th, spe_th = theoretical
    # init_z = initial guess for z
    # niter = number of iterations for basinhopping
    # Written by SXW, 2024/8/26
    
    minimizer_kwargs = {"method": "L-BFGS-B", "args":(w, spe_ms, w_th, spe_th)}
    ret = basinhopping(neg_like_ccf, [init_z], minimizer_kwargs=minimizer_kwargs, niter=niter)
    
    if ret.success:
        return ret.x[0]  # z value best-fit
    else:
        print("Convergence on z failed!")
        return 0.0

# Shift the observed spectrum onto the model's wavelength spectrum
def shift_wavelength(w_th, spe_th, w_tharK, spe_msK, w_tharH, spe_msH):
    z = fit_z(w_tharH, spe_msH, w_th, spe_th)
    w_shiftK = w_tharK / (1+z)
    w_shiftH = w_tharH / (1+z)
    return w_shiftK, w_shiftH

def drop_nan_inf(w, spe, spe_err):
    nan_ind = np.isnan(spe) | np.isinf(spe)
    w = w[~nan_ind]
    spe = spe[~nan_ind]
    spe_err = spe_err[~nan_ind]
    return w, spe, spe_err

# Interpolate the theoretical spectrum to the wavelength of the observed spectrum
def intepolate_speth2wms(wK, wH, w_th, spe_th, spe_flux):
    interp_f_th = interp1d(w_th, spe_th, kind='cubic', fill_value="extrapolate")
    new_spethK = interp_f_th(wK)
    new_spethH = interp_f_th(wH)
    interp_f_flux = interp1d(w_th, spe_flux, kind='cubic', fill_value="extrapolate")
    new_spefluxK = interp_f_flux(wK)
    new_spefluxH = interp_f_flux(wH)
    return new_spethK, new_spefluxK, new_spethH, new_spefluxH

def value_from_grid(value, grid):
    idx = np.abs(grid - value).argmin()
    return grid[idx]

# Calculate the gradient between two spectra following Perdelwitz et al. 2021
def gradient(lam, f1, lam1, f2, lam2):
    return (f2-f1)/(lam2-lam1) * lam + (f2*lam1 - f1*lam2)/(lam1-lam2)

# Calculate the integral flux of a spectrum
def integral_flux(w, spe):
    return np.trapz(spe, w)

# Calculate the emission flux for H or K line
def S_x(w, spe_th, spe_flux, spe_ms, lam_x, lam_x1, lam_x2, HW0, HW12):
    range_x, range_x1, range_x2 = [lam_x-HW0, lam_x+HW0], [lam_x1-HW12, lam_x1+HW12], [lam_x2-HW12, lam_x2+HW12]
    id_x = (w >= range_x[0]) & (w <= range_x[1])
    id_x1 = (w >= range_x1[0]) & (w <= range_x1[1])
    id_x2 = (w >= range_x2[0]) & (w <= range_x2[1])

    f1th, f2th = integral_flux(w[id_x1], spe_th[id_x1]), integral_flux(w[id_x2], spe_th[id_x2])
    f1ms, f2ms = integral_flux(w[id_x1], spe_ms[id_x1]), integral_flux(w[id_x2], spe_ms[id_x2])
    spe_rec_x = spe_ms[id_x] * gradient(w[id_x], f1th, lam_x1, f2th, lam_x2) / gradient(w[id_x], f1ms, lam_x1, f2ms, lam_x2)

    Sx = integral_flux(w[id_x]*1e-8, (spe_rec_x - spe_th[id_x]) * spe_flux[id_x] / spe_th[id_x]) #w from A to cm, spe from normalized to real flux (cgs)
    return Sx

def SHK_logRHK(Teff, wK, spethK, spefluxK, speK,
            wH, spethH, spefluxH, speH):
    '''
    Calculate SHK and RHK
    Now the SHK is only the emission flux of H and K line.
    '''
    S_K = S_x(wK, spethK, spefluxK, speK, lam_K, lam_k1, lam_k2, HW0, HW12)
    S_H = S_x(wH, spethH, spefluxH, speH, lam_H, lam_h1, lam_h2, HW0, HW12)

    SHK = S_H + S_K
    RHK = (S_H + S_K) / (sigma_SB * Teff**4)

    return SHK, np.log10(RHK)

def errorbar(samples):
    pm = np.median(samples)
    return np.percentile(samples, 84.1) - pm, pm - np.percentile(samples, 15.9)

def SHK_logRHK_err_MC(Teff, wK, spethK, spefluxK, speK, speKerr,
                      wH, spethH, spefluxH, speH, speHerr, N_sim=5000):
    '''
    Using Monte Carlo approach to estimate the error of SHK and RHK.
    Do not consider the error of the stellar parameters now.
    '''
    SHK, logRHK = SHK_logRHK(Teff, wK, spethK, spefluxK, speK, wH, spethH, speH, spefluxH)
    SHK_samples, logRHK_samples = [], []
    for n in range(N_sim):
        speK_sim = np.random.normal(speK, speKerr)
        speH_sim = np.random.normal(speH, speHerr)
        SHK_sim, RHK_sim = SHK_logRHK(Teff, wK, spethK, spefluxK, speK_sim,
                                   wH, spethH, spefluxH, speH_sim)
        SHK_samples.append(SHK_sim)
        logRHK_samples.append(RHK_sim)
    SHK_samples, logRHK_samples = np.array(SHK_samples), np.array(logRHK_samples)
    SHK_uerr, SHK_lerr = errorbar(SHK_samples)
    logRHK_uerr, logRHK_lerr = errorbar(logRHK_samples)
    return [SHK, SHK_lerr, SHK_uerr], [logRHK, logRHK_lerr, logRHK_uerr]
    # return [np.median(SHK_samples), SHK_lerr, SHK_uerr], [np.median(logRHK_samples), logRHK_lerr, logRHK_uerr]


if __name__ == '__main__':
    # TODO: change the inputs
    target = 'hd13808'
    # target = 'hd20155'
    # target = 'hip27323'
    spe_f = f'../data/{target}.dat'
    blaze_f = f'../data/nf_n66_15.dat'
    thar_f = f'../data/tharws/n66.4226.tharws.sav'
    Teff = 5002

    wK, speK, speKerr, wH, speH, speHerr = read_norm_spectrum(spe_f, blaze_f)
    w_th, spe_th, spe_flux = read_theoretical_spectrum(Teff)
    w_tharK, w_tharH = read_thar_wavelength(thar_f)
    wK_shift, wH_shift = shift_wavelength(w_th, spe_th, w_tharK, speK, w_tharH, speH)
    wK_new, speK_new, speKerr_new = drop_nan_inf(wK_shift, speK, speKerr)
    wH_new, speH_new, speHerr_new = drop_nan_inf(wH_shift, speH, speHerr)
    spethK_new, spefluxK_new, spethH_new, spefluxH_new = intepolate_speth2wms(wK_new, wH_new, w_th, spe_th, spe_flux)

    # Change how to output the results
    SHK, logRHK = SHK_logRHK(Teff, wK_new, spethK_new, spefluxK_new, speK_new,
                             wH_new, spethH_new, spefluxH_new, speH_new)
    print(SHK, logRHK)

    SHKs, logRHKs = SHK_logRHK_err_MC(Teff, wK_new, spethK_new, spefluxK_new, speK_new, speKerr_new, 
                                      wH_new, spethH_new, spefluxH_new, speH_new, speHerr_new)
    print('SHK:')
    print(SHKs)
    print('logRHK:')
    print(logRHKs)
