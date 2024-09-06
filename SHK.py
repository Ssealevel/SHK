import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import os

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
grid_dir = './PFS_SHK_phoenix_grid/'
verify_plot_dir = './verify_plot/'

# Load the spectrum and blaze for order 1 and 2, then normalize the spectrum
def read_norm_spectrum(spe_f, blaze_f, use_thar_w=False, thar_f=None, use_drop_nan_inf=True):
    data_spe = scipy.io.readsav(spe_f)
    if use_thar_w:
        data_thar = scipy.io.readsav(thar_f)
        wK = data_thar['thar_ws'][1]
        wH = data_thar['thar_ws'][2]
        speK = data_spe['sp'][1]
        speH = data_spe['sp'][2]
        speKerr = np.sqrt(speK)
        speHerr = np.sqrt(speH)
    else:
        try:
            wK, speK = data_spe['sw'][1], data_spe['sp'][1]
            wH, speH = data_spe['sw'][2], data_spe['sp'][2]
        except:
            wK, speK = data_spe['w'][1], data_spe['sp'][1]
            wH, speH = data_spe['w'][2], data_spe['sp'][2]
        speKerr = np.sqrt(speK)
        speHerr = np.sqrt(speH)

    data_blaze = scipy.io.readsav(blaze_f)
    try:
        blazeK = data_blaze['smsp'][1]
        blazeH = data_blaze['smsp'][2]
    except:
        blazeK = data_blaze['smnf'][1]
        blazeH = data_blaze['smnf'][2]

    speK = speK / blazeK
    speH = speH / blazeH
    speKerr = speKerr / blazeK
    speHerr = speHerr / blazeH

    if use_drop_nan_inf:
        wK, speK, speKerr = drop_nan_inf(wK, speK, speKerr)
        wH, speH, speHerr = drop_nan_inf(wH, speH, speHerr)

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
    if np.max(w_shift) > np.max(w_th) or np.min(w_shift) < np.min(w_th):
        return 100*len(spe_ms)
    spe_th_interp = np.interp(w_shift, w_th, spe_th) 
    # corr = np.sum((spe_th_interp - spe_ms)**2)
    id_corr = np.where((w_shift < lam_H-1) | (w_shift > lam_H+1))[0]
    corr = np.sum((spe_th_interp[id_corr] - spe_ms[id_corr])**2)
    return corr

def verify_plot_fit_z(w, spe_ms, w_th, spe_th, z_best, fig_name):
    plt.figure(figsize=(12, 6), dpi=150)
    w_shifted = w / (1.0 + z_best)   
    plt.plot(w_shifted, spe_ms, label=f"Shifted Observed Spectrum (z={z_best:.6f})", color="blue")  #Plot the observed spectrum
    plt.plot(w_th, spe_th, label=f"Theoretical Spectrum", color="red") #Plot the shifted Observed spectrum
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Theoretical vs Shifted Observed Spectrum')
    plt.legend()
    plt.xlim((min(w_shifted), max(w_shifted)))
    plt.savefig(verify_plot_dir + fig_name)
    plt.close()

# Find the reshift of this observed spectrum using the model spec
def fit_z(w, spe_ms, w_th, spe_th, init_z=0.0001, niter=1000, 
          verify=True, target='', filename=''):
    
    fit_w, fit_spe_ms = w[:], spe_ms[:] / np.percentile(spe_ms, 90.)

    buffer = 10.0
    obsw_max = np.max(fit_w) + buffer
    obsw_min = np.min(fit_w) - buffer
    ind_ref = np.where((w_th>obsw_min)&(w_th<obsw_max))[0]
    wv_ref = w_th[ind_ref]
    spe_ref = spe_th[ind_ref]
    # spe_ref = spe_ref / np.max(spe_ref)
    spe_ref = spe_ref / np.percentile(spe_ref, 90.)

    verify_plot_fit_z(fit_w, fit_spe_ms, wv_ref, spe_ref, init_z, 
                      fig_name=f'{target}_{filename}_fit_z_verify_plot_z_{init_z:.6f}.png')

    minimizer_kwargs = {"method": "L-BFGS-B", "args":(fit_w, fit_spe_ms, wv_ref, spe_ref)}
    ret = basinhopping(neg_like_ccf, [init_z], minimizer_kwargs=minimizer_kwargs, niter=niter)
    
    if ret.success:
        z_best = ret.x[0]
        # Call verify_plot if verify is True
        if verify:
            verify_plot_fit_z(fit_w, fit_spe_ms, wv_ref, spe_ref, z_best,
                              fig_name=f'{target}_{filename}_fit_z_verify_plot_z_{z_best:.6f}.png')
        return z_best  # z value best-fit
    else:
        print("Convergence on z failed!")
        if verify:
            verify_plot_fit_z(fit_w, fit_spe_ms, wv_ref, spe_ref, 0.,
                              fig_name=f'{target}_{filename}_fit_z_verify_plot_z_0.png')
        return 0.0

# Shift the observed spectrum onto the model's wavelength spectrum
def shift_wavelength(w_th, spe_th, w_tharK, spe_msK, w_tharH, spe_msH,
                     verify=True, target='', filename=''):
    z = fit_z(w_tharH, spe_msH, w_th, spe_th, 
              verify=verify, target=target, filename=filename)
    print(z) # for test
    w_shiftK = w_tharK / (1+z)
    w_shiftH = w_tharH / (1+z)
    if verify:
        verify_plot_fit_z(w_tharK, spe_msK/np.percentile(spe_msK,90), w_th, spe_th/np.percentile(spe_th,90), z,
                        fig_name=f'{target}_{filename}_shifted_K_verify_plot.png')
        verify_plot_fit_z(w_tharH, spe_msH/np.percentile(spe_msH,90), w_th, spe_th/np.percentile(spe_th,90), z,
                        fig_name=f'{target}_{filename}_shifted_H_verify_plot.png')
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

def verify_plot_SHK_gradient(w, spe_th, spe_ms, lam_x, f1th, f1ms, lam_x1, f2th, f2ms, lam_x2, fig_name):
    spe_rec = spe_ms * gradient(w, f1th, lam_x1, f2th, lam_x2) / gradient(w, f1ms, lam_x1, f2ms, lam_x2)
    plt.figure(dpi=150, figsize=(12, 6))
    plt.plot(w, spe_rec, label='Rectificated Observation')
    plt.plot(w, spe_th, label='PHOENIX')
    xlim = [lam_x1-5, lam_x2+5]
    plt.xlim(xlim)
    ind = (w >= xlim[0]) & (w <= xlim[1])
    plt.ylim(0, 1.2*np.max(spe_th[ind]))
    plt.axvline(lam_x-HW0, color='r', linestyle='--')
    plt.axvline(lam_x+HW0, color='r', linestyle='--')
    plt.axvline(lam_x1-HW12, color='b', linestyle='--')
    plt.axvline(lam_x1+HW12, color='b', linestyle='--')
    plt.axvline(lam_x2-HW12, color='b', linestyle='--')
    plt.axvline(lam_x2+HW12, color='b', linestyle='--')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Theoretical vs Rectification of the Observation')
    plt.legend()
    plt.savefig(verify_plot_dir + fig_name)
    plt.close()
    return

# Calculate the emission flux for H or K line
def S_x(w, spe_th, spe_flux, spe_ms, lam_x, lam_x1, lam_x2, HW0, HW12,
        verify=True, verify_plot_name=''):
    range_x, range_x1, range_x2 = [lam_x-HW0, lam_x+HW0], [lam_x1-HW12, lam_x1+HW12], [lam_x2-HW12, lam_x2+HW12]
    id_x = (w >= range_x[0]) & (w <= range_x[1])
    id_x1 = (w >= range_x1[0]) & (w <= range_x1[1])
    id_x2 = (w >= range_x2[0]) & (w <= range_x2[1])

    f1th, f2th = integral_flux(w[id_x1], spe_th[id_x1]), integral_flux(w[id_x2], spe_th[id_x2])
    f1ms, f2ms = integral_flux(w[id_x1], spe_ms[id_x1]), integral_flux(w[id_x2], spe_ms[id_x2])
    spe_rec_x = spe_ms[id_x] * gradient(w[id_x], f1th, lam_x1, f2th, lam_x2) / gradient(w[id_x], f1ms, lam_x1, f2ms, lam_x2)

    if verify:
        verify_plot_SHK_gradient(w, spe_th, spe_ms, lam_x, f1th, f1ms, lam_x1, f2th, f2ms, lam_x2, verify_plot_name)

    Sx = integral_flux(w[id_x]*1e-8, (spe_rec_x - spe_th[id_x]) * spe_flux[id_x] / spe_th[id_x]) #w from A to cm, spe from normalized to real flux (cgs)
    return Sx

def SHK_logRHK(Teff, wK, spethK, spefluxK, speK,
            wH, spethH, spefluxH, speH, 
            verify=False, target='', filename=''):
    '''
    Calculate SHK and RHK
    Now the SHK is only the emission flux of H and K line.
    '''
    S_K = S_x(wK, spethK, spefluxK, speK, lam_K, lam_k1, lam_k2, HW0, HW12,
              verify=verify, verify_plot_name=f'{target}_{filename}_SK_Rectification_verify_plot.png')
    S_H = S_x(wH, spethH, spefluxH, speH, lam_H, lam_h1, lam_h2, HW0, HW12,
              verify=verify, verify_plot_name=f'{target}_{filename}_SH_Rectification_verify_plot.png')

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
    SHK, logRHK = SHK_logRHK(Teff, wK, spethK, spefluxK, speK, wH, spethH, spefluxH, speH)
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
    # return [SHK, SHK_lerr, SHK_uerr], [logRHK, logRHK_lerr, logRHK_uerr]
    return [np.median(SHK_samples), SHK_lerr, SHK_uerr], [np.median(logRHK_samples), logRHK_lerr, logRHK_uerr]


if __name__ == '__main__':
    # TODO: change the inputs
    # target = 'hd20155'
    # target = 'hip27323'

    target = 'hd13808'
    spe_fs = [f'./test_data/spectra/{target}.dat']
    blaze_f = f'./test_data/blaze/nf_n66_15.dat' # hd13808
    thar_f = f'./test_data/tharws/n66.4226.tharws.sav' # hd13808
    Teff = 5002

    # target = 'hd197481' # AU Mic
    # # spe_f = f'./test_data/spectra/197481_data/rn68.2630'
    # spe_dir = './test_data/spectra/197481_data/'
    # blaze_f = f'./test_data/blaze/nf_n68_11.dat' # AU Mic (hd197481)
    # thar_f = f'./test_data/tharws/n68.2694.tharws.sav' # AU Mic (hd197481)
    # Teff = 3688
    # spe_fs = [spe_dir + f'rn68.{n}' for n in range(2631,2676)]

    results = []
    for spe_f in spe_fs:
        wK, speK, speKerr, wH, speH, speHerr = \
            read_norm_spectrum(spe_f, blaze_f, use_thar_w=True, thar_f=thar_f, use_drop_nan_inf=True)
        w_th, spe_th, spe_flux = read_theoretical_spectrum(Teff)
        wK_shift, wH_shift = shift_wavelength(w_th, spe_th, wK, speK, wH, speH,
                                            verify=True, target=target, filename=os.path.basename(spe_f))
        wK_new, speK_new, speKerr_new = drop_nan_inf(wK_shift, speK, speKerr)
        wH_new, speH_new, speHerr_new = drop_nan_inf(wH_shift, speH, speHerr)
        spethK_new, spefluxK_new, spethH_new, spefluxH_new = \
            intepolate_speth2wms(wK_new, wH_new, w_th, spe_th, spe_flux)

        # Change how to output the results
        SHK, logRHK = SHK_logRHK(Teff, wK_new, spethK_new, spefluxK_new, speK_new,
                                wH_new, spethH_new, spefluxH_new, speH_new,
                                verify=True, target=target, filename=os.path.basename(spe_f))
        print(SHK, logRHK)

        SHKs, logRHKs = SHK_logRHK_err_MC(Teff, wK_new, spethK_new, spefluxK_new, speK_new, speKerr_new, 
                                        wH_new, spethH_new, spefluxH_new, speH_new, speHerr_new)
        print('SHK:')
        print(SHKs)
        print('logRHK:')
        print(logRHKs)

        results.append(np.hstack((SHKs, logRHKs)))
    
    results = np.array(results)
    np.savetxt(f'./{target}_SHK_logRHK_results.txt', results, fmt='%.6f',
               header='SHK, SHK_lerr, SHK_uerr, logRHK, logRHK_lerr, logRHK_uerr')
