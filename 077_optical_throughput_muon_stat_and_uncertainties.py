#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import glob
import gc
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.coordinates import SkyCoord, AltAz, angular_separation
import astropy.units as u
from scipy.stats import binned_statistic
from scipy.signal import lombscargle
import h5py
from astropy.io import fits
from astropy.table import Table
from tables import open_file
from astropy.table import join, vstack
from astropy.stats import sigma_clip
from ctapipe.io import read_table 
from ctapipe.instrument import SubarrayDescription
from matplotlib.colors import LogNorm
import math
import yaml
from iminuit import Minuit
from matplotlib.backends.backend_pdf import PdfPages


conf = {
    'file': 'muon-_0deg_0deg_run000008___cta-prod6-2156m-LaPalma-lst-dark-ref-degraded-0.83.h5',
    'throughputconf': './throughput_muon_configuration.yaml',
    'min': 0.1,
    'max': 0.3,
    'nbins': 100,
    'if_fit': True,
    'if_out_pdf': True,
    'out_pdf': 'muon-_0deg_0deg_run000008___cta-prod6-2156m-LaPalma-lst-dark-ref-degraded-0.83.h5.pdf',
}

#conf = {
#    'file': 'muon+_0deg_0deg_run000006___cta-prod6-2156m-LaPalma-mst-nc-dark-ref-degraded-0.83.h5',
#    'throughputconf': './throughput_muon_configuration.yaml',
#    'min': 0.1,
#    'max': 0.3,
#    'nbins': 100,
#    'if_fit': True,
#    'if_out_pdf': True,
#    'out_pdf': 'muon+_0deg_0deg_run000006___cta-prod6-2156m-LaPalma-mst-nc-dark-ref-degraded-0.83.h5.pdf',
#}


def get_fit_conf():
    """Doc. string"""

    fit_conf = {
        'gauss_a_if_fix': False,
        'gauss_a_ampl': 500,
        'gauss_a_x0': 0.175,
        'gauss_a_sig': 0.01,
        'gauss_b_if_fix': False,
        'gauss_b_ampl': 200,
        'gauss_b_x0': 0.18,
        'gauss_b_sig': 0.02,
        'gauss_c_if_fix': False,
        'gauss_c_ampl': 30,
        'gauss_c_x0': 0.15,
        'gauss_c_sig': 0.007,
        'pedestal': 0.0,
    }
    
    return fit_conf


def get_sigma_clip_mean(data, max_sigma, iterations):
    """Doc. string"""

    #print(len(data))
    return np.ma.mean(
        sigma_clip(data,
                   sigma=max_sigma,
                   maxiters=iterations,
                   cenfunc="mean",
                   axis=0,
        ),
        axis=0
    )


def print_conf_to_canvas(conf, fig):
    """Doc. string"""


    figure=fig
    plt.axis('off')
    y_pos = 1.0
    y_step = 0.1
    for key, values in conf.items():
        plt.text(0, y_pos, f"{key}: {values}", fontsize=12, va='top')
        y_pos -= y_step

    return figure


def get_hist_stat(hist_tmp):
    """Doc. string"""


    counts = hist_tmp[0]
    bin_edges = hist_tmp[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Weighted standard deviation
    if(np.sum(counts) > 0):
        mean = np.average(bin_centers, weights=counts)
        variance = np.average((bin_centers - mean)**2, weights=counts)
        std = np.sqrt(variance)
    else:
        mean = 0
        variance = 0
        std = 0

    print("mean     = ",mean)
    print("std      = ",std)
    print("sum      = ",np.sum(counts))


def gauss_pedestal(x, A, mu, sigma, pedestal = 0.0):
    """Doc. string"""


    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + pedestal


def fit_function(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, pedestal):
    """Doc. string"""


    return (
        gauss_pedestal(x, A1, mu1, sigma1, pedestal) + 
        gauss_pedestal(x, A2, mu2, sigma2) +
        gauss_pedestal(x, A3, mu3, sigma3)
    )


def fit_function_from_conf(conf, x):
    """Doc. string"""


    return fit_function(x, 
                        conf['gauss_a_ampl'], 
                        conf['gauss_a_x0'], 
                        conf['gauss_a_sig'],
                        conf['gauss_b_ampl'], 
                        conf['gauss_b_x0'], 
                        conf['gauss_b_sig'],
                        conf['gauss_c_ampl'], 
                        conf['gauss_c_x0'], 
                        conf['gauss_c_sig'],
                        conf['pedestal'])


def fit_optical_throughput(optical_throughput_x, optical_throughput_y, fit_conf):
    """Doc. string"""

    
    fit = Minuit(
        loss(optical_throughput_x, optical_throughput_y),
        A1=fit_conf['gauss_a_ampl'],
        mu1=fit_conf['gauss_a_x0'],
        sigma1=fit_conf['gauss_a_sig'],
        A2=fit_conf['gauss_b_ampl'],
        mu2=fit_conf['gauss_b_x0'],
        sigma2=fit_conf['gauss_b_sig'],
        A3=fit_conf['gauss_c_ampl'],
        mu3=fit_conf['gauss_c_x0'],
        sigma3=fit_conf['gauss_c_sig'],
        pedestal= fit_conf['pedestal'],
    )

    fit.errordef = Minuit.LEAST_SQUARES

    fit.errors["A1"] = 0.01
    fit.errors["mu1"] = 0.1
    fit.errors["sigma1"] = 0.1
    if (fit_conf['gauss_a_if_fix']):
        fit.fixed["A1"] = True
        fit.fixed["mu1"] = True
        fit.fixed["sigma1"] = True

    fit.errors["A2"] = 0.01
    fit.errors["mu2"] = 0.1
    fit.errors["sigma2"] = 0.1
    if (fit_conf['gauss_b_if_fix']):
        fit.fixed["A2"] = True
        fit.fixed["mu2"] = True
        fit.fixed["sigma2"] = True

    fit.errors["A3"] = 0.01
    fit.errors["mu3"] = 0.1
    fit.errors["sigma3"] = 0.1
    if (fit_conf['gauss_c_if_fix']):
        fit.fixed["A3"] = True
        fit.fixed["mu3"] = True
        fit.fixed["sigma3"] = True

    fit.errors["pedestal"] = 0.001


    fit.migrad()

    fit_conf_out = get_fit_conf()

    fit_conf_out['gauss_a_ampl'] = fit.values["A1"]
    fit_conf_out['gauss_a_x0'] = fit.values["mu1"]
    fit_conf_out['gauss_a_sig'] = fit.values["sigma1"]
    fit_conf_out['gauss_b_ampl'] = fit.values["A2"]
    fit_conf_out['gauss_b_x0'] = fit.values["mu2"]
    fit_conf_out['gauss_b_sig'] = fit.values["sigma2"]
    fit_conf_out['gauss_c_ampl'] = fit.values["A3"]
    fit_conf_out['gauss_c_x0'] = fit.values["mu3"]
    fit_conf_out['gauss_c_sig'] = fit.values["sigma3"]
    fit_conf_out['pedestal'] = fit.values["pedestal"]

    return fit_conf_out


def loss(x, y):
    """Doc. string"""

    
    def loss_function(A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, pedestal):
        diff_squared = (fit_function(x, 
                                     A1, mu1, sigma1, 
                                     A2, mu2, sigma2, 
                                     A3, mu3, sigma3, 
                                     pedestal) - y) ** 2
        return diff_squared.sum()
    return loss_function


def generate_distribution_from_function( fit_conf, x_min, x_max, n_points): 
    """Doc. string"""


    n_bins = 100
    x = np.linspace(x_min, x_max, n_bins)
    y = fit_function_from_conf(fit_conf, x)
    y_max = y.max() 
    y_max = y_max + y_max/10.0

    approximate_generator_efficiency = np.sum(y) / (n_bins-1) / y_max / 2.0

    n = int(n_points / approximate_generator_efficiency)

    x_rand = np.random.uniform(x_min, x_max, n)
    y_rand = np.random.uniform(0, y_max, n)
    y_rand_x = fit_function_from_conf(fit_conf, x_rand)

    x_rand = x_rand[y_rand<y_rand_x]

    if len(x_rand) > n_points :
        return x_rand[:n_points]
    
    return x_rand


def test_generate_distribution_from_function( fit_conf, x_min, x_max, n_points):
    """Doc. string"""


    fig03=plt.figure(figsize=(10, 10))
    plt.hist(
        generate_distribution_from_function( fit_conf, x_min, x_max, n_points), 
        bins=np.linspace(conf['min'],
                         conf['max'],
                         conf['nbins']),
    )
    plt.show()


def get_error_estimation( fit_conf, number_of_trials, chunk_size, max_sigma, iterations):
    """Doc. string"""


    current_error_estimation = []
    for i in np.arange(number_of_trials):
        current_error_estimation.append(
            get_sigma_clip_mean(
                generate_distribution_from_function(
                    fit_conf,
                    conf['min'],
                    conf['max'],
                    chunk_size,
                ),
                max_sigma,
                iterations,
            )
        )


    return np.array(current_error_estimation)


def main():
    """Doc. string"""

    
    #data
    h5file=open_file(conf['file'], "a")
    df = pd.DataFrame(h5file.root.dl1.event.telescope.muon.tel_001[:])
    optical_throughput = df['muonefficiency_optical_efficiency'].values
    optical_throughput = optical_throughput[~np.isnan(optical_throughput)]

    
    #throughputconf
    with open(conf['throughputconf'], 'r') as file:
        throughputconf = yaml.safe_load(file)

    chunk_size = throughputconf['CalculateThroughputWithMuons']['chunk_size']
    max_sigma  = throughputconf['CalculateThroughputWithMuons']['SigmaClippingAggregator']['max_sigma']
    iterations = throughputconf['CalculateThroughputWithMuons']['SigmaClippingAggregator']['iterations']

    throughputconf_for_canvas = {
        'chunk_size': chunk_size,
        'max_sigma': max_sigma,
        'iterations': iterations,
        'mean': np.nan,
        'standard_error_of_the_mean': np.nan,
    }

    
    hist_optical_throughput = np.histogram(optical_throughput, 
                                           bins=np.linspace(conf['min'],
                                                            conf['max'],
                                                            conf['nbins']),
                                           )


    optical_throughput_y = hist_optical_throughput[0]
    optical_throughput_x = ((np.roll(hist_optical_throughput[1], 1) + hist_optical_throughput[1]) / 2.0)[1:]

    
    if conf['if_fit'] :
        fit_conf = fit_optical_throughput(optical_throughput_x, optical_throughput_y, get_fit_conf())
    else:
        fit_conf = get_fit_conf()


    #
    # Estimate the current error
    #
    optical_throughput_estimation_current = get_error_estimation(
        fit_conf,
        1000,
        chunk_size,
        max_sigma,
        iterations,
    )
    throughputconf_for_canvas['mean'] = np.mean(optical_throughput_estimation_current)
    throughputconf_for_canvas['standard_error_of_the_mean'] = np.std(optical_throughput_estimation_current)
    print("current_error_estimation")
    print("current_error_estimation:  mean = ", throughputconf_for_canvas['mean'])
    print("current_error_estimation:  std  = ", throughputconf_for_canvas['standard_error_of_the_mean'])


    #
    # Scan the chunk size
    #
    chunk_size_arr = np.arange(50, 1001, 50)
    error_estimation = []
    mean_estimation = []
    for chunk_size_i in chunk_size_arr:
        print("chunk_size : ", chunk_size_i)
        optical_throughput_estimation = get_error_estimation(
            fit_conf,
            1000,
            chunk_size_i,
            max_sigma,
            iterations,
        )
        error_estimation.append(np.std(optical_throughput_estimation))
        mean_estimation.append(np.mean(optical_throughput_estimation))


        
    error_estimation = np.array(error_estimation)
    mean_estimation = np.array(mean_estimation)
    rel_error_estimation = error_estimation/mean_estimation * 100.0
    
    x = np.linspace(conf['min'],conf['max'], 10*conf['nbins'])
    y_ini = fit_function_from_conf(get_fit_conf(), x)
    y_fit = fit_function_from_conf(fit_conf, x)

    
    with PdfPages(conf['out_pdf']) as pdf:

        fig01=plt.figure(figsize=(10, 10))

        plt.hist(
            optical_throughput, 
            bins=np.linspace(conf['min'],
                             conf['max'],
                             conf['nbins']),
            alpha=0.5,
            label='data',
        )
        
        #plt.scatter(
        #    x,
        #    y_ini,
        #    alpha=1.0,
        #    c='y',
        #    s=10,
        #    label='initial',
        #)
        
        plt.scatter(
            x,
            y_fit,
            alpha=1.0,
            c='g',
            s=10,
            label='Fit',
        )
        plt.axvline(x=(throughputconf_for_canvas['mean']-2*throughputconf_for_canvas['standard_error_of_the_mean']),
                    linestyle='--', linewidth=1, label=f'95 % c.l.')
        plt.axvline(x=(throughputconf_for_canvas['mean']+2*throughputconf_for_canvas['standard_error_of_the_mean']),
                    linestyle='--', linewidth=1)
        
        plt.legend()
        plt.xlabel('Optical throughput for single muon')
        if conf['if_out_pdf'] :
            pdf.savefig()
        else:
            plt.show()
        plt.close()


        fig01_meas=plt.figure(figsize=(10, 10))

        plt.hist(
            optical_throughput_estimation_current, 
            bins=30,
            alpha=0.5,
            label='Optical throughput measurements',
        )

        plt.legend()
        plt.xlabel('Optical throughput measurements')
        if conf['if_out_pdf'] :
            pdf.savefig()
        else:
            plt.show()
        plt.close()


        fig01_scan_mean=plt.figure(figsize=(15, 10))

        plt.scatter(
            chunk_size_arr,
            mean_estimation,
            alpha=1.0,
            c='g',
            s=10,
        )

        plt.xlabel('Muon sample size')
        plt.ylabel('Optical throughput')

        plt.ylim(conf['min'], conf['max'])

        if conf['if_out_pdf'] :
            pdf.savefig()
        else:
            plt.show()
        plt.close()

        fig01_scan_std=plt.figure(figsize=(15, 10))
        
        plt.scatter(
            chunk_size_arr,
            error_estimation,
            alpha=1.0,
            c='g',
            s=10,
        )

        plt.xlabel('Muon sample size')
        plt.ylabel('Absolute uncertainty of the optical throughput')
        if conf['if_out_pdf'] :
            pdf.savefig()
        else:
            plt.show()
        plt.close()



        fig01_scan_rel=plt.figure(figsize=(15, 10))
        
        plt.scatter(
            chunk_size_arr,
            rel_error_estimation,
            alpha=1.0,
            c='g',
            s=10,
        )

        plt.xlabel('Muon sample size')
        plt.ylabel('Relative uncertainty of the optical throughput')
        if conf['if_out_pdf'] :
            pdf.savefig()
        else:
            plt.show()
        plt.close()

        

        
        #
        fig02=plt.figure(figsize=(15, 5))
        fig02=print_conf_to_canvas(conf, fig02)
        if conf['if_out_pdf'] :
            pdf.savefig()
        else:
            plt.show()
        plt.close()
        
        #
        fig03=plt.figure(figsize=(15, 5))
        fig03=print_conf_to_canvas(throughputconf_for_canvas, fig03)
        if conf['if_out_pdf'] :
            pdf.savefig()
        else:
            plt.show()
        plt.close()
     

if __name__ == "__main__":
    main()
