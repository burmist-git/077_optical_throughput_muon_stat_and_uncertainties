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
import argparse


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


def uncertainty_fit_function(x, A, C, delta):
    """Doc. string"""

    return A / (np.sqrt(x) + delta) + C


def uncertainty_fit_function_inv(x, A, C, delta):
    """Doc. string"""

    return ( A / (x - C) - delta ) ** 2
    

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


def fit_uncertainty(x, y):
    """Doc. string"""

    #delta_in = 0.001
    delta_in = 0.0
    #C_in = y[np.argmax(x)]
    C_in = 0.0
    A_in = (y[0] - C_in) * (np.sqrt(x[0]) + delta_in)
    
    fit = Minuit(
        loss_uncertainty_fit_function(x, y),
        A=A_in,
        C=C_in,
        delta=delta_in
    )

    fit.errordef = Minuit.LEAST_SQUARES

    fit.errors["A"] = 0.1
    fit.errors["C"] = 0.1
    fit.errors["delta"] = 0.01
    fit.fixed["delta"] = False
    fit.fixed["C"] = False

    fit.migrad()


    return fit.values["A"], fit.values["C"], fit.values["delta"] 


def loss_uncertainty_fit_function(x, y):
    """Doc. string"""

    def ff(A, C, delta):
        diff_squared =  (uncertainty_fit_function(x, A, C, delta) - y) ** 2
        return diff_squared.sum()
    
    return ff 
    

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


def generate_distribution_from_data_inf_stat( data, n_points):
    """Doc. string"""

    
    return np.random.choice(data, size=n_points, replace=False)


def generate_distribution_from_data( data, n_points):
    """Doc. string"""

    
    assert len(data)>=n_points
    assert n_points > 0
    
    n_times = np.floor_divide(len(data), n_points)
    n_points_cut = len(data) - n_times*n_points
    remove_idx = np.random.choice(len(data), size=n_points_cut, replace=False)
    data_norm_size = np.delete(data, remove_idx)

    return data_norm_size.reshape(n_times, n_points)


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


def get_error_estimation( optical_throughput, conf, fit_conf, number_of_trials, chunk_size, max_sigma, iterations):
    """Doc. string"""

    if conf['if_sample_pdf']:    
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
    else:
        if conf['if_assume_assume_infinite_statistics']:
            current_error_estimation = []
            for i in np.arange(number_of_trials):
                current_error_estimation.append(
                    get_sigma_clip_mean(
                        generate_distribution_from_data_inf_stat( optical_throughput, chunk_size),
                        max_sigma,
                        iterations,
                    )
                )
        else:
            sample = generate_distribution_from_data( optical_throughput, chunk_size)
            number_of_trials_form_sample = sample.shape[0]
            current_error_estimation = []
            for i in np.arange(number_of_trials_form_sample):
                current_error_estimation.append(
                    get_sigma_clip_mean(
                        sample[i,:],
                        max_sigma,
                        iterations,
                    )
                )

    return np.array(current_error_estimation)


def main():
    """Doc. string"""


    parser = argparse.ArgumentParser(
        description="The script provides an uncertainty estimation for a given configuration of the calibpipe-calculate-throughput-muon tool and the provided data set."
        f" It also estimates the required statistics to achieve the defined uncertainty."
    )

    # Add arguments
    parser.add_argument(
        "--conf",
        type=str,
        required=True,
        help="Configuration file"
    )
    parser.add_argument(
        "--rel_err",
        type=float,
        default=0.5,
        help="relative uncertainty in percent",
    )

    
    # Parse arguments
    args = parser.parse_args()

    with open(args.conf, 'r') as file:
        conf = yaml.safe_load(file)
        
    
    #data
    h5file=open_file(conf['file'], "a")
    df = pd.DataFrame(h5file.root.dl1.event.telescope.muon.tel_001[:])
    optical_throughput = df['muonefficiency_optical_efficiency'].values
    optical_throughput = optical_throughput[~np.isnan(optical_throughput)]

    
    #throughputconf
    with open(conf['throughputconf'], 'r') as file:
        throughputconf = yaml.safe_load(file)
    
        
    chunk_size = throughputconf['SizeChunking']['chunk_size']
    max_sigma  = throughputconf['SigmaClippingAggregator']['max_sigma']
    iterations = throughputconf['SigmaClippingAggregator']['iterations']

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
        optical_throughput,
        conf,
        fit_conf,
        1000,
        chunk_size,
        max_sigma,
        iterations,
    )
    throughputconf_for_canvas['mean'] = np.mean(optical_throughput_estimation_current)
    throughputconf_for_canvas['standard_error_of_the_mean'] = np.std(optical_throughput_estimation_current)
    #print("current_error_estimation")
    #print("current_error_estimation:  mean = ", throughputconf_for_canvas['mean'])
    #print("current_error_estimation:  std  = ", throughputconf_for_canvas['standard_error_of_the_mean'])


    #
    # Scan the chunk size
    #
    chunk_size_arr = np.arange(50, 1001, 50)
    error_estimation = []
    mean_estimation = []
    for chunk_size_i in chunk_size_arr:
        #print("chunk_size : ", chunk_size_i)
        optical_throughput_estimation = get_error_estimation(
            optical_throughput,
            conf,
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
 
    uncertainty_fit_A, uncertainty_fit_C, uncertainty_fit_delta = fit_uncertainty(chunk_size_arr, rel_error_estimation)

    muon_sample_size_arr = np.arange(20, 1001, 1)
    uncertainty_arr = uncertainty_fit_function(muon_sample_size_arr, uncertainty_fit_A, uncertainty_fit_C, uncertainty_fit_delta)

    estimated_muon_sample_size = uncertainty_fit_function_inv(args.rel_err,uncertainty_fit_A, uncertainty_fit_C, uncertainty_fit_delta)
    
    label_uncertainty="Desired uncertainty   : " + str(round(args.rel_err,3));
    label_sample     ="Estimated muon sample : " + str(round(estimated_muon_sample_size,1));

    
    x = np.linspace(conf['min'],conf['max'], 10*conf['nbins'])
    y_ini = fit_function_from_conf(get_fit_conf(), x)
    y_fit = fit_function_from_conf(fit_conf, x)

    label_uncertainty_str  ='      A : ' + str(round(uncertainty_fit_A, 3))
    label_uncertainty_str +='\n      C : ' + str(round(uncertainty_fit_C, 3))
    label_uncertainty_str +='\n delta : ' + str(round(uncertainty_fit_delta, 4))
    
    with PdfPages(conf['out_pdf']) as pdf:

        fig01_scan_rel=plt.figure(figsize=(15, 10))       
        #plt.title(r"$\mathrm{uncertainty} = \frac{A}{\sqrt{\mathrm{muon~sample~size}} + \mathrm{delta}} + C$", fontsize=30)
        plt.title(r"$\mathrm{err} = \frac{A}{\sqrt{\mathrm{N_{muon}}} + \mathrm{delta}} + C; \mathrm{N_{muon}} = (\frac{A}{\mathrm{err} - C} - \mathrm{delta})^{2} $", fontsize=30)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.ylim(conf['rel_uncertainty_range_min'], conf['rel_uncertainty_range_max'])
        plt.scatter(
            chunk_size_arr,
            rel_error_estimation,
            alpha=1.0,
            c='g',
            s=100,
        )
        plt.plot(
            muon_sample_size_arr,
            uncertainty_arr,
            alpha=0.5,
            linestyle='-',
            linewidth=2,
            c='r',
            label=label_uncertainty_str
        )
        plt.axhline(y=args.rel_err,
                    linestyle='-', linewidth=2, color='black', label=label_uncertainty)
        plt.axvline(x=estimated_muon_sample_size,
                    linestyle='--', linewidth=2, color='black', label=label_sample)
        plt.xlabel('Muon sample size', fontsize=20)
        plt.ylabel('Relative uncertainty of the optical throughput, %', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=20)
        pdf.savefig()
        plt.close()

        
        fig01=plt.figure(figsize=(15, 10))        
        plt.hist(
            optical_throughput, 
            bins=np.linspace(conf['min'],
                             conf['max'],
                             conf['nbins']),
            alpha=0.3,
            hatch='',
            edgecolor='black',
            label='data/simulation',
        )        
        plt.scatter(
            x,
            y_fit,
            alpha=1.0,
            c='g',
            s=10,
            label='PDF from the fit',
        )        
        label_str = 'Throughput measurements'
        label_str +='\n Muon sample size: ' + str(throughputconf_for_canvas['chunk_size'])
        label_str +='\n sigma: ' + str(throughputconf_for_canvas['max_sigma'])
        label_str +='\n iterations: ' + str(throughputconf_for_canvas['iterations'])
        label_str +='\n mean: ' + str(round(throughputconf_for_canvas['mean'], 3))
        label_str +='\n std: ' + str(round(throughputconf_for_canvas['standard_error_of_the_mean'],5))
        label_str +='\n rel. err.: ' + str(round(throughputconf_for_canvas['standard_error_of_the_mean']/throughputconf_for_canvas['mean']*100,3))
        label_str +=' %'
        #
        plt.axvline(x=(throughputconf_for_canvas['mean']-2*throughputconf_for_canvas['standard_error_of_the_mean']),
                    linestyle='--', linewidth=2, color='black', label=label_str)
        plt.axvline(x=(throughputconf_for_canvas['mean']+2*throughputconf_for_canvas['standard_error_of_the_mean']),
                    linestyle='--', linewidth=2, color='black')
        plt.axvspan((throughputconf_for_canvas['mean']-2*throughputconf_for_canvas['standard_error_of_the_mean']),
                    (throughputconf_for_canvas['mean']+2*throughputconf_for_canvas['standard_error_of_the_mean']),
                    color='red',
                    alpha=0.5,
                    hatch='//',
                    edgecolor='black',
                    label="95%c.l.")
        #
        plt.legend(fontsize=20)
        plt.xlabel('Optical throughput for individual muon', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        pdf.savefig()
        plt.close()




if __name__ == "__main__":
    main()
