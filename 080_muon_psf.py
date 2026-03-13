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
from scipy.stats import skew
from scipy.stats import kurtosis
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import copy


def save_to_csv(conf, x, y, xe, ye):
    pd.DataFrame(
        {
            "x": x,
            "y": y,
            "xe": xe,
            "ye": ye,
        }
    ).to_csv(str(conf['file'] + str(".csv")), sep=" ", index=False)


def analyze(conf):
    """analyze one file"""
    

    #data
    h5file=open_file(conf['file'], "a")
    #df = pd.DataFrame(h5file.root.dl1.monitoring.telescope.calibration.optical_psf.tel_001[:])
    #optical_throughput = df['muonefficiency_optical_efficiency'].values
    #optical_throughput = optical_throughput[~np.isnan(optical_throughput)]

    slope=h5file.root.dl1.monitoring.telescope.calibration.optical_psf.tel_001[:][0][0]
    intersect=h5file.root.dl1.monitoring.telescope.calibration.optical_psf.tel_001[:][0][1]
    chi2=h5file.root.dl1.monitoring.telescope.calibration.optical_psf.tel_001[:][0][4]
    n_ev=h5file.root.dl1.monitoring.telescope.calibration.optical_psf.tel_001[:][0][9]
    r_r=h5file.root.dl1.monitoring.telescope.calibration.optical_psf.tel_001[:][0][10]
    r_r_err=h5file.root.dl1.monitoring.telescope.calibration.optical_psf.tel_001[:][0][11]
    r_w=h5file.root.dl1.monitoring.telescope.calibration.optical_psf.tel_001[:][0][12]
    r_w_err=h5file.root.dl1.monitoring.telescope.calibration.optical_psf.tel_001[:][0][13]


    rr=np.linspace(np.min(r_r),np.max(r_r),100)
    ww=slope * rr + intersect


    save_to_csv(conf, r_r, r_w, r_r_err, r_w_err)
    
    print(n_ev)
    
    
    #throughputconf
    with open(conf['psfconf'], 'r') as file:
        psfconf = yaml.safe_load(file)
    

    #print(df)
    print(psfconf)


    with PdfPages(str(conf['file'] + str(".pdf"))) as pdf:
        plt.figure(figsize=(15, 10))
        plt.errorbar(
            x=r_r,
            y=r_w,
            xerr=r_r_err,
            yerr=r_w_err,
            fmt='o',
            capsize=4,
            label=conf['file']
        )

        plt.plot(
            rr,
            ww,
        )
        plt.legend()
        plt.ylim(0,0.1)
        plt.xlim(0.8,1.3)        
        plt.xlabel("ring r, deg")
        plt.ylabel("ring w, deg")
        plt.title(f"ring w vs. ring r. chi2: {chi2}")
        pdf.savefig()
        plt.close()

    

    h5file.close()


def main():
    """Main program"""


    parser = argparse.ArgumentParser(
        description="The script provides an uncertainty estimation for a given configuration of the calibpipe-calculate-throughput-muon tool and the provided data set."
        f" It also estimates the required statistics to achieve the defined uncertainty."
        f" The script generates an output PDF file."
        f" The first page plots uncertainty versus muon sample size, with horizontal and vertical lines showing the desired uncertainty and the required muon sample size, respectively."
        f" The second page shows the optical throughput distribution for a single measurement available in the input file, along with the 95 percent confidence"
        f" level of the optical throughput calculated using the calibpipe-calculate-throughput-muon tool."
        f" Its configuration is described in the YAML file (e.g., throughput_muon_configuration.yaml)."
    )

    # Add arguments
    parser.add_argument(
        "--conf",
        type=str,
        required=True,
        help="Configuration file"
    )

    
    # Parse arguments
    args = parser.parse_args()

    with open(args.conf, 'r') as file:
        conf = yaml.safe_load(file)
        
    file_list = list(conf['file'])

    for the_file in conf['file']:
        print(the_file)
        conf['file'] = the_file
        analyze(conf)


if __name__ == "__main__":
    main()
