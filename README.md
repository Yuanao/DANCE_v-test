🌌 DANCE FRB Clustering Demo

This repository provides a test version of the DANCE (Density-based Analysis for Narrow-band Clustering and Extraction) framework for Fast Radio Burst (FRB) detection and clustering.
It demonstrates how to use the DANCE pipeline on a PSRCHIVE .ar data file to isolate transient radio bursts through unsupervised clustering.

📘 Overview

The workflow is implemented through three main scripts:

File	Role
Desffrb_t.py	Core DANCE implementation — FRB detection, clustering, and plotting
desffrb_tools.py	Utility functions — PSRCHIVE I/O, data normalization, DBSCAN clustering
test.py	Entry point — orchestrates the full pipeline

The pipeline performs:

(1) Reading and preprocessing of .ar archive data
(2) RFI masking and denoising
(3) FRB signal clustering (DBSCAN-like algorithm)
(4) Visualization of detected burst regions

🧩 Repository Structure

├── test.py                # Main execution script

├── Desffrb_t.py           # Core DANCE clustering implementation

├── desffrb_tools.py       # Utility functions for signal processing and plotting

├── 220912-45-564.555153408.ar   # Example FRB archive data (stored via Git LFS)

└── README.md


⚙️ Requirements

Create and activate an environment:
    conda create -n dance python=3.11 -y
    conda activate dance
    
Install dependencies:
    pip install numpy scipy matplotlib pandas scikit-learn pywt h5py
    conda install -c conda-forge psrchive

🚀 Usage

Run the clustering analysis directly from the command line:

    python test.py 220912-45-564.555153408.ar

Optional arguments:

Most parameters (e.g., clustering thresholds, smoothing factors, plot configuration) can be adjusted in "test.py" .
     
What Happens:

test.py loads the .ar file through desffrb_tools;
     
Performs RFI masking;
     
Applies DANCE clustering (based on density and thresholding);
     
Displays five figures: Raw spectrum, RFI-masked spectrum,Smoothed intensity map, Clustered regions, FRB detection result.
     
By default, plots are shown interactively (display=True) and not saved.
You can change this behavior in test.py: plot_data(det, display=False, save=True)

💾 Demo Data

The included file:

220912-45-564.555153408.ar

is a real pulsar archive file used to demonstrate the DANCE pipeline.
It is stored using Git LFS, so make sure LFS is enabled before cloning:
"
git lfs install
git clone https://github.com/Yuanao/DANCE_v-test.git
git lfs pull
"
Alternatively, download directly:
curl -L -o demo.ar "https://github.com/Yuanao/DANCE_v-test/raw/main/220912-45-564.555153408.ar"

🧠 Algorithm Summary
Framework: Density-based clustering (DBSCAN-like)
Feature space: Time–frequency intensity distribution
Goal: Identify compact, high-density emission regions corresponding to transient FRB events

📬 Contact
Author: M. Yuan yuanmao@bao.ac.cn

