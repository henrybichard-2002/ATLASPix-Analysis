# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:34:12 2026

@author: henry
"""

import os

# --- Directory Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- Detector Geometry ---
PITCH_COL = 0.150
PITCH_ROW = 0.050
DIST_Z = 25.4
N_LAYERS = 4
MAX_COL = 131
MAX_ROW = 372

# --- Beam & Timing Parameters ---
TRIGGER_TS_UNIT_S = 25e-9
MAJOR_BUNCH_BIN_S = 0.5
MINOR_FREQ_HZ = 12.5
MINOR_DURATION_S = 0.07

# --- Analysis Thresholds ---
TOT_THRESHOLDS = [500, 500, 800, 800]