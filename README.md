# ATLASpix Beam Test Analysis

This repository contains the data analysis pipeline for characterizing ATLASpix pixel sensors using a 6 GeV electron beam. The codebase is designed for raw list-mode data processing, track reconstruction, and comprehensive sensor characterization.

## Directory Structure
* `src/`: Primary Python analysis pipelines and scripts.
* `data/`: Decoded `.dat` list-mode data files (ignored by Git).
* `results/`: Output plots and generated datasets.
* `requirements.txt`: Python package dependencies.

## Data Loading
All raw data should be loaded from the `data/` directory using the optimized functions provided in `data_loading_optimized.py`. This ensures efficient memory management and proper NumPy array structuring before passing the data into the analytical pipelines.

## Processing Pipeline
The core analysis relies on a sequential three-stage pipeline. To ensure accurate physics reconstruction, these scripts must be executed in the following order:

### 1. Beam Frequency & Hit Labeling (`Beam_freq_pipeline.py`)
Extracts the major beam structures and optimizes the minor bunch frequency. It assigns precise Phase Time of Flight (pToF) bins to the raw data and labels individual hits based on their temporal beam status.

### 2. Clustering & Crosstalk Separation (`Clustering(perp)_pipeline.py`)
Processes the labeled hits using an optimized anisotropic clustering kernel. It reconstructs hit clusters within a dynamic temporal and spatial window, strictly separating clean events from capacitive crosstalk hits.

### 3. Track Reconstruction & Alignment (`Tracking2_pipeline.py`)
Utilizes the clean clusters to reconstruct particle tracks traversing the detector stack. This pipeline spatially re-aligns the sensor layers to account for physical offsets and generates the final structured tracking datasets.

## Analysis & Diagnostics
Once the core pipelines have generated the structured track and cluster datasets, the following standalone scripts can be used for detailed statistical and physical analysis:

* **Tracking & Sensor Performance:** `Tracking.py`, `Tracking_Stats1.py`, `Sensor_perf.py`
* **Crosstalk Diagnostics:** `TrackXtalk_analysis.py`, `TrackXtalk_analysis2.py`
* **Correlations & Hit Statistics:** `correlation_analysis2.py`, `hit_stats1.py` (alongside previously established `Comprehensive_correlation_analysis.py` and `corr_stats.py`)
* **Visualization Utilities:** `plotting_optimized.py`

## Data Format and Requirements

Due to file size constraints, raw list-mode beam data is not stored in this repository. You must download the raw `.dat` files and place them in the `data/` directory before running the pipeline. 
The analysis scripts expect tabular, list-mode data (tab or space delimited). Each row represents a single hit. The `data_loading_optimized.py` module strictly expects the following 12 columns in order:

* **PackageID**: (uint16) Data packet identifier.
* **Layer**: (uint8) Sensor layer index, constrained to values 1, 2, 3, or 4.
* **Column**: (uint8) Pixel column address, expected range 0 to 131.
* **Row**: (uint16) Pixel row address, expected range 0 to 372.
* **TS**: (uint16) Coarse timestamp.
* **TS1**: (int8) Sub-timestamp 1.
* **TS2**: (uint16) Sub-timestamp 2 (Fine timestamp).
* **TriggerTS**: (uint64) Trigger timestamp.
* **TriggerID**: (uint64) Macroscopic event trigger identifier.
* **ext_TS**: (uint64) External timestamp 1.
* **ext_TS2**: (uint64) External timestamp 2.
* **FIFO_overflow**: (uint8) Buffer overflow flag (hits with value 1 are filtered out).

A template showing the exact structure is available at `data/example_raw_data.txt`.


## Installation
It is recommended to use a virtual environment. Install the required dependencies using:

```bash
pip install -r requirements.txt