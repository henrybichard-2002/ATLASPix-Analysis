import numpy as np
from utils import progress_bar

def load_data_numpy(file_path, n_lines=None):
    """Loads data from a text file into a dictionary of NumPy arrays."""
    names = [
        'PackageID', 'Layer', 'Column', 'Row', 'TS', 'TS1', 'TS2',
        'TriggerTS', 'TriggerID', 'ext_TS', 'ext_TS2', 'FIFO_overflow'
    ]
    
    dtypes = {
        'PackageID': np.uint16,
        'Layer': np.uint8,
        'Column': np.uint8,
        'Row': np.uint16,
        'TS': np.uint16,
        'TS1': np.int8,
        'TS2': np.uint16,
        'TriggerTS': np.uint64,
        'TriggerID': np.uint64,
        'ext_TS': np.uint64, 
        'ext_TS2': np.uint64, 
        'FIFO_overflow': np.uint8
    }

    try:
        print("Loading data from file (this may take a moment)...")
        # Use genfromtxt, which handles missing values and bad lines
        raw_data = np.genfromtxt(
            file_path,
            delimiter='\t',
            comments='#',
            dtype=np.float64,
            max_rows=n_lines,
            invalid_raise=False  # <-- Don't raise error on bad lines
        )
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data with numpy.genfromtxt: {e}")
        return None
    data = {}
    for i, name in progress_bar(enumerate(names), description="Processing lines ", total=len(names)):
        col_data = raw_data[:, i]
        target_dtype = dtypes.get(name, np.float64)
        
        # Check if the target is an integer type
        if np.issubdtype(target_dtype, np.integer):
            # Convert NaNs (from bad lines) to 0 before casting to int
            col_data = np.nan_to_num(col_data, nan=0)
            
        data[name] = col_data.astype(target_dtype)

    # --- START: Data Filtering and Statistics ---
    print("Filtering data...")
    total_rows_before = len(data['PackageID'])
    
    if total_rows_before == 0:
        print("No data loaded, skipping filtering.")
        return data
    mask_col = (data['Column'] > 131) | (data['Column'] < 0)
    mask_row = (data['Row'] > 372) | (data['Row'] < 0)
    mask_layer = ~np.isin(data['Layer'], [1, 2, 3, 4])
    mask_fifo = (data['FIFO_overflow'] == 1)
    combined_mask_to_remove = mask_col | mask_row | mask_layer | mask_fifo
    removed_by_col = np.sum(mask_col)
    removed_by_row = np.sum(mask_row)
    removed_by_layer = np.sum(mask_layer)
    removed_by_fifo = np.sum(mask_fifo)
    total_removed = np.sum(combined_mask_to_remove)
    total_kept = total_rows_before - total_removed

    print("\n--- Data Filtering Statistics ---")
    print(f"Total hits loaded:     {total_rows_before}")
    print(f"Removed (Column > 131 or < 0): {removed_by_col}")
    print(f"Removed (Row > 372 or < 0):    {removed_by_row}")
    print(f"Removed (Layer not in [1,2,3,4]): {removed_by_layer}")
    print(f"Removed (FIFO_overflow == 1): {removed_by_fifo}")
    print("---------------------------------")
    print(f"Total unique hits removed: {total_removed}")
    print(f"Total hits remaining:      {total_kept}")
    print("---------------------------------\n")
    mask_to_keep = ~combined_mask_to_remove
    for name in names:
        data[name] = data[name][mask_to_keep]
    if total_kept > 0:
        data['ToT'] = np.mod(data['TS2'] * 2 - data['TS'], 256).astype(np.uint16)
    else:
        data['ToT'] = np.array([], dtype=np.uint16)
    return data