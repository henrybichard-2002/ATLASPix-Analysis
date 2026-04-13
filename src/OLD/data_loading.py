import pandas as pd

def load_data(file_path):
    """Loads data from a text file into a pandas DataFrame."""
    df = pd.read_csv(
        file_path,
        sep='\t',
        comment='#',
        header=None,
        names=[
            'PackageID', 'Layer', 'Column', 'Row', 'TS', 'TS1', 'TS2',
            'TriggerTS', 'TriggerID', 'ext_TS', 'ext_TS2', 'FIFO_overflow'
        ]
    )
    df = df.assign(ToT=(df['TS2'] * 2 - df['TS']) % 256)
    return df
