"""Data loading from CSV files."""

import pandas as pd
from pathlib import Path
import numpy as np


class DataLoader:
    """Loads acceleration data from CSV files."""
    
    def __init__(self, data_dir=None):
        """If data_dir is None, uses 'input_files' relative to script location."""
        if data_dir is None:
            script_dir = Path(__file__).parent
            self.data_dir = script_dir / 'input_files'
        else:
            self.data_dir = Path(data_dir)
    
    def load_mode_data(self, mode_number, time_col='time', acc_prefix='L', acc_suffix='AccX_filtered'):
        """Load acceleration data for a specific mode or ground motion. Returns (time, *acc_data)."""
        if mode_number == 'ground_motion':
            csv_path = self.data_dir / 'ground_motion_excitation.csv'
        else:
            csv_path = self.data_dir / f'mode_{mode_number}_excitation.csv'
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in CSV")
        
        time = df[time_col].values
        acc_data = []
        floor_num = 1
        while True:
            acc_col = f'{acc_prefix}{floor_num}{acc_suffix}'
            if acc_col in df.columns:
                acc_data.append(df[acc_col].values)
                floor_num += 1
            else:
                break
        
        if not acc_data:
            raise ValueError(f"No acceleration columns found with pattern '{acc_prefix}*{acc_suffix}'")
        
        return (time, *acc_data)
    
    def get_num_floors(self, mode_number=1):
        """Get number of floors from data file."""
        _, *acc_data = self.load_mode_data(mode_number)
        return len(acc_data)

