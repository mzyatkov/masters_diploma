import os
import pathlib
import re
import json
import pandas as pd
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SMART_BASE_PATH = Path("SMART")
DEVICE_TYPES = ['HDD', 'SSD', 'NVMe']
MTBF_FILES = {
    'HDD': 'SMART/All_HDD.md',
    'SSD': 'SMART/All_SSD.md',
    'NVMe': 'SMART/All_NVMe.md'
}

def parse_mtbf_table(file_path):
    rows = []
    capture = False
    with open(file_path, 'r') as f:
        for line in f:
            if re.match(r'\|[-\s]*\|', line):  # delimiter
                capture = True
                continue
            if capture and line.startswith('|'):
                parts = [p.strip() for p in line.strip().split('|')[1:-1]]
                if len(parts) >= 7:
                    rows.append({
                        'MFG': parts[0],
                        'Model': parts[1],
                        'Size': parts[2],
                        'Drive_ID': parts[3],
                        'Days': int(parts[4]),
                        'Errors': int(parts[5]),
                        'MTBF': float(parts[6])
                    })
    return pd.DataFrame(rows)

def parse_smart_file(filepath):
    attrs = {}
    with open(filepath, 'r', errors='ignore') as f:
        content = f.read()

    match = re.findall(r'^\s*(\d+)\s+([\w\-_]+)\s+[\w\-]+\s+\d+\s+\d+\s+\d+\s+[-\w]+\s+([\d]+)', content, re.MULTILINE)
    for attr_id, name, raw_value in match:
        if len(name) <= 2:
            continue
        attrs[name] = int(raw_value)
    return attrs

def process_drive(dtype, row):
    drive_id = row['Drive_ID']
    model = row['Model'].replace(' ', '')[:12]
    mfg = row['MFG']
    smart_path_pattern = dtype + "/" + mfg + "/*" + model + "*/" + drive_id
    smart_paths = sorted(pathlib.Path(SMART_BASE_PATH).glob(smart_path_pattern))
    if not smart_paths:
        return None

    smart_values = parse_smart_file(smart_paths[0])
    if not smart_values:
        return None

    return {
        'type': dtype,
        'mfg': mfg,
        'drive_id': drive_id,
        'model': model,
        'days_alive': row['Days'],
        'mtbf': row['MTBF'],
        'errors': row['Errors'],
        'size': row['Size'],
        'smart': smart_values,
        'cost_estimate': float(np.round(np.random.normal(6000, 1000), 2))
    }

def aggregate_data(max_workers=os.cpu_count()):
    all_records = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for dtype in DEVICE_TYPES:
            print(f"Processing {dtype}...")
            mtbf_df = parse_mtbf_table(MTBF_FILES[dtype])

            futures = [executor.submit(process_drive, dtype, row) for _, row in mtbf_df.iterrows()]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dtype} Drives"):
                result = future.result()
                if result:
                    all_records.append(result)

    return all_records

if __name__ == "__main__":
    data = aggregate_data()
    with open("final_dataset.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} entries to final_dataset.json")
