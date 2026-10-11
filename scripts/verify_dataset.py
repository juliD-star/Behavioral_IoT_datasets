#!/usr/bin/env python3
"""Verify final feature CSV structure and published device-level counts."""
import sys
from pathlib import Path
import pandas as pd
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/windows_grouped_final.csv')
df=pd.read_csv(p)
expected_columns=['device','capture_id','activity','sec','bytes_per_s','pkts_per_s','mean_pkt_size','std_pkt_size','mean_iat']
expected={'Google Doorbell':(783,13),'Google Nest Hub':(729,6),'Google Nest Mini':(1092,9),'Ring Doorbell':(616,10),'Smart Bulb':(267,14),'Smart Plug':(69,12),'Temperature Sensor':(335,7),'Wyze Camera':(813,12)}
assert df.columns.tolist()==expected_columns, f'Unexpected columns: {df.columns.tolist()}'
assert df.shape==(4704,9),f'Unexpected shape {df.shape}'
assert not df.isna().any().any(),'Missing values detected'
observed={device:(len(d),d['capture_id'].nunique()) for device,d in df.groupby('device')}
assert observed==expected,f'Device totals differ: {observed}'
assert sum(c for _,c in observed.values())==83
print('PASS: 4,704 windows, 8 devices, 83 device-specific captures, 9 columns, no missing values')
for device,(windows,captures) in observed.items(): print(f'{device}: {windows} windows, {captures} captures')
