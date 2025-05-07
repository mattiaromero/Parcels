###
# Validation project bechnmarking particles with drifter trajectories using various OGCMs. 
###

import glob 
import os
from pathlib import Path
import sys
from tools.tools import * 

dir = Path("/Users/mattiaromero/Projects/Github/ADVECTOR-Studies/tools") # "C:/Users/toc2/Projects/GitHub/ADVECTOR-Studies/tools"
sys.path.append(str(dir))
from drifter_dispersion import DrifterHandler

def prepare_tracks(drifter_df, w, e, s, n, tstart, tend, output_folder): 
    wesn = (drifter_df.lon > w) & (drifter_df.lon < e) & (drifter_df.lat > s) & (drifter_df.lat < n)
    time = (drifter_df.time >= tstart) & (drifter_df.time <= tend)
    tracks_df = drifter_df[time & wesn].reset_index(drop=True)
    tracks_df["segment_n"] = tracks_df.groupby("segment_id").ngroup()
    # tracks_df["age"] = tracks_df.groupby("segment_n")["time"].transform(lambda x: (x - x.min()).dt.total_seconds() / 86400)
    tracks_df.to_csv(f"{output_folder}/tracks_df.csv")
    return tracks_df

# Define folders
dataset_folder = "/Users/mattiaromero/Data/parcels/test/metocean" # "F:/ADVECTOR/metocean"
output_folder = "/Users/mattiaromero/Data/parcels/test/output/drones" # "F:/PARCELS/test"
drifter_folder = "/Users/mattiaromero/Data/drifters/qc_tdsi_6h_2" # "Y:/PROJECTS/DRIFTERS/data/qc_tdsi_6h_2"

# Load drifter tracks 
drifter_handler = DrifterHandler(drifter_folder)
drifter_df = drifter_handler.prepare()  

# Define filters
w, e, s, n = -160, -125, 20, 40
tstart = '2022-01-01'
tend = '2022-01-31'

# Prepare drifter df 
tracks_df = prepare_tracks(drifter_df, w, e, s, n, tstart, tend, output_folder)
ics_df = tracks_df.groupby("segment_n", group_keys=False).apply(lambda x: x.iloc[::4]) # IC @24h

# Define simulation parameters
T = int((tracks_df.time.max() - tracks_df.time.min()).total_seconds()/3600) # 7*24 
dt = 1
savet = 6 

##  Define models 
# ROMS
roms_mesh_mask = f"{dataset_folder}/grid_npo0.08_07e.nc"
c_grid_dimensions = {
    "lon": "lon_psi",
    "lat": "lat_psi",
    "time": "ocean_time",
}

paths = []

folders = glob.glob(f"{dataset_folder}/amphitrite/*")
for folder in folders: 
    paths.append(glob.glob(f"{folder}/*.nc")[0])

models = [
    {
        "name": "GLOBCURRENT",
        "filenames": {
            "U": f"{dataset_folder}/globcurrent/uo_*.nc",
            "V": f"{dataset_folder}/globcurrent/vo_*.nc"
        },
        "variables": {"U": "uo", "V": "vo"},
        "dimensions": {"time": "time", "lat": "latitude", "lon": "longitude"},
        "indices": {"lon": range(0, 4*60), "lat": range(4*105, 4*135)}
    },
    {
        "name": "GLORYS",
        "filenames": {
            "U": f"{dataset_folder}/nemo/nemo_hourly_operational_u_*.nc",
            "V": f"{dataset_folder}/nemo/nemo_hourly_operational_v_*.nc",
        },
        "variables": {"U": "uo", "V": "vo"},
        "dimensions": {"time": "time", "lon": "longitude", "lat": "latitude"},
        "indices": {"lon": range(0, 12*60), "lat": range(12*95, 12*125)}
    },
    {
        "name": "SMOC",
        "filenames": {
            "U": f"{dataset_folder}/nemo_tot/nemo_hourly_operational_utot_*.nc",
            "V": f"{dataset_folder}/nemo_tot/nemo_hourly_operational_vtot_*.nc",
        },
        "variables": {"U": "utotal", "V": "vtotal"},
        "dimensions": {"time": "time", "lon": "longitude", "lat": "latitude"},
        "indices": {"lon": range(0,12*60), "lat": range(12*95,12*125)}
    },
    {
        "name": "HYCOM", 
        "filenames": {
            "U": f"{dataset_folder}/hycom/hycom_operational_*.nc",
            "V": f"{dataset_folder}/hycom/hycom_operational_*.nc",
        },
        "variables": {"U": "water_u", "V": "water_v"},
        "dimensions": {"time": "time", "lon": "lon", "lat": "lat"},
        "indices": {'lon': range(int(180*12.5), int(240*12.5)), 'lat': range(int(25*95),int(25*125))}, 
    },
    {
        "name": "ROMS", 
        "filenames": {
            "U": {"lon": roms_mesh_mask, "lat": roms_mesh_mask, "data": f"{dataset_folder}/roms_c-grid/test134cars_npo0.08_07e_2022*.nc"},
            "V": {"lon": roms_mesh_mask, "lat": roms_mesh_mask, "data": f"{dataset_folder}/roms_c-grid/test134cars_npo0.08_07e_2022*.nc"}
        },
        "variables": {"U": "u", "V": "v"},
        "dimensions": {
            "U": c_grid_dimensions,
            "V": c_grid_dimensions
        },
        "indices": {"lon": range(0,600), "lat": range(0,375)}
    },
    {
        "name": "AMPHITRITE", 
        "filenames": {
            "U": paths, 
            "V": paths
        },
        "variables": {"U": "u", "V": "v"},
        "dimensions": {"time": "time", "lon": "longitude", "lat": "latitude"},
        "indices": None
    },
]

for model in models:
    run_drift_experiment(
        model["name"],
        model["filenames"],
        model["variables"],
        model["dimensions"],
        model["indices"],
        ics_df, T, dt, savet, output_folder
    )
