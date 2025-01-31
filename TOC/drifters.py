import glob 

import math
from datetime import timedelta
from operator import attrgetter

import numpy as np
import pandas as pd
import xarray as xr

import parcels

def WrapLongitudeKernel(particle, fieldset, time):
    if particle.lon > 180:
        particle.dlon = -360
    elif particle.lon < -180:
        particle.dlon = 360

def ConvertParticlesTo360(particle, fieldset, time):
    if particle.lon < 0:
        particle.lon += 360  # Shift to 0-360 range

def ConvertParticlesTo180(particle, fieldset, time):
    if particle.lon > 180:
        particle.lon -= 360  # Shift back to -180,180 range

def DeleteParticle(particle, fieldset, time):
    particle.delete()  # Remove particle from the simulation

class DrifterParticle(parcels.JITParticle):
    age = parcels.Variable("age", dtype=np.float32, initial=0, to_write=True)  
    drifter_id = parcels.Variable("drifter_id", dtype=np.float32, to_write=True) 

def Age(particle, fieldset, time):
    particle.age += particle.dt / 3600 # in hours

def run_parcels_test(model: str, filenames: dict, variables: dict, dimensions: dict, indices: dict, drifter_df: pd.DataFrame, T: int, dt: int, savet: int, ):
    # QC 
    print("QC:")

    if model == "AMPHITRITE":
        model_xr = xr.open_dataset(filenames.get("U")[0]) 
    else: 
        model_xr = xr.open_dataset(glob.glob(filenames.get("U"))[0]) 
 
    for attr in model_xr[variables.get("U")].attrs:
        print(attr, ":", model_xr[variables.get("U")].attrs[attr])

    for attr in model_xr.attrs:
        print(attr, ":", model_xr.attrs[attr])

    if model == "ROMS":
        fieldset = parcels.FieldSet.from_mitgcm(filenames, variables, dimensions, indices, allow_time_extrapolation=True)
    else: 
        fieldset = parcels.FieldSet.from_netcdf(filenames, variables, dimensions, indices, allow_time_extrapolation=True)

    print("")
    print("Dataset size:", len(fieldset.U.grid.lon)*len(fieldset.U.grid.lat)*len(fieldset.U.grid.time)) 
    print("Domain:", fieldset.U.grid.lon.min(), fieldset.U.grid.lon.max(),  fieldset.U.grid.lat.min(), fieldset.U.grid.lat.max())
    print("")

    # Define a new particleclass with Variable 'age' with initial value 0.
    # AgeParticle = parcels.JITParticle.add_variable(parcels.Variable("age", initial=0))

    pset = parcels.ParticleSet.from_list(
        fieldset=fieldset,
        pclass=DrifterParticle,
        lon=drifter_df["lon"].values,
        lat=drifter_df["lat"].values,
        time=drifter_df["time"].values,
        drifter_id=drifter_df["id_nr"].values.astype(np.float32)
    )

    output_file = pset.ParticleFile(
        name=f"{model}Particles.zarr", outputdt=timedelta(hours=savet)
    )

    if model =="HYCOM":
        kernels = [Age, ConvertParticlesTo360, parcels.AdvectionRK4]

        pset.execute(
            kernels, # pset.Kernel(ConvertParticlesTo360) + parcels.AdvectionRK4,
            runtime=timedelta(hours=T),
            dt=timedelta(hours=dt),
            output_file=output_file
        )

        # Convert back before saving output
        #pset.execute(
            # pset.Kernel(ConvertParticlesTo180),
            # runtime=timedelta(seconds=0),  # Just apply conversion
            # dt=timedelta(seconds=0)
        # )

        for particle in pset:
            if particle.lon > 180:
                particle.lon -= 360  # Convert from 0-360 to -180,180
    
    else:
        kernels = [Age, parcels.AdvectionRK4]

        pset.execute(
            kernels, # parcels.AdvectionRK4,
            runtime=timedelta(hours=T),
            dt=timedelta(hours=dt),
            output_file=output_file
        )

    return   

# Load drifter tracks 
drifter_folder = r"Y:/PROJECTS/DRIFTERS/data/qc_tdsi_6h_2"
files = glob.glob(f"{drifter_folder}/*.csv")

drifter_df = pd.DataFrame()
for file in files: 
    df = pd.read_csv(file)  
    drifter_df = pd.concat([drifter_df, df], ignore_index=True)  

drifter_df.rename(columns={"longitude": "lon", "latitude": "lat"}, inplace=True)
drifter_df = drifter_df[drifter_df["deployed"] != False]
drifter_df.drop(columns="deployed", inplace=True)

# Define filters
w, e, s, n = -160, -125, 20, 40
tstart = '2022-01-01'
tend = '2022-01-31'

wesn = (drifter_df.lon > w) & (drifter_df.lon < e) & (drifter_df.lat > s) & (drifter_df.lat < n)
time = (drifter_df.time >= tstart) & (drifter_df.time <= tend)
drifter_df["time"] = pd.to_datetime(drifter_df["time"])
tracks_df = drifter_df[time & wesn].reset_index(drop=True)
tracks_df["id_nr"] = tracks_df.groupby("id").ngroup()

ics_df = tracks_df.groupby("id_nr", group_keys=False).apply(lambda x: x.iloc[::4])

# Define simulation parameters
T = int((tracks_df.time.max() - tracks_df.time.min()).total_seconds()/3600) # 7*24 
dt = 1
savet = 6 

dataset_folder = "F:/ADVECTOR/metocean"


model = "GLOBCURRENT"

filenames = {
    "U": f"{dataset_folder}/globcurrent/uo_*.nc",
    "V": f"{dataset_folder}/globcurrent/vo_*.nc"
}
variables = {
    "U": "uo",
    "V": "vo",
}

dimensions = {"time": "time", "lat": "latitude", "lon": "longitude", "time": "time"}

indices = {'lon': range(0,4*60), 'lat': range(4*105,4*135)}

run_parcels_test(model, filenames, variables, dimensions, indices, ics_df, T, dt, savet)


model = "GLORYS"

filenames = {
    "U": f"{dataset_folder}/nemo/nemo_hourly_operational_u_*.nc",
    "V": f"{dataset_folder}/nemo/nemo_hourly_operational_v_*.nc",
}

variables = {"U": "uo", "V": "vo"}

dimensions = {"time": "time", "lon": "longitude", "lat": "latitude"}

indices = {"lon": range(0,12*60), "lat": range(12*95,12*125)}

run_parcels_test(model, filenames, variables, dimensions, indices, ics_df, T, dt, savet)


model = "SMOC"

filenames = {
    "U": f"{dataset_folder}/nemo_tot/nemo_hourly_operational_utot_*.nc",
    "V": f"{dataset_folder}/nemo_tot/nemo_hourly_operational_vtot_*.nc",
}

variables = {"U": "utotal", "V": "vtotal"}

dimensions = {"time": "time", "lon": "longitude", "lat": "latitude"}

indices = {"lon": range(0,12*60), "lat": range(12*95,12*125)}

run_parcels_test(model, filenames, variables, dimensions, indices, ics_df, T, dt, savet)


model = "HYCOM"

filenames = {"U": f"{dataset_folder}/hycom/hycom_operational_*.nc",
             "V": f"{dataset_folder}/hycom/hycom_operational_*.nc",
            }

variables = {"U": "water_u", "V": "water_v"}

dimensions = {"time": "time", "lon": "lon", "lat": "lat"}

indices = {'lon': range(int(180*12.5), int(240*12.5)), 'lat': range(int(25*95),int(25*125))}

run_parcels_test(model, filenames, variables, dimensions, indices, ics_df, T, dt, savet)


model = "ROMS"

filenames = {"U": f"{dataset_folder}/roms/test134cars_npo0.08_07e_2022*.nc", 
             "V": f"{dataset_folder}/roms/test134cars_npo0.08_07e_2022*.nc"
}

variables = {"U": "u", "V": "v"}

# Note that all variables need the same dimensions in a C-Grid
c_grid_dimensions = {
    "lon": "lon_rho",
    "lat": "lat_rho",
    "time": "ocean_time",
}

dimensions = {
    "U": c_grid_dimensions,
    "V": c_grid_dimensions,
}

indices = {"lon": range(0,600), "lat": range(0,375)}

run_parcels_test(model, filenames, variables, dimensions, indices, ics_df, T, dt, savet)


model = "AMPHITRITE"

paths = []

folders = glob.glob(f"{dataset_folder}/amphitrite/*")
for folder in folders: 
    paths.append(glob.glob(f"{folder}/*.nc")[0])

filenames = {"U": paths, 
             "V": paths
}

variables = {"U": "u", "V": "v"}

dimensions = {"time": "time", "lon": "longitude", "lat": "latitude"}

indices = None

run_parcels_test(model, filenames, variables, dimensions, indices, ics_df, T, dt, savet)