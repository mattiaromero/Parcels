import glob 
from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import parcels

def ConvertParticlesTo360(particle, fieldset, time):
    if particle.lon < 0:
        particle.lon += 360  # Shift to 0-360 range

def ConvertParticlesTo180(particle, fieldset, time):
    if particle.lon > 180:
        particle.lon -= 360  # Shift back to -180,180 range
     
class DrifterParticle(parcels.JITParticle):
    age = parcels.Variable("age", dtype=np.float32, initial=0, to_write=True)  
    segment_id = parcels.Variable("segment_id", dtype=np.float32, to_write=True) 

def Age(particle, fieldset, time):
    particle.age += particle.dt # in seconds  

def run_drift_experiment(model: str, filenames: dict, variables: dict, dimensions: dict, indices: dict, drifter_df: pd.DataFrame, T: int, dt: int, savet: int, out_folder: str):
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
        fieldset = parcels.FieldSet.from_mitgcm(filenames, variables, dimensions, indices)
    else: 
        fieldset = parcels.FieldSet.from_netcdf(filenames, variables, dimensions, indices)

    print("")
    print("Dataset size:", len(fieldset.U.grid.lon)*len(fieldset.U.grid.lat)*len(fieldset.U.grid.time)) 
    print("Domain:", fieldset.U.grid.lon.min(), fieldset.U.grid.lon.max(),  fieldset.U.grid.lat.min(), fieldset.U.grid.lat.max())
    print("")

    # fieldset.add_constant_field("Kh_zonal", 1, mesh="flat")

    pset = parcels.ParticleSet.from_list(
        fieldset=fieldset,
        pclass=DrifterParticle,
        lon=drifter_df["lon"].values,
        lat=drifter_df["lat"].values,
        time=drifter_df["time"].values,
        segment_id=drifter_df["segment_n"].values.astype(np.float32),
        age = np.zeros_like(drifter_df["lon"].values)
    )

    output_file = pset.ParticleFile(
        name=f"{out_folder}/{model}Particles.zarr", outputdt=timedelta(hours=savet)
    )

    if model =="HYCOM":
        kernels = [Age, ConvertParticlesTo360, parcels.AdvectionRK4, ConvertParticlesTo180] # DiffusionUniformKh #AdvectionDiffusionM1 

        pset.execute(
            kernels, 
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

