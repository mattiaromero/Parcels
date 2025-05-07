###
# Input data for AWS Hackathon. 

# Inputs: 
#   1. Pnoise pset (scales, thresholds, seed)
#   2. Kh 

# Outputs: 
#   1. pset over time 

###

import os
import shutil
from numpy import random
import numpy as np 
import xarray as xr 
from datetime import timedelta
import parcels 
from tools.toolz import PerlinNoise, PerlinNoiseThreshold, DeleteParticle, CropBox

# pset.remove_particle = lambda p, fieldset, time: p.delete()

## Input 
fieldset_file = "/Users/mattiaromero/Data/metocean/currents/amphitrite/20220602.nc"
tmp_folder = "/Users/mattiaromero/Data/parcels/tmp" 
output_folder = "/Users/mattiaromero/Data/parcels/AWSHackathon" 

os.makedirs(tmp_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

n_sim = 100 # number of simulations
n_particles = 5000 # number of particles per degree 

## Prepare simulation
# Define domain
L = 2 # external domain size in degrees
n_particles *= L**2 # upscaling domain to have particles enter. n is 2*scale due to scale**2 domain size increase

# Define simulation parameters
T = timedelta(hours=24)
dt = timedelta(minutes=5) 
savet = timedelta(minutes=5)

# Load fieldset 
fieldset_ds = xr.open_dataset(fieldset_file)

model = {
        "name": "AMPHITRITE", 
        "filenames": {
            "U": fieldset_file,  
            "V": fieldset_file 
        },
        "variables": {"U": "u", "V": "v"},
        "dimensions": {"time": "time", "lon": "longitude", "lat": "latitude"},
}

fieldset = parcels.FieldSet.from_a_grid_dataset(model["filenames"], 
                                                model["variables"], 
                                                model["dimensions"], 
                                                allow_time_extrapolation = True)

## Iterate 
for k in range(0,n_sim):
    print(f"Iteration {k}/{n_sim}")

    ##  Prepare ICs
    p_time = fieldset_ds.time.values[0]
    
    # Subdomain 
    lat, lon = random.randint(20, 40), random.randint(-150, -125)
    wesn = lon, lon+L, lat, lat+L 
    wesn_box = lon+1/(L**2), lon+3/(L**2), lat+1/(L**2), lat+3/(L**2)

    # Pnoise 
    # particles = PerlinNoise(
    #         n_particles=n_particles,
    #         grid_size=(300, 300),
    #         scale=random.randint(10, 100),
    #         seed=random.randint(0, 1999),
    #         octaves=random.randint(1, 6),
    #         persistence=random.uniform(0.3, 0.7),
    #         lacunarity=random.uniform(1.5, 3.5)
    #     )
    
    particles = PerlinNoiseThreshold(
        n_particles=n_particles,
        grid_size=(300, 300),
        scale=random.randint(10, 100),
        threshold=random.uniform(0.05, 0.5),
        seed=random.randint(0, 1999),
        octaves=random.randint(1, 6),
        persistence=random.uniform(0.3, 0.7),
        lacunarity=random.uniform(1.5, 3.5),
        plot=False
    )

    p_lon = 2*(particles[:, 0] - 0.5) + (wesn[0] + wesn[1])/2
    p_lat = 2*(particles[:, 1] - 0.5) + (wesn[2] + wesn[3])/2

    ## Advection 
    output_name = f"{k}_Particles"

    fieldset.add_constant("lonmin", wesn[0])
    fieldset.add_constant("lonmax", wesn[1])
    fieldset.add_constant("latmin", wesn[2])
    fieldset.add_constant("latmax", wesn[3])

    pset = parcels.ParticleSet.from_list(
        fieldset=fieldset,
        pclass=parcels.JITParticle, 
        lon=p_lon,
        lat=p_lat,
        time=p_time
    )

    output_file = pset.ParticleFile(
        name=f"{tmp_folder}/{output_name}", outputdt=savet
    )

    kernels = [parcels.AdvectionRK4, DeleteParticle] # parcels.DiffusionUniformKh

    pset.execute(
        kernels, 
        runtime=T,
        dt=dt,
        output_file=output_file
    )

    print("Cropping...")

    ds = xr.open_zarr(f"{tmp_folder}/{output_name}.zarr")
    ds = CropBox(ds, wesn_box)

    try: 
        ds.to_zarr(f"{output_folder}/{output_name}.zarr")
    except Exception:
        print(f"{output_folder}/{output_name}.zarr already exists.") 
        continue

    if os.path.exists(f"{tmp_folder}/{output_name}.zarr"):
        shutil.rmtree(f"{tmp_folder}/{output_name}.zarr", ignore_errors=True)

    del ds, pset 

    print("Done!")

print("Ran all simulations successfully!")
