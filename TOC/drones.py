###
# Input data for AWS Hackathon. 

# Inputs: 
#   1. pset
#       1. Pnoise pset (scales, thresholds, seed)
#       2. Blob pset (centers, stds)
#   2. Kh 
###

import glob 
import os
import shutil
import sys
from pathlib import Path
import tempfile
from numpy import random
from sklearn.datasets import make_blobs
from noise import pnoise2 

dir = Path("/Users/mattiaromero/Projects/Github/ADVECTOR-Studies/tools") # "C:/Users/toc2/Projects/GitHub/ADVECTOR-Studies/tools"
sys.path.append(str(dir))

from tools.tools import * 

def generate_particles_with_perlin(
    n_particles=1000,
    grid_size=(200, 200),
    scale=50,
    threshold=0.1,
    seed=0,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0
):
    nx, ny = grid_size
    x_coords, y_coords = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    
    # Scale to Perlin space
    x_scaled = x_coords / scale
    y_scaled = y_coords / scale

    # Vectorized Perlin noise using np.vectorize
    noise_func = np.vectorize(lambda x, y: pnoise2(
        x, y,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        repeatx=nx,
        repeaty=ny,
        base=seed
    ))

    noise = noise_func(x_scaled, y_scaled)

    # Normalize to 0–1
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    # Mask and sample particles
    y_idx, x_idx = np.where(noise > threshold)
    if len(x_idx) < n_particles:
        raise ValueError(f"Only {len(x_idx)} points exceed threshold. Lower it or reduce n_particles.")
    
    idx = np.random.choice(len(x_idx), n_particles, replace=False)
    x_norm = x_idx[idx] / nx
    y_norm = y_idx[idx] / ny

    return np.column_stack([x_norm, y_norm])

def crop_box(ds, wesn_box):

    if "z" in ds.data_vars:
        ds = ds.drop_vars("z")

    mask = ((ds.lon >= wesn_box[0]) & (ds.lon <= wesn_box[1]) &
            (ds.lat >= wesn_box[2]) & (ds.lat <= wesn_box[3])).compute()

    ds = ds.where(mask, drop=True)

    return ds
    
##  Define folders
# dataset_folder = "/Users/mattiaromero/Data/metocean/currents" # "F:/ADVECTOR/metocean"
# paths = []
# folders = glob.glob(f"{dataset_folder}/amphitrite/*")
# for folder in folders: 
#     paths.append(folder)
#     # paths.append(glob.glob(f"{folder}/*.nc")[0])

# if len(paths) < 2: 
#     path = paths 
# else: 
#     id = random.randint(0, len(paths)-1)
#     path = paths[id]

path = "/Users/mattiaromero/Data/metocean/currents/amphitrite/20220602.nc"
tmp_folder = "/Users/mattiaromero/Data/parcels/tmp" # "F:/PARCELS/test"
output_folder = "/Users/mattiaromero/Data/parcels/drones2" # "F:/PARCELS/test"

for k in range(0,20):
    print(f"Iteration {k}/20")

    ## Load fieldset 
    fieldset_ds = xr.open_dataset(path)
    # FIXME: depends on n iterables
    wesn = np.zeros((5, 4))
    wesn_box = np.zeros((5, 4))

    ##  Prepare ICs
    # lat, lon = random.randint(20, 40), random.randint(-150, -125)
    # wesn = lon, lon+1, lat, lat+1

    p_time = fieldset_ds.time.values[0]
    p_labels, p_lons, p_lats = [], [], []
    # FIXME: upscaling domain to have particles enter. n is 2*scale due to scale**2 domain size increase
    n = 4*5000 # total number of particles 

    # Pnoise
    # NOTE: iterables 
    scales = np.array([2, 50, 100]) # [2, 10, 25, 50, 100] 
    t = 0.4 # [0.1, 0.3, 0.5]
    seed=random.randint(0, 100)

    for i, s in enumerate(scales):
        # for t in thresholds: 
                
        lat, lon = random.randint(20, 40), random.randint(-150, -125)
        wesn[i, :] = lon, lon+2, lat, lat+2 # wesn = lon, lon+1, lat, lat+1
        wesn_box[i, :] = lon+0.5, lon+1.5, lat+0.5, lat+1.5

        particles = generate_particles_with_perlin(
            n_particles=n,
            grid_size=(300, 300),
            scale=s,
            threshold= t,
            seed=seed,
            octaves=1, # controls level of detail. Higher = more structure.
            persistence=0.3, # amplitude change between octaves. Try 0.3–0.6.
            lacunarity=2.5 # frequency change between octaves. 1.5–2.5 works well.
        )

        p_lon = 2*(particles[:, 0] - 0.5) + (wesn[i, 0] + wesn[i, 1])/2
        p_lat = 2*(particles[:, 1] - 0.5) + (wesn[i, 2] + wesn[i, 3])/2
        p_lons.append(p_lon)
        p_lats.append(p_lat)
        p_labels.append(f"perlin_{s}_{t}") 
        
    # Uniform 
    # lons = np.linspace(wesn[0], wesn[1], int(np.sqrt(n))) 
    # lats = np.linspace(wesn[2], wesn[3], int(np.sqrt(n))) 
    # p_lon, p_lat = np.meshgrid(lons, lats)
    # p_lons.append(p_lon)
    # p_lats.append(p_lat)
    # p_labels.append("uniform")

    # Blobs 
    # d = 50 # #/km2 (based on 70 #/km2 scans)
    # A = 1*111 #km2 
    # n = d * A
    # rseed= None # 123 #Seed of random generator 
    cluster_box = (0.5,-0.5) #Box in which random samples are generated around every centre

    # NOTE: iterables 
    center= 2*50 # np.array([10, 25, 50, 100])  #Number of blob's centers; this defines also how many blobs to generate
    cluster_stds = 2*1 / np.array([50, 100]) # np.array([10, 25, 50, 100]) #Standard deviation of the blob; this is the dispersion radius within gaussian sigma

    # for center in centers:
    for j, std in enumerate(cluster_stds):   

        lat, lon = random.randint(20, 40), random.randint(-150, -125)
        wesn[3+j, :] = lon, lon+2, lat, lat+2
        wesn_box[3+j, :] = lon+0.5, lon+1.5, lat+0.5, lat+1.5
            
        x , _ = make_blobs(n_samples=n, 
                        random_state=None, 
                        centers=center, 
                        cluster_std=std, 
                        center_box=cluster_box)
        p_lon = x[:, 0] + (wesn[3+j, 0] + wesn[3+j, 1])/2
        p_lat = x[:, 1] + (wesn[3+j, 2] + wesn[3+j, 3])/2
        p_lons.append(p_lon)
        p_lats.append(p_lat)
        p_labels.append(f"blob_{center}_{std}")

    ## Advection 
    # Define simulation parameters
    T = timedelta(hours=24)
    dt = timedelta(hours=1) 
    savet = timedelta(hours=1)

    model = {
            "name": "AMPHITRITE", 
            "filenames": {
                "U": path, # paths, 
                "V": path # paths
            },
            "variables": {"U": "u", "V": "v"},
            "dimensions": {"time": "time", "lon": "longitude", "lat": "latitude"},
    }

    fieldset = parcels.FieldSet.from_a_grid_dataset(model["filenames"], 
                                                    model["variables"], 
                                                    model["dimensions"], 
                                                    allow_time_extrapolation = True)


    # NOTE: iterable 
    # Kh = np.array([0.01]) # np.array([0, 0.1, 1]) # m2/s 
    # scale_coef = (1/(111*1e6)) * dt.total_seconds() # deg/min
    # fieldset.add_constant_field("Kh_zonal", 0, mesh="flat")
    # fieldset.add_constant_field("Kh_meridional", 0, mesh="flat")

    for l in range(0,len(p_lons)):
        # for kh in Kh:

        # output_name = f"{k}_{p_labels[l]}_Kh_{kh}_Particles.zarr"
        output_name = f"{k}_{p_labels[l]}_Particles"
        print(f"Computing {output_name} with {n} particles...")

        # FIXME: depends on domain 
        # kh_scale = kh * scale_coef 
        # fieldset.Kh_zonal.data[...] = 0 # kh_scale
        # fieldset.Kh_meridional.data[...] = 0 # kh_scale

        fieldset.add_constant("lonmin", wesn[l, 0])
        fieldset.add_constant("lonmax", wesn[l, 1])
        fieldset.add_constant("latmin", wesn[l, 2])
        fieldset.add_constant("latmax", wesn[l, 3])

        pset = parcels.ParticleSet.from_list(
            fieldset=fieldset,
            pclass=parcels.JITParticle, 
            lon=p_lons[l],
            lat=p_lats[l],
            time=p_time
        )

        output_file = pset.ParticleFile(
            name=f"{tmp_folder}/{output_name}", outputdt=savet
        )

        # kernels = [parcels.AdvectionRK4, parcels.DiffusionUniformKh, DeleteParticle] 
        kernels = [parcels.AdvectionRK4, DeleteParticle] 

        pset.execute(
            kernels, 
            runtime=T,
            dt=dt,
            output_file=output_file
        )

        print("Cropping...")

        ds = xr.open_zarr(f"{tmp_folder}/{output_name}.zarr")
        ds = crop_box(ds, wesn_box[l,:])
        ds.to_zarr(f"{output_folder}/{output_name}.zarr")

        if os.path.exists(f"{tmp_folder}/{output_name}.zarr"):
            shutil.rmtree(f"{tmp_folder}/{output_name}.zarr", ignore_errors=True)

        print("Done!")

print("Ran all simulations successfully!")
