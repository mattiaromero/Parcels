import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from noise import pnoise2

def PerlinNoise(
    n_particles=1000,
    grid_size=(200, 200),
    scale=50,
    seed=0,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    plot=False
):
    nx, ny = grid_size
    x_coords, y_coords = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    # Scale to Perlin space
    x_scaled = x_coords / scale
    y_scaled = y_coords / scale

    # Vectorized Perlin noise
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

    # Normalize noise to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    # Flatten and normalize to create probability distribution
    flat_noise = noise.flatten()
    prob = flat_noise / flat_noise.sum()

    # Sample indices based on the probability
    flat_indices = np.arange(len(flat_noise))
    sampled_indices = np.random.choice(flat_indices, size=n_particles, replace=True, p=prob)

    x_idx = sampled_indices // ny
    y_idx = sampled_indices % ny

    x_norm = x_idx / nx
    y_norm = y_idx / ny

    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(noise, origin='lower', cmap='bone')
        plt.scatter(y_idx, x_idx, s=1, c='red', alpha=0.5)
        plt.title("Gradient-Based Perlin Distribution")
        plt.axis("off")
        plt.show()

    return np.column_stack([x_norm, y_norm])

def PerlinNoiseThreshold(
    n_particles=1000,
    grid_size=(200, 200),
    scale=50,
    threshold=0.1,
    seed=0,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    plot=False
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

    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(noise, origin='lower', cmap='bone')
        plt.scatter(x_idx[idx], y_idx[idx], s=1, c='red')
        plt.title(f"Perlin Filament Distribution (scale={scale}, threshold={threshold})")
        plt.axis('off')
        plt.show()

    return np.column_stack([x_norm, y_norm])

def DeleteParticle(particle, fieldset, time):
    if (particle.lon < fieldset.lonmin or particle.lon > fieldset.lonmax or
        particle.lat < fieldset.latmin or particle.lat > fieldset.latmax):
        
        particle.delete()  # Remove particle from the simulation if out of domain or drifter has died

def CropBox(ds, wesn_box):

    if "z" in ds.data_vars:
        ds = ds.drop_vars("z")

    mask = ((ds.lon >= wesn_box[0]) & (ds.lon <= wesn_box[1]) &
            (ds.lat >= wesn_box[2]) & (ds.lat <= wesn_box[3])).compute()

    ds = ds.where(mask, drop=True)

    return ds