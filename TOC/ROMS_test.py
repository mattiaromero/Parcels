import glob 
import os
from pathlib import Path
import sys
from tools.tools import * 

from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def interp_center_on_grid(fieldset_a): 
     
    # --------- Grid ---------

    dlon = 0.08
    dlat = 0.08
    # Outside corner coordinates - coordinates + 0.5 dx
    x_outcorners, y_outcorners = np.meshgrid(
        np.append(
            (fieldset_a.U.lon - 0.5 * dlon),
            (fieldset_a.U.lon + 0.5 * dlon),
        ),
        np.append(
            (fieldset_c.U.lat - 0.5 * dlat),
            (fieldset_c.U.lat + 0.5 * dlat),
        ),
    )

    # Inside corner coordinates - coordinates + 0.5 dx
    # needed to plot cells between velocity field nodes
    x_incorners, y_incorners = np.meshgrid(
        (fieldset_a.U.lon + 0.5 * dlon)[:-1],
        (fieldset_a.U.lat + 0.5 * dlat)[:-1],
    )

    # Center coordinates
    x_centers, y_centers = np.meshgrid(
        fieldset_a.U.lon, fieldset_a.U.lat
    )

    # --------- Velocity fields ---------

    # Empty cells between coordinate nodes - essentially on inside corners
    cells = np.zeros((len(fieldset_a.U.lat), len(fieldset_a.U.lon)))

    # Interpolate U
    fu = interpolate.RectBivariateSpline(
        fieldset_a.U.lat,
        fieldset_a.U.lon,
        fieldset_a.U.data[0],
        kx=1,
        ky=1,
    )

    # Velocity field interpolated on the inside corners
    u_corners = fu(y_incorners[:, 0], x_incorners[0, :])

    # Interpolate V
    fv = interpolate.RectBivariateSpline(
        fieldset_a.U.lat,
        fieldset_a.U.lon,
        fieldset_a.V.data[0],
        kx=1,
        ky=1,
    )

    v_corners = fv(y_incorners[:, 0], x_incorners[0, :])

    return u_corners, v_corners, x_incorners, y_incorners 

# Define folders
dataset_folder = "F:/ADVECTOR/metocean"
mesh_mask = f"{dataset_folder}/grid_npo0.08_07e.nc"

# Define models 
a_grid_dimensions = { 
    "lon": "lon_rho",
    "lat": "lat_rho",
    "time": "ocean_time",
}

c_grid_dimensions = {
    "lon": "lon_psi",
    "lat": "lat_psi",
    "time": "ocean_time",
}

models = [
    {
        "name": "ROMS_C-GRID", 
        "filenames": {
            "U": {"lon": mesh_mask, "lat": mesh_mask, "data": f"{dataset_folder}/roms_c-grid/test134cars_npo0.08_07e_2020*.nc"},
            "V": {"lon": mesh_mask, "lat": mesh_mask, "data": f"{dataset_folder}/roms_c-grid/test134cars_npo0.08_07e_2020*.nc"}
        },
        "variables": {"U": "u", "V": "v"},
        "dimensions": {
            "U": c_grid_dimensions,
            "V": c_grid_dimensions
        },
        "indices": {"lon": range(0,600), "lat": range(0,375)}
    },
    {
        "name": "ROMS_A-GRID", 
        "filenames": {
            "U": f"{dataset_folder}/roms_a-grid/test134cars_npo0.08_07e_2020*.nc", 
            "V": f"{dataset_folder}/roms_a-grid/test134cars_npo0.08_07e_2020*.nc"
        },
        "variables": {"U": "u_eastward", "V": "v_northward"},
        "dimensions": a_grid_dimensions,
    }  
]

for model in models:
    print(model)
    if model['name'] == "ROMS_C-GRID":
        fieldset_c = parcels.FieldSet.from_mitgcm(model["filenames"], model["variables"], model["dimensions"], model["indices"]) # FIXME: same result with mitgcm 
    elif model['name'] == "ROMS_A-GRID":
        fieldset_a = parcels.FieldSet.from_netcdf(model["filenames"], model["variables"], model["dimensions"])

fieldset_c.computeTimeChunk()
fieldset_a.computeTimeChunk()

U, V, X, Y = interp_center_on_grid(fieldset_a)
diff_U = (fieldset_c.U.data[0]  - U) # / fieldset_c.U.data[0] # fieldset_a.U.data[0, 1:, 1:]
diff_V = (fieldset_c.V.data[0] - V) # / fieldset_c.V.data[0] # fieldset_a.V.data[0, 1:, 1:]

fig, axs = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})
nind = -1 # 30
scale = 10 # 4

quiver_args = {
    "scale_units": "xy",
    "scale": 5,  
}

for ax, diff, title in zip(axs, [diff_U, diff_V], ["ΔU (Eastward Vel.)", "ΔV (Northward Vel.)"]):
    
    # Background difference field
    im = ax.pcolormesh(X[0, :nind], Y[:nind, 0], diff[:nind, :nind], vmin = -0.5, vmax = 0.5, cmap="coolwarm", transform=ccrs.PlateCarree()) # , vmin = -10, vmax = 10

    # Quiver sets
    q1 = ax.quiver(fieldset_c.U.lon[:nind][::scale], fieldset_c.U.lat[:nind][::scale],
                   fieldset_c.U.data[0, :nind, :nind][::scale, ::scale],
                   fieldset_c.V.data[0, :nind, :nind][::scale, ::scale], 
                   color="g", label="C-grid", **quiver_args)

    q2 = ax.quiver(fieldset_a.U.lon[:nind][::scale], fieldset_a.U.lat[:nind][::scale],
                   fieldset_a.U.data[0, :nind, :nind][::scale, ::scale],
                   fieldset_a.V.data[0, :nind, :nind][::scale, ::scale], 
                   color="k", label="A-grid", **quiver_args)

    q3 = ax.quiver(X[0, :nind][::scale], Y[:nind, 0][::scale],
                   U[:nind, :nind][::scale, ::scale], V[:nind, :nind][::scale, ::scale], 
                   color="r", label="A-grid nodes", **quiver_args)

    # Map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_title(title, fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Velocity Difference [m/s]", fontsize=12) #(perc. of total)

    # Custom legend
    handles = [
        mpatches.Patch(color="g", label="C-grid"),
        mpatches.Patch(color="k", label="A-grid"),
        mpatches.Patch(color="r", label="A-grid nodes")
    ]
    ax.legend(handles=handles, loc="upper right")

plt.tight_layout()
plt.savefig(f"{dataset_folder}/ROMS_qc_grid.png")

print("Plot generated successfully!")
print("Check done!")