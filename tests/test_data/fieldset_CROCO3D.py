import os

import parcels


def create_fieldset(indices=None):
    example_dataset_folder = parcels.download_example_dataset("CROCOidealized_data")
    file = os.path.join(example_dataset_folder, "CROCO_idealized.nc")

    variables = {"U": "u", "V": "v", "W": "w", "H": "h"}
    dimensions = {
        "U": {"lon": "x_rho", "lat": "y_rho", "depth": "s_w", "time": "time"},
        "V": {"lon": "x_rho", "lat": "y_rho", "depth": "s_w", "time": "time"},
        "W": {"lon": "x_rho", "lat": "y_rho", "depth": "s_w", "time": "time"},
        "H": {"lon": "x_rho", "lat": "y_rho"},
    }
    fieldset = parcels.FieldSet.from_croco(
        file,
        variables,
        dimensions,
        allow_time_extrapolation=True,
        mesh="flat",
        indices=indices,
    )

    return fieldset