from pathlib import Path
import pandas as pd
import numpy as np


def npy_load(npy_name):
    return np.load(npy_name).astype(np.int8)


def save_antenna_data(ant_obj):
    data = pd.DataFrame(
        {
            "frequencyRange": ant_obj.frequencyRange,
            "real_Z": ant_obj.real_Z,
            "imaginary_Z": ant_obj.imaginary_Z,
            "absorption_coef": ant_obj.absorption_coef,
        }
    )
    if ant_obj.lossVal != None:
        csvName = (
            f"output_iter{ant_obj.iter_}_Loss_{ant_obj.lossVal:1.0f}.csv"
        )
    else:
        csvName = f"output_iter_{ant_obj.iter_}.csv"

    output_save_dir = Path(ant_obj.SAVE_DIR, r"simulation_data_csv")
    output_save_dir.mkdir(parents=True, exist_ok=True)

    output_csvPath = Path(output_save_dir, csvName)
    # Save the Frame to a CSV file
    data.to_csv(output_csvPath, index=False)

    return output_save_dir, output_csvPath


def save_ndarray_as_npy(opt_obj, x_ini, iter_=0):
    subDir = Path(opt_obj.SAVE_DIR, r"npy")
    subDir.mkdir(parents=True, exist_ok=True)

    fname = f"bin_matrix_iter{iter_}.npy"
    npyPath = Path(subDir, fname)
    np.save(npyPath, x_ini)

    return npyPath
