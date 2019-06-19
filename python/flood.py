import numpy as np
import sys
import os
import shutil
import xarray as xr
from tqdm import tqdm
import dask.array as da
_ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_ + '/cython')
from cflood_delineate import cflood_delineate, get_tile

def flood_delineate(lat, lon, elevation):
    pix_deg = 1 / 1200
    tile_deg = 5
    lat = (lat // pix_deg) * pix_deg + pix_deg
    lon = (lon // pix_deg) * pix_deg
    dem_tile, y, x = get_tile(lat, lon, pix_deg, tile_deg)
    if dem_tile[y, x] >= elevation:
        return
    # output mask ->
    mxw = 10000 # bytes
    myw = mxw * 8 # bits
    mm = np.empty((myw, mxw), dtype=np.uint8)
    # <- output mask

    flood_mask, latlon = cflood_delineate(lat, lon, pix_deg, tile_deg, elevation, mm)
    clat = np.array([latlon[0] - (i + 0.5) * pix_deg for i in range(flood_mask.shape[0])])
    clon = np.array([latlon[1] + (i + 0.5) * pix_deg for i in range(flood_mask.shape[1])])
    da_mask = xr.DataArray(flood_mask, coords=[clat, clon], dims=['lat', 'lon'])
    return da_mask
