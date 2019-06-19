import numpy as np
import sys
import os
import shutil
import xarray as xr
from tqdm import tqdm
import dask.array as da
_ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_ + '/cython')
from cdelineate import cdelineate, get_tile

def delineate(lat, lon, sub_latlon=[], sub_nb=None, acc_delta=np.inf, progress=False):
    pix_deg = 1 / 1200
    tile_deg = 5
    lat = (lat // pix_deg) * pix_deg + pix_deg
    lon = (lon // pix_deg) * pix_deg
    for ll in sub_latlon:
        ll[0] = (ll[0] // pix_deg) * pix_deg + pix_deg
        ll[1] = (ll[1] // pix_deg) * pix_deg
    if sub_nb is not None:
        dir_tile, acc_tile, y, x = get_tile(lat, lon, 2, pix_deg, tile_deg)
        acc_delta = acc_tile[y,  x] / (sub_nb + 1)
    getSubBass = True
    sample_i = 0
    samples = np.empty((1024, 2), dtype=np.float64)
    lengths = np.empty(1024, dtype=np.float64)
    areas = np.empty(1024, dtype=np.float64)
    labels = np.empty((1024, 3), dtype=np.int32)
    dirNeighbors = np.empty(1024, dtype=np.uint8)
    accNeighbors = np.empty(1024, dtype=np.float64)
    ws_latlon = np.empty(2, dtype=np.float64)
    # output mask ->
    mxw = 10000 # bytes
    myw = mxw * 8 # bits
    mm = np.empty((myw, mxw), dtype=np.uint8)
    mm_back = np.empty((myw, mxw), dtype=np.uint8)
    mx0_deg = 0
    my0_deg = 0
    # <- output mask

    simple_delineation = False
    if len(sub_latlon) == 0:
        _sub_latlon = np.empty((1, 2), dtype=np.float64)
        _sub_latlon[0, :] = [lat, lon]
        if not np.isfinite(acc_delta):
            simple_delineation = True
    else:
        _sub_latlon = np.empty((len(sub_latlon), 2), dtype=np.float64)
        _sub_latlon[:, :] = sub_latlon
    tile_size = int(round(tile_deg / pix_deg))
    if simple_delineation:
        sample_size = 1
        samples[0] = [lat, lon]
    else:
        if progress:
            print('Getting bassin partition...')
        samples, labels, areas, lengths, sample_size, mx0_deg, my0_deg, ws_mask, ws_latlon, dirNeighbors, accNeighbors = cdelineate(lat, lon, getSubBass, sample_i, samples, labels, areas, lengths, pix_deg, tile_deg, acc_delta, _sub_latlon, mm, mm_back, mx0_deg, my0_deg, dirNeighbors, accNeighbors)
        if not is_empty_latlon(_sub_latlon):
            print("WARNING: not all subbasins have been processed. This means that they don't fall into different pixels, or that they are not located in the basin. Please check their lat/lon coordinates.")
            for i in range(_sub_latlon.shape[0]):
                if _sub_latlon[i, 0] > -900:
                    print(_sub_latlon[i, 0], _sub_latlon[i, 1])
    #print('Delineating sub-bassins...')
    mask, latlon = [], []
    getSubBass = False
    lat_min = np.inf
    lat_max = -np.inf
    lon_min = np.inf
    lon_max = -np.inf
    new_labels = []
    shutil.rmtree('tmp/ws', ignore_errors=True)
    shutil.rmtree('tmp/ds_mask', ignore_errors=True)
    this_range = range(sample_size)
    if progress:
        this_range = tqdm(this_range)
    for sample_i in this_range:
        _, _, _, _, _, mx0_deg, my0_deg, ws_mask, ws_latlon, dirNeighbors, accNeighbors = cdelineate(lat, lon, getSubBass, sample_i, samples, labels, areas, lengths, pix_deg, tile_deg, acc_delta, _sub_latlon, mm, mm_back, mx0_deg, my0_deg, dirNeighbors, accNeighbors)
        clat = np.array([ws_latlon[0] - (i + 0.5) * pix_deg for i in range(ws_mask.shape[0])])
        clon = np.array([ws_latlon[1] + (i + 0.5) * pix_deg for i in range(ws_mask.shape[1])])
        da_mask = xr.DataArray(ws_mask, coords=[clat, clon], dims=['lat', 'lon'])
        if sample_i == 0:
            new_labels.append('0')
        else:
            i = labels[sample_i][0]
            new_labels.append(new_labels[i] + ',' + str(labels[sample_i][2]))
        ds_mask = da_mask.to_dataset(name='mask')
        ds_mask.to_zarr(store=f'tmp/ws/{sample_i}', mode='w')
        #lat_min = min(lat_min, clat[-1])
        #lat_max = max(lat_max, clat[0])
        #lon_min = min(lon_min, clon[0])
        #lon_max = max(lon_max, clon[-1])
    #vmin = {'lat': lat_min, 'lon': lon_min}
    #vmax = {'lat': lat_max, 'lon': lon_max}
    #new_lat = np.arange(lat_max, lat_min-tolerance, -pix_deg)
    #new_lon = np.arange(lon_min, lon_max+tolerance, pix_deg)
    #for sample_i in range(sample_size):
    #    label = new_labels[sample_size-1-sample_i]
    #    ds_mask = xr.open_zarr(f'tmp/ws/{sample_i}').compute()
    #    da_mask = ds_mask[str(sample_i)]
    #    ilat = (np.abs(new_lat - da_mask.lat.values[0])).argmin()
    #    ilon = (np.abs(new_lon - da_mask.lon.values[0])).argmin()
    #    nlat = 1 + int(round((lat_max - lat_min) / pix_deg))
    #    nlon = 1 + int(round((lon_max - lon_min) / pix_deg))
    #    mask = da.zeros((nlat, nlon), chunks=(1000,1000), dtype='uint8')
    #    #mask = zarr.zeros((nlat, nlon), chunks=(nlat, nlon), dtype='uint8')
    #    da_mask2 = xr.DataArray(mask, coords=[new_lat, new_lon], dims=['lat', 'lon'])
    #    da_mask2[ilat:ilat+da_mask.shape[0], ilon:ilon+da_mask.shape[1]] = da_mask.values
    #    if sample_i == 0:
    #        ds_mask = da_mask2.to_dataset(name=label)
    #    else:
    #        ds_mask[label] = da_mask2
    #ds_mask.to_zarr(store=f'tmp/ds_ws', mode='w')
    #shutil.rmtree('tmp/ds_ws', ignore_errors=True)
    for sample_i in range(sample_size):
        label = new_labels[sample_size-1-sample_i]
        ds_mask = xr.open_zarr(f'tmp/ws/{sample_i}').compute()
        da_mask = ds_mask['mask']
        olatlon = [samples[sample_i][0]-pix_deg/2, samples[sample_i][1]+pix_deg/2]
        da_mask.attrs = {'outlet': olatlon, 'area': areas[sample_size-1-sample_i], 'length': lengths[sample_size-1-sample_i], 'label': label}
        ds_mask = da_mask.to_dataset()
        ds_mask.to_zarr(store=f'tmp/ds_mask/{label}', mode='w')
        shutil.rmtree(f'tmp/ws/{sample_i}', ignore_errors=True)
        #if sample_i == 0:
        #    shutil.copytree(f'tmp/ws/{sample_i}', 'tmp/ds_ws')
        #    os.rename(f'tmp/ds_ws/{sample_i}', f'tmp/ds_ws/{label}')
        #else:
        #    shutil.copytree(f'tmp/ws/{sample_i}/{sample_i}', f'tmp/ds_ws/{label}')
    #shutil.rmtree('tmp/ws', ignore_errors=True)
    #ds_mask = xr.open_zarr('tmp/ds_ws')
    ##da_mask = xr.concat([ds_mask[label] for label in new_labels], 'label').assign_coords(label=new_labels)
    ##ds_mask = da_mask.to_dataset(name='mask')
    ##ds_mask.to_zarr(store='ds_mask', mode='w')
    #return ds_mask

def is_empty_latlon(ll_list):
    for i in range(ll_list.shape[0]):
        if ll_list[i, 0] > -900:
            return False
    return True

def reindex(arrays, dim_deg, vmin=None, vmax=None):
    '''
    Reindex on a regular grid and join
    '''
    dims = list(dim_deg)
    pix_deg = abs(dim_deg[dims[0]])
    tolerance = 0.1 * pix_deg # 10% tolerance
    if (vmin is None) and (vmax is None):
        vmin, vmax = {}, {}
        for d in dims:
            vmin[d] = np.inf
            vmax[d] = -np.inf
        for da in arrays:
            this_vmin, this_vmax = {}, {}
            for d in dims:
                vmin[d] = min(vmin[d], np.min(da[d]).values)
                vmax[d] = max(vmax[d], np.max(da[d]).values)
    coord = {}
    for d in dims:
        if dim_deg[d] > 0: # increasing
            coord[d] = np.arange(vmin[d], vmax[d]+tolerance, pix_deg)
        else:
            coord[d] = np.arange(vmax[d], vmin[d]-tolerance, -pix_deg)
    for i in range(len(arrays)):
        arrays[i] = arrays[i].reindex({d: coord[d] for d in dims}, method='nearest', tolerance=tolerance)
