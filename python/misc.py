import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import block_reduce
from shapely.geometry.polygon import Polygon
from shapely.ops import transform
import pyproj
import gcsfs
from datetime import datetime, timedelta
from tqdm import tqdm
import os

def gcs_get_dir(src, dst, fs):
    os.mkdir(dst)
    if not src.endswith('/'):
        src += '/'
    ls = fs.ls(src)
    for rname in ls:
        if rname.endswith('/'):
            lname = os.path.basename(rname[:-1])
            gcs_get_dir(rname, f'{dst}/{lname}', fs)
        else:
            lname = os.path.basename(rname)
            fs.get(rname, f'{dst}/{lname}')

def adjust_bbox(da, dims):
    """Adjust the bounding box of a DaskArray to a coarser resolution.

    Args:
        da: the DaskArray to adjust.
        dims: a dictionary where keys are the name of the dimensions on which
              to adjust, and the values are of the form
              (unsigned_coarse_resolution, signed_original_resolution)
    Returns:
        The DataArray bounding box adjusted to the coarser resolution.
    """
    coords = {}
    for k, v in dims.items():
        every, step = v
        offset = step / 2
        dim0 = da[k].values[0] - offset
        dim1 = da[k].values[-1] + offset
        if step < 0: # decreasing coordinate
            dim0 = dim0 + (every - dim0 % every) % every
            dim1 = dim1 - dim1 % every
        else: # increasing coordinate
            dim0 = dim0 - dim0 % every
            dim1 = dim1 + (every - dim1 % every) % every
        coord0 = np.arange(dim0+offset, da[k].values[0]-offset, step)
        coord1 = da[k].values
        coord2 = np.arange(da[k].values[-1]+step, dim1, step)
        coord = np.hstack((coord0, coord1, coord2))
        coords[k] = coord
    #return da.reindex(**coords).fillna(0)
    # the following code breaks the generality of the preceding code
    a = np.zeros((coords['lat'].shape[0], coords['lon'].shape[0]), dtype=np.uint8)
    pix_deg = 1 / 1200
    dlat = int(round((coords['lat'][0] - da.lat.values[0]) / pix_deg))
    dlon = int(round((da.lon.values[0] - coords['lon'][0]) / pix_deg))
    a[dlat:dlat+da.shape[0], dlon:dlon+da.shape[1]] = da.values
    da_reindex = xr.DataArray(a, coords=[coords['lat'], coords['lon']], dims=['lat', 'lon'])
    return da_reindex

# This code comes from https://github.com/pydata/xarray/issues/2525
# It should be replaced by coarsen (https://github.com/pydata/xarray/pull/2612)
def aggregate_da(da, agg_dims, suf='_agg'):
    input_core_dims = list(agg_dims)
    n_agg = len(input_core_dims)
    core_block_size = tuple([agg_dims[k] for k in input_core_dims])
    block_size = (da.ndim - n_agg)*(1,) + core_block_size
    output_core_dims = [dim + suf for dim in input_core_dims]
    output_sizes = {(dim + suf): da.shape[da.get_axis_num(dim)]//agg_dims[dim] for dim in input_core_dims}
    output_dtypes = da.dtype
    da_out = xr.apply_ufunc(block_reduce, da, kwargs={'block_size': block_size},
                            input_core_dims=[input_core_dims],
                            output_core_dims=[output_core_dims],
                            output_sizes=output_sizes,
                            output_dtypes=[output_dtypes],
                            dask='parallelized')
    for dim in input_core_dims:
        new_coord = block_reduce(da[dim].data, (agg_dims[dim],), func=np.mean)
        da_out.coords[dim + suf] = (dim + suf, new_coord)
    return da_out

def concat(arrays, array_dims, concat_dim, pix_deg):
    tolerance = 0.1 * pix_deg
    vmin, vmax = {}, {}
    for d in array_dims:
        vmin[d] = np.inf
        vmax[d] = -np.inf
    for da in arrays:
        this_vmin, this_vmax = {}, {}
        for d in array_dims:
            vmin[d] = min(vmin[d], np.min(da[d]).values)
            vmax[d] = max(vmax[d], np.max(da[d]).values)
    coord = {}
    for d in array_dims:
        if arrays[0][d].values[1] - arrays[0][d].values[0] > 0: # increasing
            coord[d] = np.arange(vmin[d], vmax[d]+tolerance, pix_deg)
        else:
            coord[d] = np.arange(vmax[d], vmin[d]-tolerance, -pix_deg)
    for i, da in enumerate(arrays):
        da_reindex = da.reindex({d: coord[d] for d in array_dims}, method='nearest', tolerance=tolerance)
        if i == 0:
            da_concat = da_reindex
        else:
            da_concat = xr.concat(da_concat, da_reindex, dim)
        del da_reindex
    return da_concat

# pixel area only depends on latitude (not longitude)
# we re-project WGS84 to cylindrical equal area
def pixel_area(pix_deg):
    project = lambda x, y: pyproj.transform(pyproj.Proj(init='epsg:4326'), pyproj.Proj(proj='cea'), x, y)
    offset = pix_deg / 2
    lts = np.arange(90-offset, -90, -pix_deg)
    area = np.empty_like(lts)
    lon = 0
    for y, lat in enumerate(lts):
        pixel1 = Polygon([(lon - offset, lat + offset), (lon + offset, lat + offset), (lon + offset, lat - offset), (lon - offset, lat - offset)])
        pixel2 = transform(project, pixel1)
        area[y] = pixel2.area
    return xr.DataArray(area, coords=[lts], dims=['lat'])

def get_label_tree(labels):
    prev_sources = []
    label_tree = {}
    while len(labels) > 0:
        sources = source_label(labels)
        ups = []
        for l in sources:
            up = startswith_label(l, prev_sources)
            ups += up
            for d in up:
                label_tree[d]['down'] = l
            if l not in label_tree:
                label_tree[l] = {}
            label_tree[l]['up'] = up
        labels = subtract_label(sources, labels)
        prev_sources = [l for l in prev_sources if l not in ups] + sources
    return label_tree

def source_label(lbls):
    lbl_list = []
    for lbl0 in lbls:
        replaced = False
        ignore = False
        for i1, lbl1 in enumerate(lbl_list):
            if lbl0.startswith(lbl1): # lbl0 is lbl1's source
                lbl_list[i1] = lbl0
                replaced = True
                break
            if lbl1.startswith(lbl0): # lbl1 is lbl0's source
                ignore = True
                break
        if (not replaced) and (not ignore):
            lbl_list.append(lbl0)
    return lbl_list

def startswith_label(lbl, lbls):
    return [l for l in lbls if l.startswith(lbl)]

def subtract_label(lbls, from_lbls):
    return [l for l in from_lbls if l not in lbls]

def get_path(path, gcs=None):
    prefix = 'gs://'
    if path.startswith(prefix):
        return gcsfs.GCSMap(path[len(prefix):], gcs)
    else:
        return path

def get_mask(mask_path, labels, gcs=None):
    pix_deg_flow = 1 / 1200
    das = []
    lat0, lat1, lon0, lon1 = -np.inf, np.inf, np.inf, -np.inf
    for label in tqdm(labels):
        ds = xr.open_zarr(get_path(f'{mask_path}/{label}', gcs), auto_chunk=False)
        da = ds['mask'].compute()
        das.append(da)
        lat0 = max(lat0, da.lat.values[0])
        lat1 = min(lat1, da.lat.values[-1])
        lon0 = min(lon0, da.lon.values[0])
        lon1 = max(lon1, da.lon.values[-1])
    nlat = int(round((lat0 - lat1) / pix_deg_flow + 1))
    nlon = int(round((lon1 - lon0) / pix_deg_flow + 1))
    tolerance = pix_deg_flow / 10
    lat = np.arange(lat0, lat1-tolerance, -pix_deg_flow)
    lon = np.arange(lon0, lon1+tolerance, pix_deg_flow)
    a = np.zeros((nlat, nlon), dtype=np.uint8)
    for da in das:
        dlat = int(round((lat0 - da.lat.values[0]) / pix_deg_flow))
        dlon = int(round((da.lon.values[0] - lon0) / pix_deg_flow))
        a[dlat:dlat+da.shape[0], dlon:dlon+da.shape[1]] += da.values
    da = xr.DataArray(a, coords=[lat, lon], dims=['lat', 'lon'])
    return da

def get_coarser_mask(mask, pix_deg, gcs=None):
    if type(mask) is str:
        ds = xr.open_zarr(get_path(mask, gcs), auto_chunk=False)
    else:
        ds = mask
    pix_deg_flow = 1 / 1200
    ratio = int(round(pix_deg / pix_deg_flow))
    da1 = ds['mask'].compute()
    da2 = adjust_bbox(da1, {'lat': (pix_deg, -pix_deg_flow), 'lon': (pix_deg, pix_deg_flow)})
    da3 = aggregate_da(da2, {'lat': ratio, 'lon': ratio}) / (ratio * ratio)
    da3 = da3.rename({'lat_agg': 'lat', 'lon_agg': 'lon'})
    da3.lon.values = np.round(da3.lon.values / pix_deg, 1) * pix_deg
    da3.lat.values = np.round(da3.lat.values / pix_deg, 1) * pix_deg
    return da3

def get_trmm_mask(mask, gcs=None):
    return get_coarser_mask(mask, 0.25, gcs)

def get_gpm_mask(mask, gcs):
    return get_coarser_mask(mask, 0.1, gcs)

def get_pet_mask(mask, gcs):
    return get_coarser_mask(mask, 1/120, gcs)

def get_watershed_p(ds_p, da_mask, da_area, tolerance):
    da_mask = da_area.reindex_like(da_mask, method='nearest', tolerance=tolerance) * da_mask
    da_mask = da_mask / da_mask.sum(['lat', 'lon'])
    # clip(0) because GPM has negative values!
    p = (ds_p.reindex_like(da_mask, method='nearest', tolerance=tolerance).clip(0) * da_mask).sum(['lat', 'lon'])
    return p

def get_ws_p(pix_deg, da_mask, da_p, tolerance=None):
    if tolerance is None:
        s = str(pix_deg)
        tolerance = 10 ** (-(len(s) - s.find('.')))
    da_area = pixel_area(pix_deg)
    da_mask = da_area.reindex_like(da_mask, method='nearest', tolerance=tolerance) * da_mask
    da_mask = da_mask / da_mask.sum(['lat', 'lon'])
    p = (da_p.reindex_like(da_mask, method='nearest', tolerance=tolerance) * da_mask).sum(['lat', 'lon']).persist()
    return p.to_series()

def get_trmm_precipitation(from_time, to_time, da_mask, ds_p=None, gcs=None):
    if ds_p is None:
        ds_p = xr.open_zarr(gcsfs.GCSMap('pangeo-data/trmm_3b42rt', gcs))
    ds_p = ds_p.sel(time=slice(from_time, to_time))
    # TRMM data was stored with lon in 0/360 range, rearrange it in -180/180:
    long_0_360 = ds_p.lon.values
    ds_p.lon.values = np.where(long_0_360 < 180, long_0_360, long_0_360 - 360)
    ds_p = ds_p.sortby('lon')
    da_p = ds_p['precipitation']
    return get_ws_p(0.25, da_mask, da_p)

def get_gpm_precipitation(from_time, to_time, da_mask, ds_p=None, gcs=None):
    if ds_p is None:
        ds_p = xr.open_zarr(gcsfs.GCSMap('pangeo-data/gpm_imerg_early', gcs))
    ds_p = ds_p.sel(time=slice(from_time, to_time))
    da_p = ds_p['precipitationCal']
    return get_ws_p(0.1, da_mask, da_p)

def str2datetime(s):
    if type(s) is str:
        try:
            st = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            st = datetime.strptime(s, '%Y-%m-%d')
        return st
    else:
        return s

def get_precipitation(from_time, to_time, mask, ds_trmm=None, ds_gpm=None, gcs=None):
    from_time = str2datetime(from_time)
    to_time = str2datetime(to_time)
    trmm_start_time = datetime(2000, 3, 1, 12)
    gpm_start_time = datetime(2014, 3, 12)
    from_time_gpm = from_time
    p_trmm = None
    p_gpm = None
    if from_time < gpm_start_time:
        # before GPM, take TRMM
        if to_time > gpm_start_time:
            to_time_trmm = gpm_start_time
            from_time_gpm = gpm_start_time
        else:
            to_time_trmm = to_time
        print('Getting TRMM precipitation from ' + str(from_time) + ' to ' + str(to_time_trmm))
        da_trmm_mask = get_trmm_mask(mask, gcs)
        p_trmm = get_trmm_precipitation(from_time, to_time_trmm, da_trmm_mask, ds_trmm, gcs)
        # TRMM is 3-hourly, resample to 30min and interpolate with -15min to be like GPM
        p_trmm = p_trmm.resample('30min').asfreq()
        not_nan = np.isfinite(p_trmm.values)
        idx = np.arange(len(p_trmm))
        p_trmm[:] = np.interp(idx-0.5, idx[not_nan], p_trmm.values[not_nan]) # -0.5 translates to -30min/2 = -15min
        p_trmm.index = p_trmm.index - timedelta(minutes=15)
        p_trmm = p_trmm[1:]
    if to_time > gpm_start_time:
        # take GPM
        print('Getting GPM precipitation from ' + str(from_time_gpm) + ' to ' + str(to_time))
        da_gpm_mask = get_gpm_mask(mask, gcs)
        p_gpm = get_gpm_precipitation(from_time, to_time, da_gpm_mask, ds_gpm, gcs)
    if (p_trmm is not None) and (p_gpm is not None):
        print('Concatenating TRMM and GPM precipitations')
        precipitation = pd.concat([p_trmm, p_gpm])
    elif p_trmm is not None:
        precipitation = p_trmm
    else:
        precipitation = p_gpm
    return precipitation

def get_pet(from_time, to_time, mask, ds_pet=None, gcs=None):
    from_time = str2datetime(from_time)
    to_time = str2datetime(to_time)
    da_pet_mask = get_pet_mask(mask, gcs)
    if ds_pet is None:
        ds_pet = xr.open_zarr(gcsfs.GCSMap('pangeo-data/cgiar_pet', gcs))
    da_pet = ds_pet['PET']
    pet = get_ws_p(1/120, da_pet_mask, da_pet, tolerance=0.000001)
    date_range = pd.date_range(start=from_time+timedelta(minutes=15), end=to_time, freq='30min')
    pet_over_time = pd.Series(index=date_range)
    for month in range(1, 13):
        pet_over_time.loc[date_range.month==month] = pet.loc[month] / 30 / 24
    return pet_over_time

def get_peq_from_df(df, suffix):
    '''Get p, e and q time series from a DataFrame where these might be present with a suffix.
    Returns a DataFrame with p, e and q if they are found.
    '''
    df_peq = DataFrame()
    df_peq.index = df.index
    for prefix in ['p', 'e', 'q']:
        name = prefix + suffix
        if name in df.columns:
            df_peq[prefix] = df[name]
    return df_peq
