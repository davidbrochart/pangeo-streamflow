import numpy as np
import xarray as xr
from skimage.measure import block_reduce
from shapely.geometry.polygon import Polygon
from shapely.ops import transform
import pyproj
import gcsfs

def adjust_bbox(da, dims):
    coords = {}
    for k, v in dims.items():
        every = v[0]
        step = v[1] #da[k].values[1] - da[k].values[0]
        offset = step / 2
        dim0 = da[k].values[0] - offset
        dim1 = da[k].values[-1] + offset
        if step < 0: # decreasing
            dim0 = dim0 + every - dim0 % every
            dim1 = dim1 - dim1 % every
        else: # increasing
            dim0 = dim0 - dim0 % every
            dim1 = dim1 + every - dim1 % every
        coord0 = np.arange(dim0+offset, da[k].values[0]-offset, step)
        coord1 = da[k].values
        coord2 = np.arange(da[k].values[-1]+step, dim1, step)
        coord = np.hstack((coord0, coord1, coord2))
        coords[k] = coord
    return da.reindex(**coords).fillna(0)

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

def reindex(arrays, dims):
    tolerance = 0.1 * pix_deg_flow
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
        if arrays[0][d].values[1] - arrays[0][d].values[0] > 0: # increasing
            coord[d] = np.arange(vmin[d], vmax[d]+tolerance, pix_deg_flow)
        else:
            coord[d] = np.arange(vmax[d], vmin[d]-tolerance, -pix_deg_flow)
    for i in range(len(arrays)):
        arrays[i] = arrays[i].reindex({d: coord[d] for d in dims}, method='nearest', tolerance=tolerance)

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

def get_trmm_mask(labels, gcs_path):
    pix_deg_flow = 1 / 1200
    pix_deg_trmm = 0.25
    ratio = int(pix_deg_trmm / pix_deg_flow)
    da_mask_trmm = []
    for label in labels:
        ds = xr.open_zarr(gcsfs.GCSMap(f'{gcs_path}/{label}'))
        da1 = ds['mask'].compute()
        da2 = adjust_bbox(da1, {'lat': (pix_deg_trmm, -pix_deg_flow), 'lon': (pix_deg_trmm, pix_deg_flow)})
        da3 = aggregate_da(da2, {'lat': ratio, 'lon': ratio}) / (ratio * ratio)
        da3 = da3.rename({'lat_agg': 'lat', 'lon_agg': 'lon'})
        da3.lon.values = np.round(da3.lon.values, 3)
        da3.lat.values = np.round(da3.lat.values, 3)
        da_mask_trmm.append(da3)
    da_mask_trmm = xr.concat(da_mask_trmm, 'label').assign_coords(label=labels)
    return da_mask_trmm

def get_gpm_mask(labels, gcs_path):
    pix_deg_flow = 1 / 1200
    pix_deg_gpm = 0.1
    ratio = int(pix_deg_gpm / pix_deg_flow)
    da_mask_gpm = []
    for label in labels:
        ds = xr.open_zarr(gcsfs.GCSMap(f'{gcs_path}/{label}'))
        da1 = ds['mask'].compute()
        da2 = adjust_bbox(da1, {'lat': (pix_deg_gpm, -pix_deg_flow), 'lon': (pix_deg_gpm, pix_deg_flow)})
        da3 = aggregate_da(da2, {'lat': ratio, 'lon': ratio}) / (ratio * ratio)
        da3 = da3.rename({'lat_agg': 'lat', 'lon_agg': 'lon'})
        da3.lon.values = np.round(da3.lon.values, 2)
        da3.lat.values = np.round(da3.lat.values, 2)
        da_mask_gpm.append(da3)
    da_mask_gpm = xr.concat(da_mask_gpm, 'label').assign_coords(label=labels)
    return da_mask_gpm

def get_trmm_precipitation(da_mask_trmm):
    ds_trmm = xr.open_zarr(gcsfs.GCSMap('pangeo-data/trmm_3b42rt'))
    # TRMM data was stored with lon in 0/360 range, rearrange it in -180/180:
    long_0_360 = ds_trmm.lon.values
    ds_trmm.lon.values = np.where(long_0_360 < 180, long_0_360, long_0_360 - 360)
    da_trmm = ds_trmm['precipitation'].sortby('lon')
    pix_deg_trmm = 0.25
    da_area_trmm = pixel_area(pix_deg_trmm)
    da_mask_trmm = da_area_trmm.reindex_like(da_mask_trmm, method='nearest', tolerance=0.001) * da_mask_trmm
    da_mask_trmm = da_mask_trmm / da_mask_trmm.sum(['lat', 'lon'])
    p_trmm = (da_trmm.reindex_like(da_mask_trmm, method='nearest', tolerance=0.001) * da_mask_trmm).sum(['lat', 'lon'])
    p_trmm = p_trmm.persist()
    return p_trmm

def get_gpm_precipitation(da_mask_gpm):
    ds_gpm = xr.open_zarr(gcsfs.GCSMap('pangeo-data/gpm_imerg_early'))
    da_gpm = ds_gpm['precipitationCal']
    pix_deg_gpm = 0.1
    da_area_gpm = pixel_area(pix_deg_gpm)
    da_mask_gpm = da_area_gpm.reindex_like(da_mask_gpm, method='nearest', tolerance=0.001) * da_mask_gpm
    da_mask_gpm = da_mask_gpm / da_mask_gpm.sum(['lat', 'lon'])
    p_gpm = (da_gpm.reindex_like(da_mask_gpm, method='nearest', tolerance=0.01) * da_mask_gpm).sum(['lat', 'lon'])
    p_gpm = p_gpm.persist()
    return p_gpm
