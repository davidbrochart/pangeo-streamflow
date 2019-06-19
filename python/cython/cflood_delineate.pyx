import numpy as np
cimport numpy as np
np.import_array()
from libcpp cimport bool
from cpython cimport bool
cimport cython

import xarray as xr

@cython.boundscheck(False)
@cython.wraparound(False)
def cflood_delineate(double lat, double lon, double pix_deg, double tile_deg, double elevation, np.ndarray[np.uint8_t, ndim=2] mm):
    cdef np.ndarray[np.int16_t, ndim=2] dem_tile
    cdef int x, y, mx, my, done, next_size, i, dx, dy
    cdef np.ndarray[np.float64_t, ndim=1] latlon
    cdef np.ndarray[np.uint8_t, ndim=2] flood_mask
    cdef np.ndarray[np.uint16_t, ndim=2] next_xy
    next_xy = np.empty((1024, 2), dtype=np.uint16)

    next_size = 0
    mx = int(mm.shape[0] / 2 - 1)
    my = int(mm.shape[0] / 2 - 1)
    next_xy[0] = [mx, my]

    mm[:] = 0
    mx0_deg = lon - pix_deg * mx
    my0_deg = lat + pix_deg * my
    dem_tile, y, x = get_tile(lat, lon, pix_deg, tile_deg)
    done = 0
    while done == 0:
        # flood this pixel
        mm[my, int(np.floor(mx / 8))] |= 1 << (mx % 8)
        # look around this pixel
        for i in range(8):
            if i == 0:
                dx, dy = 1, 0
            elif i == 1:
                dx, dy = 1, 1
            elif i == 2:
                dx, dy = 0, 1
            elif i == 3:
                dx, dy = -1, 1
            elif i == 4:
                dx, dy = -1, 0
            elif i == 5:
                dx, dy = -1, -1
            elif i == 6:
                dx, dy = 0, -1
            elif i == 7:
                dx, dy = 1, -1
            if (((mm[my+dy, int(np.floor((mx+dx) / 8))] >> ((mx+dx) % 8)) & 1) == 0) and (-32768 < dem_tile[y+dy, x+dx] < elevation):
                # pixel not already processed, and water can flow into it
                next_xy[next_size] = [mx+dx, my+dy]
                next_size += 1
                if next_size == next_xy.shape[0]:
                    next_xy = np.resize(next_xy, (next_xy.shape[0] * 2, 2))
        if next_size == 0:
            # no more pixels to visit, we are done
            done = 1
        else:
            next_size -= 1
            mx_next, my_next = next_xy[next_size]
            dx, dy = mx_next-mx, my_next-my
            x, y = x+dx, y+dy
            lat, lon = lat-dy*pix_deg, lon+dx*pix_deg
            mx, my = mx_next, my_next
            if (x <= 0) or (x >= dem_tile.shape[1] - 1) or (y <= 0) or (y >= dem_tile.shape[0] - 1):
                # reached the border of the tile, re-center
                dem_tile, y, x = get_tile(lat, lon, pix_deg, tile_deg)
    latlon = np.empty(2, dtype=np.float64)
    flood_mask, latlon[0], latlon[1] = get_bbox(mm, pix_deg, mx0_deg, my0_deg)
    return flood_mask, latlon

@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_bbox(np.ndarray[np.uint8_t, ndim=2] mm, double pix_deg, double mx0_deg, double my0_deg):
    cdef int going_down, i, i0, i1, done, j, x0, x1, y0, y1, found_x, found_y
    cdef np.ndarray[np.uint8_t, ndim=2] mask
    going_down = 1
    i = mm.shape[0] >> 1
    i0 = i
    i1 = i - 1
    done = 0
    while done == 0:
        for j in range(mm.shape[1]):
            if mm[i, j] != 0:
                done = 1
        if done == 0:
            if going_down == 1:
                i0 += 1
                i = i1
            else:
                i1 -= 1
                i = i0
            going_down = 1 - going_down
    if i > 0:
        i -= 1
    done = 0
    while done == 0:
        done = 1
        for j in range(mm.shape[1]):
            if mm[i, j] != 0:
                done = 0
        if done == 0:
            i -= 1
            if i < 0:
                done = 1
    i += 1

    x0 = mm.shape[1] * 8
    x1 = -1
    y0 = -1
    y1 = -1
    found_y = 0
    done = 0
    while done == 0:
        found_x = 0
        for j in range(mm.shape[1]):
            if mm[i, j] != 0:
                found_x = 1
                for k in range(8):
                    if (mm[i, j] >> k) & 1 == 1:
                        l = j * 8 + k
                        if x0 > l:
                            x0 = l
                        if x1 < l:
                            x1 = l
        if found_x == 1:
            found_y = 1
            y0 = i
            if y1 < 0:
                y1 = i
        if (found_x == 0) and (found_y == 1):
            done = 1
        else:
            i += 1
            if i == mm.shape[0]:
                done = 1
    y0 += 1
    x1 += 1
    mask = np.empty((y0 - y1, x1 - x0), dtype=np.uint8)
    for i in range(y1, y0):
        for j in range(x0, x1):
            mask[(i - y1), j - x0] = (mm[i, int(np.floor(j / 8))] >> (j % 8)) & 1
    return mask, my0_deg - pix_deg * y1, mx0_deg + pix_deg * x0

def get_tile(lat, lon, pix_deg, tile_deg):
    lat -= pix_deg
    da = xr.open_rasterio(f'../data/hydrosheds/dem.vrt')
    da = da.sel(band=1, y=slice(lat+tile_deg/2, lat-tile_deg/2), x=slice(lon-tile_deg/2, lon+tile_deg/2))
    # FIXME
    # there might be problems for tiles in the corners
    assert da.y.shape[0] == int(round(tile_deg / pix_deg))
    assert da.x.shape[0] == int(round(tile_deg / pix_deg))
    y = da.y.shape[0] // 2 - 1
    x = da.x.shape[0] // 2
    return da.values, y, x
