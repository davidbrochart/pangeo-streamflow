import numpy as np
cimport numpy as np
np.import_array()
from libcpp cimport bool
from cpython cimport bool
cimport cython

import xarray as xr

@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_length(double lat, double lon, double olat, double olon, double pix_deg, double tile_deg):
    '''
    Computes the length from lat/lon to olat/olon
    '''
    cdef np.ndarray[np.uint8_t, ndim=2] dir_tile
    cdef np.ndarray[np.float64_t, ndim=2] acc_tile
    cdef double length
    cdef int x, y, done, x_keep, y_keep
    length = 0.
    if (abs(olon - lon) < pix_deg / 4) and (abs(olat - lat) < pix_deg / 4):
        return length
    dir_tile, acc_tile, y, x = get_tile(lat, lon, 2, pix_deg, tile_deg)
    done = 0
    while done == 0:
        x_keep, y_keep = x, y
        _, x, y, _, _, lon, lat = go_get_dir(dir_tile[y, x], dir_tile, x, y, 0, 0, lon, lat, pix_deg)
        if (x == 0) or (x == dir_tile.shape[1] - 1) or (y == 0) or (y == dir_tile.shape[0] - 1):
            # reached the border of the tile, re-center
            dir_tile, acc_tile, y, x = get_tile(lat, lon, 2, pix_deg, tile_deg)
        if x != x_keep and y != y_keep:
            length += np.sqrt(acc_tile[y, x] * 2)
        else:
            length += np.sqrt(acc_tile[y, x])
        if (abs(olon - lon ) < pix_deg / 4) and (abs(olat - lat) < pix_deg / 4):
            done = 1
    return length

@cython.boundscheck(False)
@cython.wraparound(False)
def cdelineate(double lat, double lon, bool getSubBass, int sample_i, np.ndarray[np.float64_t, ndim=2] samples, np.ndarray[np.int32_t, ndim=2] labels, np.ndarray[np.float64_t, ndim=1] areas, np.ndarray[np.float64_t, ndim=1] lengths, double pix_deg, double tile_deg, double accDelta, np.ndarray[np.float64_t, ndim=2] sub_latlon, np.ndarray[np.uint8_t, ndim=2] mm, np.ndarray[np.uint8_t, ndim=2] mm_back, double mx0_deg, double my0_deg, np.ndarray[np.uint8_t, ndim=1] dirNeighbors, np.ndarray[np.float64_t, ndim=1] accNeighbors):
    cdef getSubBass_i
    getSubBass_i = int(getSubBass)
    cdef np.ndarray[np.uint8_t, ndim=2] dir_tile
    cdef np.ndarray[np.float64_t, ndim=2] acc_tile
    cdef int x, y, label_i, new_label, mx, my, neighbors_i, done, skip, reached_upper_ws, append_sample, nb, i, dir_back, dir_next, passed_ws, go_down, this_length, sample_size
    cdef double acc, this_acc, this_accDelta, y_down, x_down
    cdef np.ndarray[np.float64_t, ndim=2] samples_new
    cdef np.ndarray[np.int32_t, ndim=2] labels_new
    cdef np.ndarray[np.float64_t, ndim=1] lengths_new
    cdef np.ndarray[np.int32_t, ndim=1] this_label
    cdef np.ndarray[np.uint8_t, ndim=1] dirNeighbors_new
    cdef np.ndarray[np.float64_t, ndim=1] accNeighbors_new
    cdef np.ndarray[np.float64_t, ndim=1] ws_latlon
    cdef np.ndarray[np.uint8_t, ndim=2] ws_mask

    if getSubBass_i == 1:
        dir_tile, acc_tile, y, x = get_tile(lat, lon, 2, pix_deg, tile_deg)
        acc = acc_tile[y,  x]
        samples[0, :] = [lat, lon]
        areas[0] = acc
        lengths[0] = 0.
        rm_latlon(samples[0][0], samples[0][1], sub_latlon, pix_deg)
        sample_i = 0
        labels[0, :] = [-1, 1, 0] # iprev, size, new_label
        label_i = 0
        new_label = 0
        mx = 0
        my = 0
    else:
        lat, lon = samples[sample_i]
        dir_tile, acc_tile, y, x = get_tile(lat, lon, 2, pix_deg, tile_deg)
        if sample_i == 0:
            mm_back[:] = 0
            mx = int(mm.shape[0] / 2 - 1)
            my = int(mm.shape[0] / 2 - 1)
            mx0_deg = lon - pix_deg * mx
            my0_deg = lat + pix_deg * my
            mm[:] = 0
        else:
            mm_back[:] |= mm[:]
            mx = int(round((lon - mx0_deg) / pix_deg))
            my = int(round((my0_deg - lat) / pix_deg))
    neighbors_i = 0
    dirNeighbors[0] = 255 # 255 is for uninitialized
    accNeighbors[0] = 0.
    done = 0
    skip = 0
    while done == 0:
        reached_upper_ws = 0
        if skip == 0:
            if getSubBass_i == 1:
                this_acc = acc_tile[y, x]
                this_accDelta = acc - this_acc
                append_sample = 0
                if (this_accDelta >= accDelta) and (this_acc >= accDelta):
                    append_sample = 1
                if in_latlon(lat, lon, sub_latlon, pix_deg):
                    rm_latlon(lat, lon, sub_latlon, pix_deg)
                    append_sample = 1
                if append_sample == 1:
                    acc = this_acc
                    sample_i += 1
                    if sample_i == samples.shape[0]:
                        # extend arrays
                        samples_new = np.empty((samples.shape[0] * 2, 2), dtype=np.float64)
                        samples_new[:samples.shape[0], :] = samples
                        samples = samples_new
                        labels_new = np.empty((labels.shape[0] * 2, 3), dtype=np.int32)
                        labels_new[:labels.shape[0], :] = labels
                        labels = labels_new
                        lengths_new = np.empty(lengths.shape[0] * 2, dtype=np.float64)
                        lengths_new[:lengths.shape[0]] = lengths
                        lengths = lengths_new
                        areas_new = np.empty(areas.shape[0] * 2, dtype=np.float64)
                        areas_new[:areas.shape[0]] = areas
                        areas = areas_new
                    samples[sample_i, :] = [lat, lon]
                    labels[sample_i, :] = [label_i, labels[label_i, 1] + 1, new_label]
                    lengths[sample_i] = get_length(lat, lon, samples[0, 0], samples[0, 1], pix_deg, tile_deg)
                    areas[sample_i] = this_acc
                    new_label = 0
                    label_i = sample_i
            else:
                if (mm_back[my, int(np.floor(mx / 8))] >> (mx % 8)) & 1 == 1:
                    # we reached the upper subbasin
                    reached_upper_ws = 1
                else:
                    mm[my, int(np.floor(mx / 8))] |= 1 << (mx % 8)
        nb = dirNeighbors[neighbors_i]
        if (reached_upper_ws == 0) and (nb == 255):
            # find which pixels flow into this pixel
            nb = 0
            for i in range(8):
                if i < 4:
                    dir_back = 1 << (i + 4)
                else:
                    dir_back = 1 << (i - 4)
                dir_next, _, _, _, _, _, _ = go_get_dir(1 << i, dir_tile, x, y, mx, my, lon, lat, pix_deg)
                if dir_next == dir_back:
                    nb = nb | (1 << i)
            dirNeighbors[neighbors_i] = nb
            if getSubBass_i == 1:
                accNeighbors[neighbors_i] = acc
        if (reached_upper_ws == 1) or (nb == 0):
            # no pixel flows into this pixel (this is a source), so we cannot go higher
            if neighbors_i == 0:
                # we are at the outlet and we processed every neighbor pixels, so we are done
                done = 1
            else:
                passed_ws = 0
                go_down = 1
                while go_down == 1:
                    _, x, y, mx, my, lon, lat = go_get_dir(dir_tile[y, x], dir_tile, x, y, mx, my, lon, lat, pix_deg)
                    if (x == 0) or (x == dir_tile.shape[1] - 1) or (y == 0) or (y == dir_tile.shape[0] - 1):
                        # reached the border of the tile, re-center
                        dir_tile, acc_tile, y, x = get_tile(lat, lon, 2, pix_deg, tile_deg)
                    if getSubBass_i == 1:
                        if passed_ws == 1:
                            # we just passed a sub-basin
                            this_label = labels[label_i]
                            new_label = this_label[2] + 1
                            this_length = this_label[1]
                            while labels[label_i, 1] >= this_length:
                                label_i -= 1
                            passed_ws = 0
                        # check if we are at a sub-basin outlet that we already passed
                        y_down, x_down = samples[label_i]
                        if (abs(y_down - lat) < pix_deg / 4) and (abs(x_down - lon) < pix_deg / 4):
                            passed_ws = 1
                    neighbors_i -= 1
                    nb = dirNeighbors[neighbors_i]
                    i = find_first1(nb)
                    nb = nb & (255 - (1 << i))
                    if nb == 0:
                        if neighbors_i == 0:
                            go_down = 0
                            done = 1
                    else:
                        go_down = 0
                        skip = 1
                    dirNeighbors[neighbors_i] = nb
                acc = accNeighbors[neighbors_i]
        else: # go up
            skip = 0
            neighbors_i += 1
            if neighbors_i == dirNeighbors.shape[0]:
                dirNeighbors_new = np.empty(dirNeighbors.shape[0] * 2, dtype=np.uint8)
                dirNeighbors_new[:dirNeighbors.shape[0]] = dirNeighbors
                dirNeighbors = dirNeighbors_new
                accNeighbors_new = np.empty(accNeighbors.shape[0] * 2, dtype=np.float64)
                accNeighbors_new[:accNeighbors.shape[0]] = accNeighbors
                accNeighbors = accNeighbors_new
            dirNeighbors[neighbors_i] = 255
            accNeighbors[neighbors_i] = 0.
            i = find_first1(nb)
            _, x, y, mx, my, lon, lat = go_get_dir(1 << i, dir_tile, x, y, mx, my, lon, lat, pix_deg)
            if (x == 0) or (x == dir_tile.shape[1] - 1) or (y == 0) or (y == dir_tile.shape[0] - 1):
                # reached the border of the tile, re-center
                dir_tile, acc_tile, y, x = get_tile(lat, lon, 2, pix_deg, tile_deg)
        if done == 1:
            ws_latlon = np.empty(2, dtype=np.float64)
            if getSubBass_i == 1:
                sample_size = sample_i + 1
                # we need to reverse the samples (incremental delineation must go downstream)
                samples[:sample_size, :] = samples[sample_size-1::-1, :].copy()
                ws_mask = np.empty((1, 1), dtype=np.uint8)
            else:
                sample_size = 0
                mm[:] &= ~mm_back[:]
                ws_mask, ws_latlon[0], ws_latlon[1] = get_bbox(mm, pix_deg, mx0_deg, my0_deg)
    return samples, labels, areas, lengths, sample_size, mx0_deg, my0_deg, ws_mask, ws_latlon, dirNeighbors, accNeighbors

@cython.boundscheck(False)
@cython.wraparound(False)
cdef in_latlon(double lat, double lon, np.ndarray[np.float64_t, ndim=2] ll_list, double pix_deg):
    cdef int i
    for i in range(ll_list.shape[0]):
        if ll_list[i, 0] > -900:
            if (abs(lat - ll_list[i, 0]) < pix_deg / 4) and (abs(lon - ll_list[i, 1]) < pix_deg / 4):
                return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef rm_latlon(double lat, double lon, np.ndarray[np.float64_t, ndim=2] ll_list, double pix_deg):
    cdef int i
    for i in range(ll_list.shape[0]):
        if ll_list[i, 0] > -900:
            if (abs(lat - ll_list[i, 0]) < pix_deg / 4) and (abs(lon - ll_list[i, 1]) < pix_deg / 4):
                ll_list[i] = [-999, -999]
                return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef go_get_dir(int dire, np.ndarray[np.uint8_t, ndim=2] dir_tile, int x, int y, int mx, int my, double lon, double lat, double pix_deg):
    for i in range(8):
        if (dire >> i) & 1 == 1:
            break
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
    else:
        dx, dy = 1, -1
    return dir_tile[y + dy, x + dx], x + dx, y + dy, mx + dx, my + dy, lon + dx * pix_deg, lat - dy * pix_deg

@cython.boundscheck(False)
@cython.wraparound(False)
cdef find_first1(int x):
    cdef int i
    i = 0
    while (x & 1) == 0:
        x = x >> 1
        i += 1
    return i

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

def get_tile(lat, lon, nb, pix_deg, tile_deg):
    lat -= pix_deg
    types = ['dir', 'acc'][:nb]
    tiles = {}
    for t in types:
        da = xr.open_rasterio(f'../data/hydrosheds/{t}.vrt')
        da = da.sel(band=1, y=slice(lat+tile_deg/2, lat-tile_deg/2), x=slice(lon-tile_deg/2, lon+tile_deg/2))
        # FIXME
        # there might be problems for tiles in the corners
        assert da.y.shape[0] == int(round(tile_deg / pix_deg))
        assert da.x.shape[0] == int(round(tile_deg / pix_deg))
        tiles[t] = da.values
    y = da.y.shape[0] // 2 - 1
    x = da.x.shape[0] // 2
    return tiles.get('dir'), tiles.get('acc'), y, x
