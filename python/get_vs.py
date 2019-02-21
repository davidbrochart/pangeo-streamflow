import requests
import json
import pandas as pd
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
import xarray as xr

def get_vs(url):
    response = requests.get(url)
    s = str(response.content)

    j = 'json = '
    i0 = s.find(j) + len(j)
    i1 = s[i0:].find('}') + i0 + 1
    hs = s[i0:i1]

    i1 = s.find('(lon, lat)')
    j = '<td>'
    i0 = s[:i1].rfind(j) + len(j)
    lon, lat = [float(i) for i in s[i0:i1].split(',')]

    h = json.loads(hs)
    hh = [i[0] for i in h['h']]
    he = [i[1] for i in h['h']]
    h['h'] = hh
    h['e'] = he

    df = DataFrame(h).set_index('dates')
    df.index = pd.to_datetime(df.index)
    return df, lat, lon

def locate_vs(vs_file, pix_nb=20, acc_min=1_000_000):
    with open(vs_file) as f:
        virtual_stations = f.read().split()
    da = xr.open_rasterio('../data/hydrosheds/acc.vrt')
    pix_deg = 1 / 1200
    width2 = pix_deg * pix_nb
    station_l, lat_l, lon_l, new_lat_l, new_lon_l = [], [], [], [], []
    for vs in tqdm(virtual_stations):
        url = f'http://hydroweb.theia-land.fr/hydroweb/view/{vs}?lang=en&basin=AMAZONAS'
        df, lat, lon = get_vs(url)
        max_lat, max_lon = np.nan, np.nan
        for pix_nb2 in range(1, pix_nb+1):
            # look further and further away from original lat/lon
            width = pix_nb2 * pix_deg
            acc = da.loc[1, lat+width:lat-width, lon-width:lon+width]
            max_acc = acc.where(acc==acc.max(), drop=True)
            if max_acc.values[0, 0] >= acc_min:
                # consider only rivers with a minimum accumulated flow
                max_lat = max_acc.y.values[0]
                max_lon = max_acc.x.values[0]
                break
        station_l.append(vs)
        lat_l.append(lat)
        lon_l.append(lon)
        new_lat_l.append(max_lat)
        new_lon_l.append(max_lon)
    return DataFrame({'station':station_l, 'lat':lat_l, 'lon':lon_l, 'new_lat':new_lat_l, 'new_lon':new_lon_l})
