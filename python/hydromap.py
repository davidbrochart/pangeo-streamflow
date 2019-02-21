import pickle
import os
from tqdm import tqdm
import xarray as xr
from pydap.cas.urs import setup_session
import rasterio
import rasterio.features
import pyproj
from affine import Affine
import numpy as np
import scipy.ndimage
from rasterio import transform
from rasterio.warp import reproject, Resampling
import PIL
import matplotlib.pyplot as plt
from base64 import b64encode
from io import StringIO, BytesIO
from ipyleaflet import Map, Popup, ImageOverlay, Polygon, Marker
from ipywidgets import ToggleButtons
from IPython.display import display
from delineate import delineate

def to_webmercator(source, affine, bounds):
    with rasterio.Env():
        rows, cols = source.shape
        src_transform = affine
        src_crs = {'init': 'EPSG:4326'}
        dst_crs = {'init': 'EPSG:3857'}
        dst_transform, width, height = rasterio.warp.calculate_default_transform(src_crs, dst_crs, cols, rows, *bounds)
        dst_shape = height, width
        destination = np.zeros(dst_shape)
        reproject(
            source,
            destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
    return destination, dst_transform, dst_shape

def get_img(a_web):
    if (np.all(np.isnan(a_web))):
        a_web[:, :] = 0
        a_norm = a_web
    else:
        a_norm = a_web - np.nanmin(a_web)
        vmax = np.nanmax(a_norm)
        if vmax != 0:
            a_norm = a_norm / vmax
        a_norm = np.where(np.isfinite(a_web), a_norm, 0)
    a_im = PIL.Image.fromarray(np.uint8(plt.cm.viridis(a_norm)*255))
    a_mask = np.where(np.isfinite(a_web), 255, 0)
    mask = PIL.Image.fromarray(np.uint8(a_mask), mode='L')
    im = PIL.Image.new('RGBA', a_norm.shape[::-1], color=None)
    im.paste(a_im, mask=mask)
    f = BytesIO()
    im.save(f, 'png')
    data = b64encode(f.getvalue())
    data = data.decode('ascii')
    imgurl = 'data:image/png;base64,' + data
    return imgurl

def show_acc(label, coord, m, current_io, width, da):
    width2 = width / 2.
    lat, lon = coord
    acc = da.loc[1, lat+width2:lat-width2, lon-width2:lon+width2]
    acc_v = int(acc.sel(y=lat, x=lon, method='nearest').values)

    imgurl = get_img(np.sqrt(acc.values))
    bounds = [
        (acc.y[-1].values - 0.5 / 1200,
        acc.x[0].values - 0.5 / 1200),
        (acc.y[0].values + 0.5 / 1200,
        acc.x[-1].values + 0.5 / 1200)
        ]
    io = ImageOverlay(url=imgurl, bounds=bounds, opacity=0.5)
    if current_io is not None:
        m.remove_layer(current_io)
    m.add_layer(io)
    return io, acc_v

class Flow(object):
    def __init__(self, m, label):
        self.m = m
        self.label = label
        self.width = 0.1
        self.coord = None
        self.io = None
        self.s = None
        self.p = None
        self.show_flow = False
        self.show_menu = False
        self.da = xr.open_rasterio('../data/hydrosheds/acc.vrt')
        self.marker = None
    def show(self, **kwargs):
        if not self.show_menu:
            if kwargs.get('type') == 'mousemove':
                self.coord = kwargs.get('coordinates')
                if self.show_flow:
                    self.io, flow = show_acc(self.label, self.coord, self.m, self.io, self.width, self.da)
                    self.label.value = f'lat/lon = {self.coord}, flow = {flow}'
                else:
                    self.label.value = f'lat/lon = {self.coord}'
                    pass
            elif 'width' in kwargs:
                self.width = kwargs.get('width')
                if self.coord and self.show_flow:
                    self.io, flow = show_acc(self.label, self.coord, self.m, self.io, self.width, self.da)
        if kwargs.get('type') == 'contextmenu':
            self.show_menu = True
            if self.show_flow:
                showHideFlow = 'Hide flow'
            else:
                showHideFlow = 'Show flow'
            if showHideFlow == 'Hide flow':
                self.s = ToggleButtons(options=[showHideFlow, 'Delineate watershed', 'Set marker', 'Close'], value=None)
            else:
                self.s = ToggleButtons(options=[showHideFlow, 'Set marker', 'Close'], value=None)
            self.s.observe(self.get_choice, names='value')
            self.p = Popup(location=self.coord, child=self.s, max_width=160, close_button=False, auto_close=True, close_on_escape_key=False)
            self.m.add_layer(self.p)
    def get_choice(self, x):
        self.show_menu = False
        self.s.close()
        self.m.remove_layer(self.p)
        self.p = None
        choice = x['new']
        if choice == 'Show flow':
            self.show_flow = True
        elif choice == 'Hide flow':
            self.show_flow = False
            self.m.remove_layer(self.io)
            self.io = None
        elif choice == 'Delineate watershed':
            self.show_flow = False
            self.m.remove_layer(self.io)
            self.io = None
            self.label.value = 'Delineating watershed, please wait...'
            delineate(*self.coord)
            self.label.value = 'Watershed delineated'
            ds_mask = xr.open_zarr('tmp/ds_mask/0').compute()
            mask = ds_mask['mask'].values
            polygon = get_polygon(mask, ds_mask.lat.values[0]+0.5/1200, ds_mask.lon.values[0]-0.5/1200)
            self.m.add_layer(polygon)
            self.label.value = 'Watershed displayed'
        elif choice == 'Set marker':
            if self.marker is not None:
                self.m.remove_layer(self.marker)
            self.marker = Marker(location=self.coord)
            self.m.add_layer(self.marker)
        elif choice == 'Close':
            pass

def get_polygon(mask, lat, lon):
    x0 = lon
    x1 = x0 + mask.shape[1] / 1200
    y0 = lat
    y1 = y0 - mask.shape[0] / 1200
    mask2 = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint16)
    mask2[1:-1, 1:-1] = mask
    affine = Affine(1/1200, 0, lon-1/1200, 0, -1/1200, lat+1/1200)
    shapes = list(rasterio.features.shapes(mask2, transform=affine))
    polygons = []
    polygon = polygons
    i = 0
    for shape in shapes:
        if len(shape[0]['coordinates'][0]) > 5:
            if i == 1:
                # more than one polygon
                polygons = [polygons]
            if i >= 1:
                polygons.append([])
                polygon = polygons[-1]
            for coord in shape[0]['coordinates'][0]:
                x, y = coord
                polygon.append((y, x))
            i += 1
    polygon = Polygon(locations=polygons, color='green', fill_color='green')
    return polygon
