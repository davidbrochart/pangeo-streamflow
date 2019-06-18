import pickle
import os
from tqdm import tqdm
import xarray as xr
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
from traitlets import Int
from ipywidgets import ToggleButtons
from IPython.display import display
from delineate import delineate

class CustomPopup(Popup):
    max_width = Int(160).tag(sync=True, o=True)
    min_width = Int(160).tag(sync=True, o=True)

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

def get_img(a_web, nan=None):
    if nan is not None:
        a_web = np.where(a_web==nan, np.nan, a_web)
    if (np.all(np.isnan(a_web))):
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

def overlay(label, coord, m, current_io, width, da, func=None, nan=None):
    width2 = width / 2.
    lat, lon = coord
    tile = da.loc[1, lat+width2:lat-width2, lon-width2:lon+width2]
    value = int(tile.sel(y=lat, x=lon, method='nearest').values)

    values = tile.values
    if func is not None:
        values = func(values)
    imgurl = get_img(values, nan)
    bounds = [
        (tile.y[-1].values - 0.5 / 1200,
        tile.x[0].values - 0.5 / 1200),
        (tile.y[0].values + 0.5 / 1200,
        tile.x[-1].values + 0.5 / 1200)
        ]
    io = ImageOverlay(url=imgurl, bounds=bounds, opacity=0.5)
    if current_io is not None:
        m.remove_layer(current_io)
    m.add_layer(io)
    return io, value

class Control(object):
    def __init__(self, m, label):
        self.m = m
        self.label = label
        self.width = 0.1
        self.coord = None
        self.io = None
        self.s = None
        self.p = None
        self.show_data = ''
        self.show_menu = False
        self.da_acc = xr.open_rasterio('../data/hydrosheds/acc.vrt')
        self.da_dem = xr.open_rasterio('../data/hydrosheds/dem.vrt')
        self.marker = None
    def show(self, **kwargs):
        if not self.show_menu:
            if kwargs.get('type') == 'mousemove':
                self.coord = kwargs.get('coordinates')
                if self.show_data == 'flow':
                    self.io, flow = overlay(self.label, self.coord, self.m, self.io, self.width, self.da_acc, func=np.sqrt, nan=0)
                    self.label.value = f'lat/lon = {self.coord}, flow = {flow}'
                elif self.show_data == 'elevation':
                    self.io, elevation = overlay(self.label, self.coord, self.m, self.io, self.width, self.da_dem, nan=-32768)
                    self.label.value = f'lat/lon = {self.coord}, elevation = {elevation}'
                else:
                    self.label.value = f'lat/lon = {self.coord}'
            elif 'width' in kwargs:
                self.width = kwargs.get('width')
                if self.coord and self.show_data == 'flow':
                    self.io, flow = overlay(self.label, self.coord, self.m, self.io, self.width, self.da_acc, func=np.sqrt, nan=0)
                elif self.coord and self.show_data == 'elevation':
                    self.io, elevation = overlay(self.label, self.coord, self.m, self.io, self.width, self.da_dem, nan=-32768)
        if kwargs.get('type') == 'contextmenu':
            self.show_menu = True
            if self.show_data == 'flow':
                showHideFlow = 'Hide flow'
            else:
                showHideFlow = 'Show flow'
            if self.show_data == 'flow':
                self.s = ToggleButtons(options=['Hide flow', 'Delineate watershed', 'Set marker', 'Close'], value=None)
            elif self.show_data == 'elevation':
                self.s = ToggleButtons(options=['Hide elevation', 'Set marker', 'Close'], value=None)
            else:
                self.s = ToggleButtons(options=['Show flow', 'Show elevation', 'Set marker', 'Close'], value=None)
            self.s.observe(self.get_choice, names='value')
            self.p = Popup(location=self.coord, child=self.s, max_width=160, close_button=False, auto_close=True, close_on_escape_key=False)
            #self.p = CustomPopup(location=self.coord, child=self.s, max_width=160, close_button=False, auto_close=True, close_on_escape_key=False)
            self.m.add_layer(self.p)
    def get_choice(self, x):
        self.show_menu = False
        self.s.close()
        self.m.remove_layer(self.p)
        self.p = None
        choice = x['new']
        if choice == 'Show flow':
            self.show_data = 'flow'
        elif choice == 'Show elevation':
            self.show_data = 'elevation'
        elif choice == 'Hide flow' or choice == 'Hide elevation':
            self.show_data = ''
            self.m.remove_layer(self.io)
            self.io = None
        elif choice == 'Delineate watershed':
            self.show_data = ''
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
