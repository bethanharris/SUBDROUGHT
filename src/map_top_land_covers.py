import xarray as xr
import dask.array
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from plot_utils import binned_cmap


lc = xr.open_dataset("/prj/nceo/bethar/CCI_landcover/0pt25deg/CCI_land_cover_0pt25deg_2020.nc")
global_lcc = (dask.array.floor(lc.lccs_class/10.)*10).astype(int).squeeze().sel(lat=slice(-60, 80))

classes_to_plot = {10: 'rainfed\ncropland', 120: 'shrubland', 130: 'grassland', 60: 'broadleaf\ndeciduous'}
list_class_codes = list(classes_to_plot.keys())

masked_global_lcc = global_lcc.where(dask.array.isin(global_lcc, list_class_codes)).data.compute()
masked_global_lcc_indexed = np.copy(masked_global_lcc)
for i, lc_code in enumerate(list_class_codes):
    masked_global_lcc_indexed[masked_global_lcc==lc_code] = i


px_deg = 0.25
n_classes = len(list_class_codes)
class_colours = ['#44AA88','#FFDD44', '#AADDCC', '#99BB55']
lons = np.arange(-180, 180, px_deg) + 0.5*px_deg
lats = np.arange(-60, 80, px_deg) + 0.5*px_deg
lon_bounds = np.hstack((lons - 0.5*px_deg, np.array([lons[-1]+0.5*px_deg])))
lat_bounds = np.hstack((lats - 0.5*px_deg, np.array([lats[-1]+0.5*px_deg])))
cmap, norm = binned_cmap(np.arange(n_classes+1)-0.5, 'Oranges', fix_colours=[(i, c) for i, c in enumerate(class_colours)])

projection = ccrs.PlateCarree()
fig = plt.figure(figsize=(5, 2.5)) 
ax = plt.axes(projection=projection)
p = plt.pcolormesh(lon_bounds, lat_bounds, masked_global_lcc_indexed, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), rasterized=True)
cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.22, ax.get_position().width, 0.07])
cbar = fig.colorbar(p, orientation='horizontal', cax=cax, aspect=40, pad=0.42)
cbar.set_ticks(np.arange(n_classes))
cbar.set_ticklabels(classes_to_plot.values(), fontsize=13, rotation=0)
cbar.ax.minorticks_off()
ax.coastlines(color='black', linewidth=1)
ax.set_xticks(np.arange(-180, 181, 90), crs=projection)
ax.set_yticks(np.arange(-60, 61, 60), crs=projection)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(labelsize=14)
ax.tick_params(axis='x', pad=5)
plt.savefig(f'../figures/top_4_lcs.png', dpi=600, bbox_inches='tight')
plt.savefig(f'../figures/top_4_lcs.pdf', dpi=600, bbox_inches='tight')
plt.show()