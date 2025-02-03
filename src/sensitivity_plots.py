import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.feature as feat
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from plot_utils import binned_cmap, remappedColorMap
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import pandas as pd


def sensitivity_subplots():
    px_deg = 2.5
    vod_sensitivity = np.load(f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/sensitivity/VOD_v2_sensitivity/sensitivity_{px_deg}deg.npy')
    t2m_sensitivity = np.load(f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/sensitivity/T2m_sensitivity/sensitivity_{px_deg}deg.npy')
    LHFmax_sensitivity = np.load(f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/sensitivity/LHF_max_sensitivity/sensitivity_{px_deg}deg.npy')

    vod_cmap = remappedColorMap(cm.BrBG, np.array([-1.25, np.nanmax(-vod_sensitivity)]))
    t2m_cmap = remappedColorMap(cm.RdYlBu_r, np.array([np.nanmin(-t2m_sensitivity), 1.1]))
    LHFmax_cmap = remappedColorMap(cm.PuOr, np.array([-0.25, 1]))
    vod_cmap.set_bad('#cccccc')
    t2m_cmap.set_bad('#cccccc')
    LHFmax_cmap.set_bad('#cccccc')

    lons = np.arange(-180, 180, px_deg) + 0.5*px_deg
    lats = np.arange(-60, 80, px_deg) + 0.5*px_deg
    lon_bounds = np.hstack((lons - 0.5*px_deg, np.array([lons[-1]+0.5*px_deg])))
    lat_bounds = np.hstack((lats - 0.5*px_deg, np.array([lats[-1]+0.5*px_deg])))

    mosaic = """AABB
                .CC."""
    fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    vod_ax = ax_dict['A']
    t2m_ax = ax_dict['B']
    LHFmax_ax = ax_dict['C']
    fig.subplots_adjust(hspace=0.4, wspace=0.5)

    p_vod = vod_ax.pcolormesh(lon_bounds, lat_bounds, -vod_sensitivity, cmap=vod_cmap, vmin=-1.25, transform=ccrs.PlateCarree(), rasterized=True)    
    cax = fig.add_axes([vod_ax.get_position().x0, vod_ax.get_position().y0-0.08, vod_ax.get_position().width, 0.03])
    cbar = fig.colorbar(p_vod, orientation='horizontal', cax=cax, aspect=40, pad=0.12, extend='max')
    # cbar.set_ticks(np.arange(0, max_n+1, max_n//5))
    cbar.ax.set_xlabel("VOD sensitivity", fontsize=15)
    cbar.ax.tick_params(labelsize=16)
    vod_ax.coastlines(color='black', linewidth=1)
    vod_ax.set_xticks(np.arange(-180, 181, 90), crs=ccrs.PlateCarree())
    vod_ax.set_yticks(np.arange(-60, 61, 60), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    vod_ax.xaxis.set_major_formatter(lon_formatter)
    vod_ax.yaxis.set_major_formatter(lat_formatter)
    vod_ax.tick_params(labelsize=14)
    vod_ax.tick_params(axis='x', pad=5)
    vod_ax.set_title('(a)', weight='bold', fontsize=16)

    p_t2m = t2m_ax.pcolormesh(lon_bounds, lat_bounds, -t2m_sensitivity, cmap=t2m_cmap, vmax=1.1, transform=ccrs.PlateCarree(), rasterized=True)
    cax = fig.add_axes([t2m_ax.get_position().x0, t2m_ax.get_position().y0-0.08, t2m_ax.get_position().width, 0.03])
    cbar = fig.colorbar(p_t2m, orientation='horizontal', cax=cax, aspect=40, pad=0.12, extend='max')
    # cbar.set_ticks(np.arange(0, max_n+1, max_n//5))
    cbar.ax.set_xlabel("2m air temperature sensitivity", fontsize=15)
    cbar.ax.tick_params(labelsize=16)
    t2m_ax.coastlines(color='black', linewidth=1)
    t2m_ax.set_xticks(np.arange(-180, 181, 90), crs=ccrs.PlateCarree())
    t2m_ax.set_yticks(np.arange(-60, 61, 60), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    t2m_ax.xaxis.set_major_formatter(lon_formatter)
    t2m_ax.yaxis.set_major_formatter(lat_formatter)
    t2m_ax.tick_params(labelsize=14)
    t2m_ax.tick_params(axis='x', pad=5)
    t2m_ax.set_title('(b)', weight='bold', fontsize=16)

    p_LHFmax = LHFmax_ax.pcolormesh(lon_bounds, lat_bounds, -LHFmax_sensitivity, cmap=LHFmax_cmap, vmin=-0.25, vmax=1, transform=ccrs.PlateCarree(), rasterized=True)
    cax = fig.add_axes([LHFmax_ax.get_position().x0, LHFmax_ax.get_position().y0-0.08, LHFmax_ax.get_position().width, 0.03])
    cbar = fig.colorbar(p_LHFmax, orientation='horizontal', cax=cax, aspect=40, pad=0.12, extend='max')
    # cbar.set_ticks(np.arange(0, max_n+1, max_n//5))
    cbar.ax.set_xlabel("LHF$_{\\mathrm{max}}$ sensitivity", fontsize=15)
    cbar.ax.tick_params(labelsize=16)
    LHFmax_ax.coastlines(color='black', linewidth=1)
    LHFmax_ax.set_xticks(np.arange(-180, 181, 90), crs=ccrs.PlateCarree())
    LHFmax_ax.set_yticks(np.arange(-60, 61, 60), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    LHFmax_ax.xaxis.set_major_formatter(lon_formatter)
    LHFmax_ax.yaxis.set_major_formatter(lat_formatter)
    LHFmax_ax.tick_params(labelsize=14)
    LHFmax_ax.tick_params(axis='x', pad=5)
    LHFmax_ax.set_title('(c)', weight='bold', fontsize=16)
    
    plt.savefig(f'../figures/vod_t2m_LHFmax_sensitivity_{px_deg}deg.png', dpi=800, bbox_inches='tight')
    plt.savefig(f'../figures/vod_t2m_LHFmax_sensitivity_{px_deg}deg.pdf', dpi=800, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    sensitivity_subplots()
