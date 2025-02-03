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
import matplotlib.patheffects as path_effects
import calendar
import pandas as pd


regions = {}
regions['southern_africa'] = {'west': 12.5, 'east': 37.5, 'south': -30, 'north': -12.5}
regions['west_africa'] = {'west': -18, 'east': 10, 'south': 4, 'north': 20}
regions['east_africa'] = {'west': 34, 'east': 51.5, 'south': -2.5, 'north': 12.5}

seasons = {}
seasons['southern_africa'] = 'DJF'
seasons['west_africa'] = 'JJA'
seasons['east_africa'] = 'MAM'


def season_from_abbr(season_abbr):
    """
    Produce a list of month numbers from a season abbreviation,
    e.g. MAM -> [3, 4, 5]
    Parameters
    ----------
    season_abbr: str
        Abbreviation for season (i.e. initials of 2+ consecutive months)
    Returns
    -------
    month_list: list
        List of integers corresponding to the month number of each month in season (1=Jan,...)
    """
    if len(season_abbr) < 2:
        raise KeyError('Use seasons longer than one month')
    rolling_months = ''.join([m[0] for m in calendar.month_abbr[1:]])*2
    if season_abbr in rolling_months:
        season_start_idx = rolling_months.find(season_abbr)
        season_end_idx = season_start_idx + len(season_abbr)
        month_list = [(m%12)+1 for m in range(season_start_idx, season_end_idx)]
    else:
      raise KeyError(f'{season_abbr} not a valid sequence of months')
    return month_list


def get_regional_catalogue(cat, region_coords, season='all'):
    region_lats = np.logical_and(cat['latitude (degrees north)']>region_coords['south'], cat['latitude (degrees north)']<region_coords['north'])
    region_cat = cat.drop(cat[~region_lats].index)
    region_lons = np.logical_and(region_cat['longitude (degrees east)']<region_coords['east'], region_cat['longitude (degrees east)']>region_coords['west'])
    region_cat = region_cat.drop(region_cat[~region_lons].index)
    if season != 'all':
        season_months = season_from_abbr(season)
        month = region_cat['start date'].str[5:7].astype(int)
        in_season = np.isin(month, season_months)
        region_cat = region_cat.drop(region_cat[~in_season].index)
    return region_cat


def get_regional_n(cat, region_name, season='all'):
    region_coords = regions[region_name]
    region_cat = get_regional_catalogue(cat, region_coords, season=season)
    region_lons = np.arange(regions[region_name]['west'], regions[region_name]['east'], 0.25) + 0.5*0.25
    region_lats = np.arange(regions[region_name]['south'], regions[region_name]['north'], 0.25) + 0.5*0.25
    n = np.zeros((region_lats.size, region_lons.size))
    for event in region_cat.iterrows():
        lat_idx = int(np.where(region_lats == event[1]['latitude (degrees north)'])[0])
        lon_idx = int(np.where(region_lons == event[1]['longitude (degrees east)'])[0])
        n[lat_idx, lon_idx] += 1
    n = n.astype(float)
    n[n==0.] = np.nan
    return region_lons, region_lats, n


def plot_africa_map(max_n, ax=None, save=True, title='', cbar_vertical=False):
    cmap = cm.Oranges
    projection = ccrs.PlateCarree()
    if ax is None:
        fig = plt.figure(figsize=(6, 4.5)) 
        ax = plt.axes(projection=projection)
    else:
        fig = plt.gcf()
    if not cbar_vertical:
        cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.12, ax.get_position().width, 0.03])
    ax.coastlines(color='gray', linewidth=1,zorder=4)
    ax.set_title(title, fontsize=12)

    cat = pd.read_csv("/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_event_catalogue_with_event_anomalies.csv")
    regions_to_plot = ['west_africa', 'southern_africa', 'east_africa']
    for i, region_name in enumerate(regions_to_plot):
        region_lons, region_lats, region_n = get_regional_n(cat, region_name, season=seasons[region_name])
        region_lon_bounds = np.hstack((region_lons - 0.5*0.25, np.array([region_lons[-1]+0.5*0.25])))
        region_lat_bounds = np.hstack((region_lats - 0.5*0.25, np.array([region_lats[-1]+0.5*0.25])))
        p = ax.pcolormesh(region_lon_bounds, region_lat_bounds, region_n, cmap=cmap, vmin=0, vmax=max_n, transform=ccrs.PlateCarree())
        if i==0:
            if cbar_vertical:
                cbar = fig.colorbar(p, orientation='vertical', aspect=20, pad=0.05, extend='max')
                cbar.set_ticks(np.arange(0, max_n+1, max_n//5))
                cbar.ax.set_ylabel(f'number of\nflash droughts', fontsize=13)
            else:
                cbar = fig.colorbar(p, orientation='horizontal', cax=cax, aspect=40, pad=0.12, extend='max')
                cbar.set_ticks(np.arange(0, max_n+1, max_n//5))
                cbar.ax.set_xlabel(f'number of flash droughts', fontsize=14)
            cbar.ax.tick_params(labelsize=16)
        rect_width = regions[region_name]['east'] - regions[region_name]['west']
        rect_height = regions[region_name]['north'] - regions[region_name]['south']
        ax.add_patch(mpatches.Rectangle(xy=[regions[region_name]['west'], regions[region_name]['south']], width=rect_width, height=rect_height,
                                facecolor='none', edgecolor='k', zorder=5,
                                transform=ccrs.PlateCarree()))
        box_lon_centre = 0.5*(regions[region_name]['west']+regions[region_name]['east'])
        season_label = ax.text(box_lon_centre, regions[region_name]['north']+3., seasons[region_name], 
                               horizontalalignment='center', verticalalignment='center',zorder=6)
        season_label.set_path_effects([path_effects.withStroke(linewidth=3, foreground="w", alpha=0.7)])
    ax.set_xticks(np.arange(-20, 50, 20), crs=projection)
    ax.set_yticks(np.arange(-30, 31, 30), crs=projection)
    ax.set_extent([-22, 55, -30, 31])
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=14)
    ax.tick_params(axis='x', pad=5)
    if save:
        plt.savefig(f'../figures/africa_n_map.png', dpi=600, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    plot_africa_map(5)
