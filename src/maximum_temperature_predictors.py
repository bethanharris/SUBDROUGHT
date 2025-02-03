import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
from africa_seasonal_maps import plot_africa_map, season_from_abbr


regions = {}
regions['southern_africa'] = {'west': 12.5, 'east': 37.5, 'south': -30, 'north': -12.5}
regions['west_africa'] = {'west': -18, 'east': 10, 'south': 4, 'north': 20}
regions['east_africa'] = {'west': 34, 'east': 51.5, 'south': -2.5, 'north': 12.5}


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


def print_extremes_report(region_cat, extreme_threshold, quartile_variable):
    region_cat['quartile'] = pd.qcut(region_cat[quartile_variable], q=4, labels=(np.arange(4)+1).astype(str))
    print('*****')
    for q in np.arange(1, 5).astype(str):
        q_data = region_cat.loc[region_cat['quartile'] == q]
        total_number_events = len(q_data)
        events_over = (q_data['std_anom_t2m_max_0_20_smooth5']>extreme_threshold).sum()
        percent_over = float(events_over)/float(total_number_events)
        print(f'Q{q}: {100.*percent_over}% above {extreme_threshold}sigma ({events_over}/{total_number_events})')
    print('*****')


def plot_region_t2m_feedbacks(region_cat, region_label, quartile_variable, quartile_variable_label, title,
                              ax=None, save=True, show=True, legend=True, temp_legend=False, ax_labels=True):         
    dependent_variable =  'std_anom_t2m_max_0_20_smooth5'
    region_cat = region_cat[[dependent_variable, quartile_variable]].dropna()
    region_cat['quartile'] = pd.qcut(region_cat[quartile_variable], q=4, labels=(np.arange(4)+1).astype(str))
    plt.rc('legend',fontsize=12)
    plt.rcParams['legend.title_fontsize'] = 12
    if ax is None:
        plt.figure(figsize=(7.5, 4.5))
        ax = plt.gca()
    if 'lst-t2m' in quartile_variable:
        quartile_palette = ['#0571b0', '#92c5de', '#f4a582', '#ca0020']
    elif 'VODCA' in quartile_variable:
        quartile_palette = ['#a6611a','#dfc27d','#80cdc1','#018571']
    else:
        raise KeyError(f'Colour palette not set for quartile variable {quartile_variable}')
    p = sns.kdeplot(ax=ax, data=region_cat, x=dependent_variable, hue='quartile', linewidth=1.25, palette=quartile_palette)
    ax.tick_params(labelsize=14)
    if legend:
        ax.legend_.set_title(f'{quartile_variable_label} anomaly\nquartile:')
        ax.legend_.set_alignment('left')
        sns.move_legend(ax, "upper left")
        ax.text(0.02, 0.5, f'n = {len(region_cat)}', horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, fontsize=12)
    else:
        ax.legend_.set_visible(False)
        ax.text(0.05, 0.9, f'n = {len(region_cat)}', horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, fontsize=12)
    en_dash = '\u2013'
    if ax_labels:
        ax.set_xlabel(f'peak T2m anomaly', fontsize=14)
        ax.set_ylabel('probability density', fontsize=14)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax.set_title(title, fontsize=12)
    if save:
        plt.savefig(f'../figures/t2m_feedbacks/T2m_max_by_{quartile_variable}_{region_label}.png', dpi=800, bbox_inches='tight')
    if show:
        plt.show()


def plot_t2m_multiregion(cat, quartile_variable, quartile_variable_label):
    region_list = ['west_africa', 'east_africa', 'southern_africa']
    region_labels = ['West Africa', 'East Africa', 'Southern Africa']
    season_list = ['JJA', 'MAM', 'DJF']
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 2.5))
    axlist = axs.flatten()
    subfig_labels = ['$\\bf{(a)}$', '$\\bf{(b)}$', '$\\bf{(c)}$']
    for i, ax in enumerate(axlist):
        region_name = region_list[i]
        region_events = get_regional_catalogue(cat, regions[region_name], season=season_list[i])
        plot_region_t2m_feedbacks(region_events, region_name, quartile_variable, quartile_variable_label,
                                  f'{subfig_labels[i]} {region_labels[i]} {season_list[i]}', ax=ax, save=False, show=False, legend=False, ax_labels=False)
        ax.set_xticks(np.arange(-1, 4, 1))
    plt.subplots_adjust(right=0.8)
    fig.legend(ax.legend_.legend_handles, [t.get_text() for t in ax.legend_.texts], bbox_to_anchor=(0.82, 0.5),
               loc='center left', title_fontsize=14, fontsize=14, alignment='left')
    fig.legends[0].set_title(f'{quartile_variable_label} anomaly\nquartile:') 
    fig.text(0.5, -0.08, 'maximum 2m air temperature standardised anomaly', ha='center', fontsize=16)
    fig.text(0.055, 0.5, 'probability density', va='center', rotation='vertical', fontsize=16)
    plt.savefig(f'../figures/t2m_feedbacks/T2m_{quartile_variable}_multiregion_africa.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'../figures/t2m_feedbacks/T2m_{quartile_variable}_multiregion_africa.png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_t2m_multiregion_with_map(cat, quartile_variable, quartile_variable_label):
    region_list = ['west_africa', 'east_africa', 'southern_africa']
    region_labels = ['West Africa', 'East Africa', 'Southern Africa']
    season_list = ['JJA', 'MAM', 'DJF']
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    axlist = [ax1, ax2, ax3, ax4]
    plot_africa_map(5, ax=axlist[0], save=False, cbar_vertical=True, title='$\\bf{(a)}$')
    subfig_labels = ['$\\bf{(b)}$', '$\\bf{(c)}$', '$\\bf{(d)}$']
    for i, ax in enumerate(axlist[1:]):
        region_name = region_list[i]
        region_events = get_regional_catalogue(cat, regions[region_name], season=season_list[i])
        plot_region_t2m_feedbacks(region_events, region_name, quartile_variable, quartile_variable_label,
                                  f'{subfig_labels[i]} {region_labels[i]} {season_list[i]}', ax=ax, save=False, show=False, legend=False, ax_labels=False)
        ax.set_xticks(np.arange(-1, 4, 1))
    plt.subplots_adjust(wspace=0.3, hspace=0.35, right=0.8)
    fig.legend(ax.legend_.legend_handles, [t.get_text() for t in ax.legend_.texts], bbox_to_anchor=(0.82, 0.5),
               loc='center left', title_fontsize=14, fontsize=14, alignment='left')
    fig.legends[0].set_title(f'{quartile_variable_label} anomaly\nquartile:') 
    fig.text(0.5, 0, 'maximum 2m air temperature standardised anomaly', ha='center', fontsize=16)
    fig.text(0.05, 0.5, 'probability density', va='center', rotation='vertical', fontsize=16)
    plt.savefig(f'../figures/t2m_feedbacks/T2m_{quartile_variable}_multiregion_africa_with_map.png', dpi=400, bbox_inches='tight')
    plt.savefig(f'../figures/t2m_feedbacks/T2m_{quartile_variable}_multiregion_africa_with_map.pdf', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    os.system('mkdir -p ../figures/t2m_feedbacks')
    cat = pd.read_csv('/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_event_catalogue_with_event_anomalies.csv')
    plot_t2m_multiregion(cat, 'std_anom_VODCA_CXKu_mean_-60_-30', 'VOD')
    plot_t2m_multiregion(cat, 'std_anom_lst-t2m_mean_-60_-30nweighted', '$\\Delta T$')
    plot_t2m_multiregion_with_map(cat, 'std_anom_lst-t2m_max_0_20_smooth5', '$\\Delta T$')
   
    region_label = 'west_africa'
    region_cat = get_regional_catalogue(cat, regions[region_label], season='JJA')
    print_extremes_report(region_cat, 1.5, 'std_anom_lst-t2m_max_0_20_smooth5')
    print_extremes_report(region_cat, 1.5, 'std_anom_VODCA_CXKu_mean_-60_-30')
