import os
import pandas as pd
import numpy as np
import xarray as xr
import cc3d
from dask.diagnostics import ProgressBar
from scipy import ndimage
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from maximum_temperature_predictors import get_regional_catalogue


os.system('mkdir -p ../figures/t2m_feedbacks/bootstrapping_95ci/')
os.system('mkdir -p ../data/bootstraps')

regions = {}
regions['southern_africa'] = {'west': 12.5, 'east': 37.5, 'south': -30, 'north': -12.5}
regions['west_africa'] = {'west': -18, 'east': 10, 'south': 4, 'north': 20}
regions['east_africa'] = {'west': 34, 'east': 51.5, 'south': -2.5, 'north': 12.5}


def save_fuzzy_clusters():
    # Join nearby flash drought pixels up into larger-scale events
    fd = xr.open_dataset("/prj/nceo/bethar/SUBDROUGHT/HESS_paper/subseasonal_drought_development_events_mask_frozen.nc", chunks={'time':-1,'latitude':40,'longitude':40})
    with ProgressBar():
        flags = fd.sm.data.compute()
    blur_size = 3
    blur_structure = ndimage.generate_binary_structure(3, 2)
    blur_structure[0] = 0
    blur_structure[-1] = 0
    flags_fuzzy = ndimage.binary_dilation(flags, structure=blur_structure, iterations=blur_size)
    labels_out = cc3d.connected_components(flags_fuzzy, connectivity=26)
    labels_out[flags==0] = 0 # remove wherever was only put into "flash drought" as part of the structure blurring
    fd_clusters = fd.sm.copy(data=labels_out)
    fd_clusters.to_netcdf('/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_clusters_fuzzy_all_seasons.nc', encoding={'sm': {'chunksizes': (7671, 40, 40), 'zlib': True}})


def add_clusters_to_catalogue(regional_cat, cluster_map, region_coords):
    # Add a column to the flash drought catalogue to indicate which larger cluster each pixel-wise event belongs to
    fd_clusters_region = cluster_map.sel(latitude=slice(region_coords['south'], region_coords['north']), longitude=slice(region_coords['west'], region_coords['east']))
    lats = fd_clusters_region.latitude.data
    lons = fd_clusters_region.longitude.data
    times = fd_clusters_region.time.data
    cluster_ids = []
    fd_clusters_region_data = fd_clusters_region.sm.data.compute()
    for _, row in regional_cat.iterrows():
        event_lat_idx = np.where(lats == row['latitude (degrees north)'])[0]
        event_lon_idx = np.where(lons == row['longitude (degrees east)'])[0]
        event_time_idx = np.where(times == np.datetime64(row['start date'], 'ns'))
        cluster = fd_clusters_region_data[event_time_idx, event_lat_idx, event_lon_idx]
        cluster_ids.append(cluster[0][0])
    regional_cat['cluster'] = np.array(cluster_ids)
    return regional_cat


def save_mann_whitney_probabilities(region_name, season_abbr, quartile_variable, quartile_variable_label):
    fd_clusters = xr.open_dataset("/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_clusters_fuzzy_all_seasons.nc", chunks={'time': -1, 'latitude': 40, 'longitude': 40})
    region = regions[region_name]
    cat = pd.read_csv("/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_event_catalogue_with_event_anomalies.csv")
    region_cat = get_regional_catalogue(cat, region, season_abbr)
    region_cluster_cat = add_clusters_to_catalogue(region_cat, fd_clusters, region)
    dependent_variable =  'std_anom_t2m_max_0_20_smooth5'
    region_cluster_cat = region_cluster_cat[[quartile_variable, dependent_variable, 'cluster']].dropna()
    # cluster resampling
    all_clusters = np.unique(region_cluster_cat['cluster'], return_counts=True)
    large_clusters = []
    cluster_px_size = []
    for i in range(all_clusters[0].size):
        if all_clusters[1][i]>=0: # can use this to set threshold for how many pixels must be in cluster
            large_clusters.append(all_clusters[0][i])
            cluster_px_size.append(all_clusters[1][i])
    clusters = np.array(large_clusters)
    region_cluster_cat = region_cluster_cat.loc[np.isin(region_cluster_cat['cluster'], clusters)]
    prob_higher = []
    resample_size = []
    print('****')
    print(region_name)
    print(quartile_variable_label)
    print(f'number of pixels: {len(region_cluster_cat)}')
    print(f'number of clusters: {clusters.size}')
    if clusters.size >= 10:
        for i in tqdm(range(1000)):
            resample_clusters = np.random.choice(clusters, clusters.size, replace=True) # resample clusters with replacement
            resample_df = pd.DataFrame()
            for c in resample_clusters:
                cluster_data = region_cluster_cat.loc[region_cluster_cat['cluster'] == c]
                resample_within_cluster = cluster_data.sample(frac=1, replace=False) # set to no resampling within cluster, as recommended by Davison & Hinkley 1997.
                resample_df = pd.concat([resample_df, resample_within_cluster], axis=0)
            resample_size.append(len(resample_df))
            resample_df['quartile'] = pd.qcut(resample_df[quartile_variable], q=4, labels=(np.arange(4)+1).astype(str))
            Q4 = resample_df.loc[resample_df['quartile'] == '4'][dependent_variable]
            Q1 = resample_df.loc[resample_df['quartile'] == '1'][dependent_variable]
            mw = mannwhitneyu(Q1, Q4, alternative='greater', nan_policy='omit')
            prob_higher.append(mw.statistic/(len(Q1)*len(Q4)))  
        print(f'mean number of pixels in resamples: {int(np.array(resample_size).mean())}')
        region_cluster_cat['quartile'] = pd.qcut(region_cluster_cat[quartile_variable], q=4, labels=(np.arange(4)+1).astype(str))
        q4 = region_cluster_cat.loc[region_cluster_cat['quartile'] == '4'][dependent_variable]
        q1 = region_cluster_cat.loc[region_cluster_cat['quartile'] == '1'][dependent_variable]
        obs_mw = mannwhitneyu(q1, q4, alternative='greater', nan_policy='omit')
        obs_prob_higher = obs_mw.statistic/(len(q1)*len(q4))
        if 'lst-t2m' in quartile_variable: # expect Q4>Q1 instead of Q1>Q4
            prob_higher = [1.-p for p in prob_higher]
            obs_prob_higher = 1.-obs_prob_higher
            high_quartile = 'Q4'
        else:
            high_quartile = 'Q1'
        p5 = np.percentile(np.array(prob_higher), 5)
        print(f'95% CI: [{p5:.2f}, 1]')
        np.savetxt(f"../data/bootstraps/prob_{high_quartile}_higher_distribution_{region_name}_{quartile_variable_label}toT2m_{season_abbr}.csv", prob_higher, delimiter=",")
        np.savetxt(f"../data/bootstraps/prob_{high_quartile}_higher_distribution_{region_name}_{quartile_variable_label}toT2m_{season_abbr}_OBSSTAT.csv", np.array([obs_prob_higher]), delimiter=",")


def significance_histogram_subplots(quartile_variable, quartile_variable_label):
    dependent_variable =  'std_anom_t2m_max_0_20_smooth5'
    if 'lst-t2m' in quartile_variable:
        high_quartile = 'Q4'
        legend_loc = 'upper right'
    else:
        high_quartile = 'Q1'
        legend_loc = 'upper left'
    region_list = ['west_africa', 'east_africa', 'southern_africa']
    region_labels = ['West Africa', 'East Africa', 'Southern Africa']
    season_list = ['JJA', 'MAM', 'DJF']
    fig, axs = plt.subplots(1, 3, sharex=False, sharey=True, figsize=(12, 2.5))
    axlist = axs.flatten()
    subfig_labels = ['$\\bf{(a)}$', '$\\bf{(b)}$', '$\\bf{(c)}$']
    for i, ax in enumerate(axlist):
        fd_clusters = xr.open_dataset("/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_clusters_fuzzy_all_seasons.nc", chunks={'time': -1, 'latitude': 40, 'longitude': 40})
        region_name = region_list[i]
        region = regions[region_name]
        cat = pd.read_csv("/prj/nceo/bethar/SUBDROUGHT/HESS_paper/flash_drought_event_catalogue_with_event_anomalies.csv")
        region_cat = get_regional_catalogue(cat, region, season_list[i])
        region_cluster_cat = add_clusters_to_catalogue(region_cat, fd_clusters, region)
        region_cluster_cat = region_cluster_cat[[quartile_variable, dependent_variable, 'cluster']].dropna()
        number_clusters = np.unique(region_cluster_cat['cluster'].values).size
        bootstrapped_probs = np.genfromtxt(f"../data/bootstraps/prob_{high_quartile}_higher_distribution_{region_name}_{quartile_variable_label}toT2m_{season_list[i]}.csv", delimiter=",")
        obs_stat = float(np.genfromtxt(f"../data/bootstraps/prob_{high_quartile}_higher_distribution_{region_name}_{quartile_variable_label}toT2m_{season_list[i]}_OBSSTAT.csv", delimiter=","))
        pc5 = np.nanpercentile(bootstrapped_probs, 5)
        ax.hist(bootstrapped_probs, bins=20, edgecolor='k', facecolor='#5a95bf', linewidth=0.75)
        ax.axvline(obs_stat, color='#D55E00')
        ax.axvline(pc5, color='#D55E00', linestyle='--')
        if quartile_variable_label == 'SHFmax':
            ax.set_xticks([0.5, 0.7, 0.9])
            ax.set_xlim([0.4, 0.9])
        elif quartile_variable_label == 'preSHF':
            ax.set_xticks([0.3, 0.5, 0.7, 0.9])
            ax.set_xlim([0.2, 0.9])
        else:
            ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        ax.set_title(f'{subfig_labels[i]} {region_labels[i]} {season_list[i]}', fontsize=14)
        obs_line = Line2D([0], [0], color='#D55E00')
        p5_line = Line2D([0], [0], color='k', linestyle='--')
        ax.legend([obs_line, p5_line], [f'observed = {obs_stat:.2f}', f'95% CI = [{pc5:.2f}, 1]'], loc=legend_loc, fontsize=7, framealpha=1)
        if 'lst-t2m' in quartile_variable:
            ax.text(0.98, 0.75, f'{number_clusters} clusters', ha='right', fontsize=7, transform=ax.transAxes)
        else:
            ax.text(0.02, 0.75, f'{number_clusters} clusters', fontsize=7, transform=ax.transAxes)
    plt.subplots_adjust(right=0.9)
    if 'lst-t2m' in quartile_variable:
        fig.text(0.5, -0.08, '$P\\left(\\mathrm{Q}4>\\mathrm{Q}1\\right)$', ha='center', fontsize=16)
    else:
        fig.text(0.5, -0.08, '$P\\left(\\mathrm{Q}1>\\mathrm{Q}4\\right)$', ha='center', fontsize=16)
    fig.text(0.055, 0.5, 'number of\nbootstrap samples', va='center', ha='center', rotation='vertical', fontsize=14)
    plt.savefig(f'../figures/t2m_feedbacks/bootstrapping_95ci/T2m_{quartile_variable_label}_multiregion_africa_bootstrapping_95ci.png', dpi=400, bbox_inches='tight')
    plt.savefig(f'../figures/t2m_feedbacks/bootstrapping_95ci/T2m_{quartile_variable_label}_multiregion_africa_bootstrapping_95ci.pdf', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    region_seasons = {'west_africa': 'JJA', 'east_africa': 'MAM', 'southern_africa': 'DJF'}
    quartile_variable_labels = {'std_anom_lst-t2m_max_0_20_smooth5': 'SHFmax',
                                'std_anom_lst-t2m_mean_-60_-30nweighted': 'preSHF',
                                'std_anom_VODCA_CXKu_mean_-60_-30': 'preVOD'}
    for quartile_variable in quartile_variable_labels.keys():
        for region_name in region_seasons.keys():
            save_mann_whitney_probabilities(region_name, region_seasons[region_name],
                                            quartile_variable, quartile_variable_labels[quartile_variable])
        significance_histogram_subplots(quartile_variable, quartile_variable_labels[quartile_variable])
