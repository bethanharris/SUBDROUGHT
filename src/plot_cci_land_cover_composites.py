# from composites_by_cci_land_cover import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string


variable_legend_labels = {'ssm_pentad_means': 'surface soil moisture',
                         'ssm': 'SSM',
                         'ssm_GLEAM': 'GLEAM SSM (10cm)',
                         'vod': 'VOD',
                         'vod_v2': 'VOD (v2)',
                         'vod_v2_SIFyears': 'VOD (v2)',
                         'vod_v1_nofilter': 'VOD (v1 unfiltered)',
                         'rzsm': 'RZSM-GLEAM',
                         'rzsm_cci': 'ESA CCI RZSM (1m)',
                         'rzsm_cci_40cm': 'ESA CCI RZSM (40cm)',
                         'rzsm_cci_10cm': 'ESA CCI RZSM (10cm)',
                         'rzsm_ERA5_7cm': "ERA5 RZSM (7cm)",
                         't2m': 'T2m',
                         'evap': 'latent heat flux',
                         'lst_mw': 'LST (MW)',
                         'precip': 'precipitation (IMERG)',
                         'tdiff_mw_18': '$\Delta T$ (MW)',
                         'tdiff_aqua': '$\Delta T$ (MODIS Aqua)',
                         'vimd_mean': 'moisture divergence',
                         'vimd_max': 'vimd_max',
                         'rad': 'net surface radiation ($\\mathrm{R_n}$)',
                         'sw_down': 'downwelling SW',
                         'wind_speed': '10m wind speed (ERA5)',
                         'SESR_GLEAM': 'ESR (GLEAM)',
                         'SESR_ERA5': 'ESR (ERA5)',
                         'SIF_PK': 'SIF (PK)',
                         'SIF_JJ': 'SIF (JJ)',
                         'vpd': 'VPD (ERA5)'
                         }

variable_colours = {'ssm_pentad_means': 'k',
                    'ssm': '#cccccc',
                    'ssm_GLEAM': 'k',
                    'vod': '#009E73',
                    'vod_v2': 'C2',
                    'vod_v2_SIFyears': 'C2',
                    'vod_v1_nofilter': '#084719',
                    'rzsm': '#F35B8F',
                    'rzsm_cci': 'b',
                    'rzsm_cci_40cm': 'r',
                    'rzsm_cci_10cm': 'c',
                    'rzsm_ERA5_7cm': 'C1',
                    't2m': 'r',# '#D55E00',
                    'evap': '#88CCEE',
                    'lst_mw': '#CC79A7',
                    'precip': '#0072B2',
                    'tdiff_mw_18': '#C65ED6',
                    'tdiff_aqua': '#882255',
                    'vimd_mean': '#691681',
                    'sw_down': '#F0E442',
                    'rad': '#DDCC77',
                    'wind_speed': '#11026e',
                    'SESR_GLEAM': '#DC267F',
                    'SESR_ERA5': 'C1',
                    'SIF_JJ': '#52c728',
                    'SIF_PK': '#52c728',
                    'vpd': 'purple'
                    }


def plot_composites_single_land_cover(land_cover, variables_to_plot, ax=None, show=True, save_name=None, legend=True, title=''):
    if ax is None:
        # plt.figure(figsize=(6, 3.375))
        plt.figure(figsize=(6, 4.5))
        ax = plt.gca()
    ax.axhline(color='gray', linewidth=0.5)
    ax.axvline(color='gray', linewidth=0.5)
    for v in variables_to_plot:
        if v == 'tdiff_aqua':
            composite = pd.read_csv(f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/ESA_CCI_land_cover_composites/all_seasons/{v}_n-weighted_composite_{land_cover}_mask20pct.csv')   
        else:
            composite = pd.read_csv(f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/ESA_CCI_land_cover_composites/all_seasons/{v}_composite_{land_cover}.csv')
        if v == 'sw_down':
            linestyle = '--'
        else:
            linestyle = '-'
        ax.plot(composite['days_since_drought_start'][60:-60], composite['composite_mean'][60:-60], 
                label=variable_legend_labels[v], color=variable_colours[v], linewidth=2, linestyle=linestyle)
    n = pd.read_csv(f'/prj/nceo/bethar/SUBDROUGHT/HESS_paper/ESA_CCI_land_cover_composites/all_seasons/ssm_pentad_means_composite_{land_cover}.csv')['composite_n'][60:-60]
    # ax.text(0.03, 0.9, f'{int(n.max())} events', fontsize=13, transform=ax.transAxes)
    ax.text(0.03, 0.03, f'max(n) = {int(n.max())}', fontsize=13, transform=ax.transAxes)
    if legend:
        ax.legend(loc='lower left', fontsize=10, frameon=False, bbox_to_anchor=(0,-0.02))
    ax.tick_params(labelsize=14)
    ax.set_xlim([-60, 60])
    ax.set_xticks(np.arange(-60, 61, 30))
    ax.set_xlabel('days since flash drought onset', fontsize=14)
    ax.set_ylabel('standardised anomaly', fontsize=14)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(f'../figures/single_panel_composites/{save_name}.png', dpi=800)
    if show:
        plt.show()


def plot_composites_top4_land_covers(variables_to_plot, save_name):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 6.25))
    axlist = ax.flatten()
    top_lcs = ['all_cropland_rainfed', 'all_shrubland', 'grassland', 'all_broadleaved_deciduous']
    alphabet = string.ascii_lowercase
    for i, lc in enumerate(top_lcs):
        ax = axlist[i]
        lc_name_neat = lc.strip('all_').replace('_', ' ')
        neat_title = f'$\\bf{{({alphabet[i]})}}$ {lc_name_neat}'
        plot_composites_single_land_cover(lc, variables_to_plot, show=False, ax=ax, legend=False, title=neat_title)
        ax.label_outer()
        h, l = ax.get_legend_handles_labels()
    plt.subplots_adjust(wspace=0.15, right=0.9)
    fig.legend(h, l, bbox_to_anchor=(1.1, 0.5), loc='center', fontsize=14)
    plt.savefig(f'../figures/{save_name}.png', dpi=500, bbox_inches='tight')
    plt.savefig(f'../figures/{save_name}.pdf', dpi=500, bbox_inches='tight')
    plt.show()

    

if __name__ == '__main__':
    plot_composites_top4_land_covers(['ssm_pentad_means', 'tdiff_aqua', 'tdiff_mw_18', 'rad', 'evap'], save_name='top_4_lc_SEB_main_fig_mask20pct_GLEAMv42_ALLSEASONS_newlabels')
    plot_composites_top4_land_covers(['rad', 'sw_down', 'precip', 'wind_speed'], save_name='top_4_lc_SEB_suppinfo_ALLSEASONS')
