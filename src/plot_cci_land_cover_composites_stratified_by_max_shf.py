# from composites_by_cci_land_cover_stratified_by_max_shf import *
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt


variable_legend_labels = {'ssm_pentad_means_temp': 'SSM (ESA CCI)',
                        #  'ssm_pentad_means': 'SSM',
                         'ssm_pentad_means': 'ESA CCI SM',
                         'ssm': 'SSM',
                         'ssm_GLEAM': 'SSM (GLEAM)',
                         'vod': 'VOD',
                         'vod_v2': 'VOD (VODCA-CXKu)',
                         'vod_v1_nofilter': 'VOD (v1 unfiltered)',
                         'rzsm': 'RZSM (GLEAM)',
                         'rzsm_cci': 'RZSM (CCI)',
                         'rzsm_cci_10cm': 'RZSM (CCI 10cm)',
                         't2m': 'T2m (ERA5)',
                         'evap': 'LHF (GLEAM)',
                         'lst_mw': 'LST (MW)',
                         'lst_aqua': 'LST (MODIS Aqua)',
                         'wind_speed': '10m wind speed (ERA5)',
                         'wind_speed_raw_data': '10m wind speed (raw)',
                         'precip': 'precipitation (IMERG)',
                         'tdiff_mw_18':  '$\\mathrm{\Delta T}$\n(MW LST - ERA5 T2m)',
                        #  'tdiff_aqua': 'SHF (MODIS Aqua)',
                         'tdiff_aqua': '$\\mathrm{\Delta T}$\n(MODIS Aqua LST - ERA5 T2m)',
                         'vimd_mean': 'moisture div.',
                         'vimd_max': 'vimd_max',
                         'rad': '$\\mathrm{R_n}$ (CERES)',
                         'sw_down': 'downwelling SW (CERES)',
                         'SESR_GLEAM': 'SESR (GLEAM)',
                         'SESR_ERA5': 'SESR (ERA5)',
                         'SIF_JJ': 'SIF (GOME2-JJ)',
                         'SIF_PK': 'SIF (GOME2-PK)',
                         'vpd': 'VPD (ERA5)'
                         }


variable_colours = {'ssm_pentad_means': 'k',
                    'ssm': '#cccccc',
                    'ssm_GLEAM': 'k',
                    'vod': '#009E73',
                    'vod_v1_nofilter': '#084719',
                    'vod_v2': 'C2',
                    'rzsm': '#F35B8F',
                    'rzsm_cci': 'b',
                    'rzsm_cci_40cm': 'r',
                    'rzsm_cci_10cm': 'c',
                    'rzsm_ERA5_7cm': 'C1',
                    't2m': 'r',
                    'evap': '#88CCEE',
                    'lst_mw': '#CC79A7',
                    'lst_aqua': '#CC79A7',
                    'precip': '#0072B2',
                    'tdiff_mw_18': '#C65ED6',
                    'tdiff_aqua_temp': '#882255',
                    'tdiff_aqua': '#882255',
                    'vimd_mean': '#691681',
                    'sw_down': '#DDCC77',
                    'rad': '#DDCC77',
                    'wind_speed': 'C0',
                    'wind_speed_raw_data': 'C0',
                    'SESR_GLEAM': '#DC267F',
                    'SESR_ERA5': 'C1',
                    'SIF_JJ': '#52c728',
                    'SIF_PK': '#52c728',
                    'vpd': 'purple'
                    }

fixed_ylims = {'tdiff_aqua': [-0.2, 0.97],
         'lst_aqua': [-0.2, 0.97],
         't2m': [-0.2, 0.97],
         'ssm_pentad_means': [-1.61, 0.17],
         'rzsm': [-1.61, 0.17],
         'SESR_ERA5': [-1.61, 0.17]}


def compare_quartiles_by_variable_main_fig(land_cover, show=True, smooth=False, save_name=None):
    variables_to_plot = ['tdiff_aqua', 'lst_aqua', 't2m', 'evap', 'rad', 'sw_down','ssm_pentad_means', 'rzsm', 'SESR_ERA5', 'vod_v2', 'SIF_PK', 'precip']
    num_rows = np.ceil(np.sqrt(len(variables_to_plot)))
    num_columns = np.ceil(float(len(variables_to_plot))/float(num_rows))
    linestyles = ['-', '--', ':', '-.']
    save_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/ESA_CCI_land_cover_composites/stratified_by_max_shf/mask20pct'
    fig, axs = plt.subplots(int(num_rows), int(num_columns), sharex=True, figsize=(16, 16))
    plt.subplots_adjust(wspace=18)
    axlist = axs.flatten()
    alphabet = string.ascii_lowercase
    for i, v in enumerate(variables_to_plot):
        ax = axlist[i]
        for q in range(1, 5):
            if v == 'tdiff_aqua' or v == 'lst_aqua':
                composite = pd.read_csv(f'{save_dir}/{v}_n-weighted_composite_{land_cover}_Q{q}.csv')
            else:
                composite = pd.read_csv(f'{save_dir}/{v}_composite_{land_cover}_Q{q}.csv')
            if smooth:
                window = 10
                ax.plot(composite['days_since_drought_start'], composite['composite_mean'].rolling(window=window, min_periods=1, center=True).mean(), 
                    label=f'Q{q}', color=variable_colours[v], linestyle=linestyles[q-1], linewidth=2)
            else:
                ax.plot(composite['days_since_drought_start'], composite['composite_mean'], 
                        label=f'Q{q}', color=variable_colours[v], linestyle=linestyles[q-1], linewidth=2)
        if i==0:
            ax.legend(loc='best', fontsize=13)
        title_size = 16 if v=='tdiff_aqua' else 18
        ax.set_title(f'$\\bf{{({alphabet[i]})}}$ {variable_legend_labels[v]}', fontsize=title_size, color=variable_colours[v])
        ax.tick_params(labelsize=16)
        ylims = ax.get_ylim()
        print(v, ylims)
        if v in fixed_ylims.keys():
            ax.set_ylim(fixed_ylims[v])
        ax.axhline(color='gray', linewidth=0.5)
        ax.axvline(color='gray', linewidth=0.5)
        ax.set_xlim([-120, 120])
        ax.set_xticks(np.arange(-120, 121, 60))
        ax.set_xlabel('days since\nflash drought onset', fontsize=18)
        ax.set_ylabel('standardised\nanomaly', fontsize=18)
        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)
    if len(axlist) > len(variables_to_plot):
        for ax in axlist[len(variables_to_plot):]:
            ax.set_axis_off()
    plt.suptitle(land_cover.strip('all_').replace('_', ' '), fontsize=18)
    plt.tight_layout(pad=2)
    fig.subplots_adjust(left=0.15, right=0.85, top=0.94, bottom=0.4, hspace=0.3)
    if smooth:
        plt.savefig(f'../figures/{land_cover}_smoothed{save_name}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'../figures/{land_cover}_smoothed{save_name}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(f'../figures/{land_cover}{save_name}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'../figures/{land_cover}{save_name}.pdf', dpi=600, bbox_inches='tight')
    if show:
        plt.show()


def compare_quartiles_by_variable_supp_fig(land_cover, show=True, smooth=False, save_name=None):
    variables_to_plot = ['tdiff_mw_18', 'lst_mw', 'SESR_GLEAM', 'SIF_JJ', 'wind_speed', 'vpd']
    num_rows = np.ceil(np.sqrt(len(variables_to_plot)))
    num_columns = np.ceil(float(len(variables_to_plot))/float(num_rows))
    linestyles = ['-', '--', ':', '-.']
    save_dir = '/prj/nceo/bethar/SUBDROUGHT/HESS_paper/ESA_CCI_land_cover_composites/stratified_by_max_shf/mask20pct'
    fig, axs = plt.subplots(int(num_columns), int(num_rows), sharex=True, figsize=(16, 10)) # for supp fig
    # plt.subplots_adjust(wspace=18)
    axlist = axs.flatten()
    alphabet = string.ascii_lowercase
    for i, v in enumerate(variables_to_plot):
        ax = axlist[i]
        for q in range(1,5):
            if v == 'tdiff_aqua' or v == 'lst_aqua':
                composite = pd.read_csv(f'{save_dir}/{v}_n-weighted_composite_{land_cover}_Q{q}.csv')
            else:
                composite = pd.read_csv(f'{save_dir}/{v}_composite_{land_cover}_Q{q}.csv')
            if smooth:
                window = 10
                ax.plot(composite['days_since_drought_start'], composite['composite_mean'].rolling(window=window, min_periods=1, center=True).mean(), 
                    label=f'Q{q}', color=variable_colours[v], linestyle=linestyles[q-1], linewidth=2)
            else:
                ax.plot(composite['days_since_drought_start'], composite['composite_mean'], 
                        label=f'Q{q}', color=variable_colours[v], linestyle=linestyles[q-1], linewidth=2)
        if i==0:
            ax.legend(loc='best', fontsize=13)
        title_size = 16 if v=='tdiff_aqua' else 18
        ax.set_title(f'$\\bf{{({alphabet[i]})}}$ {variable_legend_labels[v]}', fontsize=title_size, color=variable_colours[v])
        ylims = ax.get_ylim()
        print(v, ylims)
        ax.tick_params(labelsize=16)
        ax.axhline(color='gray', linewidth=0.5)
        ax.axvline(color='gray', linewidth=0.5)
        ax.set_xlim([-120, 120])
        ax.set_xticks(np.arange(-120, 121, 60))
        ax.set_xlabel('days since\nflash drought onset', fontsize=18)
        ax.set_ylabel('standardised\nanomaly', fontsize=18)
        ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)
    if len(axlist) > len(variables_to_plot):
        for ax in axlist[len(variables_to_plot):]:
            ax.set_axis_off()
    plt.suptitle(land_cover.strip('all_').replace('_', ' '), fontsize=18, y=1) # use y for supp fig
    plt.tight_layout(pad=2)
    fig.subplots_adjust(left=0.15, right=0.85, top=0.94, bottom=0.4, hspace=0.3)
    if smooth:
        plt.savefig(f'../figures/{land_cover}_smoothed{save_name}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'../figures/{land_cover}_smoothed{save_name}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(f'../figures/{land_cover}{save_name}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'../figures/{land_cover}{save_name}.pdf', dpi=600, bbox_inches='tight')
    if show:
        plt.show()

if __name__ == '__main__':
    compare_quartiles_by_variable_main_fig('all_cropland_rainfed', show=False, smooth=True, save_name='_composites_stratified_by_max_shf_mainfig_mask20pct_GLEAMv4_fixERA5SESR_shareaxes')
    compare_quartiles_by_variable_supp_fig('all_cropland_rainfed', show=False, smooth=True, save_name='_composites_stratified_by_max_shf_suppfig_mask20pct_GLEAMv4_fixERA5SESR_shareaxes')
