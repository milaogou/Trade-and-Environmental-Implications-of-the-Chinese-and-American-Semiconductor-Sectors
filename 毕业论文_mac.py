# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:56:12 2024
"/Users/bytedance/Documents/毕业论文/EXIO3"
"E:/毕业论文数据/EXIO3"

#https://pymrio.readthedocs.io/en/latest/notebooks/aggregation_examples.html
@author: milaogou
"""
import pandas as pd
import numpy as np
import time
from mpmath import mp
from multiprocessing import Pool
from scipy.io import loadmat
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import pymrio
import os
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter,PercentFormatter
from matplotlib.patches import Rectangle, Circle, RegularPolygon
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from adjustText import adjust_text
from textwrap import fill
import concurrent.futures
import ast
import re
import multiprocessing
# exio_path="E:/毕业论文数据/EXIO3"
exio_path="/Users/bytedance/Documents/毕业论文/EXIO3"
# root_path="E:/OneDrive/毕业论文"
root_path="/Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文"
# data_path="E:/OneDrive/毕业论文/Data"
data_path="/Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Data"
# graph_path="E:/OneDrive/毕业论文/Graph"
graph_path="/Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Graph"
#%%default
#"/Users/bytedance/Documents/毕业论文/EXIO3"
# =============================================================================
# exio3_folder = "E:/毕业论文数据/EXIO3"
# exio_downloadlog = pymrio.download_exiobase3(
#     storage_folder=exio3_folder, system="pxp", years=[2002,2007]
#     #list(range(2016, 2022, 1))
# )
# print(exio_downloadlog)
# years=list(range(2000, 2022, 1))
# =============================================================================
#%%read current year
year=2022
#"/Users/bytedance/Documents/毕业论文/EXIO3
# E:/毕业论文数据/EXIO3
exio3 = pymrio.parse_exiobase3(path=os.path.join(exio_path,f"IOT_{year}_pxp.zip"))
sectors=exio3.get_sectors()
regions=exio3.get_regions()
impacts=exio3.impacts.get_rows()
n_countries=49
n_products=200
n_Y_categories=5
ghg="GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
upstream_semi_products=[
    #Basic materials and raw materials:
    'Other non-ferrous metal ores and concentrates',
    'Secondary other non-ferrous metals for treatment, Re-processing of secondary other non-ferrous metals into new other non-ferrous metals',
    'Precious metals',
    'Secondary preciuos metals for treatment, Re-processing of secondary preciuos metals into new preciuos metals'
    ]
downstream_semi_products=[
    #Electronic equipment and instruments:
    "Electrical machinery and apparatus n.e.c. (31)",
    "Office machinery and computers (30)",
    "Radio, television and communication equipment and apparatus (32)",
    "Medical, precision and optical instruments, watches and clocks (33)"
]
secondary_products=[item for item in list(sectors) if 'secondary' in item.lower()]
non_secondary_products=[item for item in list(sectors) if 'secondary' not in item.lower()]
semi_products=upstream_semi_products+downstream_semi_products
order_dict=dir(exio3)
impacts_order_dict=dir(exio3.impacts)
# E:/OneDrive/毕业论文/Data
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Data
matched=pd.read_csv(os.path.join(data_path,"Country Info.csv"))
country_info=matched[['iso_3166_1_','iso3','name']]
country_info=country_info.set_index('iso_3166_1_')
country_info.loc["WA",'iso3']="RoW Asia and Pacific"
country_info.loc["WE",'iso3']="RoW Europe"
country_info.loc["WF",'iso3']="RoW Africa"
country_info.loc["WL",'iso3']="RoW America"
country_info.loc["WM",'iso3']="RoW Middle East"
#%%calc data
exio3.calc_all()
Y=exio3.Y
X=exio3.x
X_reg=X.groupby(level=0).sum()
A=exio3.A
L=exio3.L
Z=exio3.Z
Z_reg=Z.groupby(level=0).sum().sum(axis=1)
Y_reg=Y.groupby(level=0,axis=1).sum()
VA=exio3.impacts.get_row_data(row='Value Added')
GWP100=exio3.impacts.get_row_data(row=ghg)
GWP100_list=list(GWP100.keys())


#%%check logic
#S@L@Y=CBA,CBA!=CBA_reg?!
check1=VA['S'].T @ L @ Y.groupby(level=0,axis=1).sum()-VA['D_cba'].groupby(level=0).sum().reindex(index=regions).T
VA['D_cba'].groupby(level=0).sum().reindex(index=regions).T-VA['D_cba_reg'].T
GWP100['D_cba'].sum()-GWP100['D_cba_reg'].sum()
check1.values.mean()
#S*X=PBA=F
check2=VA['S'] * X.values-VA['D_pba']
sns.kdeplot(VA['D_pba']-VA['F'])
check2.mean()
#S@((L@Y)*export_mask)=imp
Y.columns.levels[0]
by_country = L @ Y_reg.reindex(columns=regions)
for i in range(n_countries):
    by_country.iloc[i*n_products:(i + 1) * n_products,i]=0
check3=VA['S'].T @ by_country - VA['D_imp'].groupby(level=0).sum().reindex(index=regions).values.flatten()
check3.mean(axis=1)
#仍然不会算出口账户
# =============================================================================
# check4=VA['S'] *(X.values-by_country.sum(axis=1).values.reshape(-1,1))-VA['D_exp']
# Z_trade=Z.copy()
# Y_trade=Y.copy()
# Z_local = pd.DataFrame(np.zeros_like(Z.values), columns=Z.columns, index=Z.index)
# Y_local = pd.DataFrame(np.zeros_like(Y.values), columns=Y.columns, index=Y.index)
# for country in regions:
#     Z_trade.loc[(country, slice(None)), (country, slice(None))] = 0
#     Z_local.loc[(country, slice(None)), (country, slice(None))] = Z.loc[(country, slice(None)), (country, slice(None))]
#     Y_trade.loc[(country, slice(None)), (country, slice(None))] = 0
#     Y_local.loc[(country, slice(None)), (country, slice(None))] = Y.loc[(country, slice(None)), (country, slice(None))]
# exp=VA['S'] * (Z_trade.groupby(level=0,axis=1).sum()+ (A @ Z_local).groupby(level=0,axis=1).sum()+Y_trade.groupby(level=0,axis=1).sum()).reindex(columns=regions).sum(axis=1).values.reshape(-1,1)
# local=VA['S'] * (Z_local.groupby(level=0,axis=1).sum()+Y_local.groupby(level=0,axis=1).sum()).reindex(columns=regions).sum(axis=1).values.reshape(-1,1)
# check4=exp-VA['D_exp']
# exp.sum()/VA['D_exp'].sum()
# local.sum()/VA['D_exp'].sum()
# exp.sum()/VA['D_pba'].sum()
# local.sum()/VA['D_pba'].sum()
# VA['D_exp'].sum()/VA['D_pba'].sum()
# check4.mean()
# check4.sum()
# (Z_trade.groupby(level=0,axis=1).sum().values.sum(axis=1).sum()+Y_trade.groupby(level=0,axis=1).sum().values.sum(axis=1).sum())/X.values.sum()
# (Z_local.groupby(level=0,axis=1).sum().values.sum(axis=1).sum()+Y_local.groupby(level=0,axis=1).sum().values.sum(axis=1).sum())/X.values.sum()
# =============================================================================
#
# =============================================================================
# #%%无贸易情景
# cba_x = L @ Y_reg.reindex(columns=regions)
# cba_ghg=pd.Series()
# for country in regions:
#     S=pd.concat([GWP100['S'].loc[(country, slice(None)),:]]*49,axis=0,ignore_index=True)
#     S.index=cba_x.index
#     cba_ghg[country]=S[ghg] @ cba_x.loc[:,country]
# carbon_gap=cba_ghg/ ((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions)))-1
# carbon_uplift=cba_ghg.sum()/((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions))).sum().sum()-1
# #%%无双边半导体贸易
# cba_x = L @ Y_reg.reindex(columns=regions)
# cba_ghg=pd.Series()
# for country in regions:
#     S=GWP100['S'].copy()
#     new_s=S.loc[(country, semi_products),:]
#     for other_country in set(regions)-set([country]): 
#         new_s.index=S.loc[(other_country, semi_products),:].index
#         S.loc[(other_country, semi_products),:]=new_s
#     cba_ghg[country]=S[ghg] @ cba_x.loc[:,country]
# carbon_gap_nosemitrade=cba_ghg/ ((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions)))-1
# carbon_nosemitrade_uplift=cba_ghg.sum()/((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions))).sum().sum()-1
# #%%无双边半导体贸易
# cba_x = L @ Y_reg.reindex(columns=regions)
# cba_ghg=pd.Series()
# for country in regions:
#     S=GWP100['S'].copy()
#     if country in ['CN','US']:
#         new_s=S.loc[(country, semi_products),:]
#         for other_country in set(['CN','US'])-set([country]): 
#             new_s.index=S.loc[(other_country, semi_products),:].index
#             S.loc[(other_country, semi_products),:]=new_s
#     cba_ghg[country]=S[ghg] @ cba_x.loc[:,country]
# carbon_gap_nobilsemitrade=cba_ghg/ ((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions)))-1
# carbon_nobilsemitrade_uplift=cba_ghg.sum()/((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions))).sum().sum()-1
# =============================================================================
#%%明细数据计算
# =============================================================================
# for year in range(2000,2023):
#     print(year)
#     #/Users/bytedance/Documents/毕业论文/EXIO3
#     # E:/毕业论文数据/EXIO3/
#     exio3 = pymrio.parse_exiobase3(path=f"/Users/bytedance/Documents/毕业论文/EXIO3/IOT_{year}_pxp.zip")
#     exio3.calc_all()
#     Y=exio3.Y
#     X=exio3.x
#     X_reg=X.groupby(level=0).sum()
#     A=exio3.A
#     L=exio3.L
#     Z=exio3.Z
#     Z_reg=Z.groupby(level=0).sum().sum(axis=1)
#     Y_reg=Y.groupby(level=0,axis=1).sum()
#     VA=exio3.impacts.get_row_data(row='Value Added')
#     GWP100=exio3.impacts.get_row_data(row=ghg)
#     # 初始化一个DataFrame来存储每个产品的碳排放增加
#     # carbon_uplift_by_sector = pd.DataFrame(index=['Global'] + list(regions))
#     # cba_x = L @ Y_reg.reindex(columns=regions)
#     # for sector in list(sectors)+[semi_products,upstream_semi_products,downstream_semi_products,list(sectors),secondary_products,non_secondary_products]:  # 假设sectors变量包含所有行业的列表
#     #     cba_ghg=pd.Series(dtype=float)
#     #     for country in regions:
#     #         S = GWP100['S'].copy()
#     #         new_s = S.loc[(country, sector), :]
#     #         for other_country in set(regions) - {country}: 
#     #             new_s.index = S.loc[(other_country, sector), :].index
#     #             S.loc[(other_country, sector), :] = new_s
#     #         cba_ghg[country]=S[ghg] @ cba_x.loc[:,country]
#         
#     #     # 计算每个国家以及全球的碳排放增加
#     #     carbon_gap_by_sector = cba_ghg/ ((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions)))-1
#     #     if sector in list(sectors):
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename(f"{sector}")
#     #     elif sector == semi_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Semiconductivity Related Products")
#     #     elif sector == upstream_semi_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Semiconductivity Upstream Products")
#     #     elif sector == downstream_semi_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Semiconductivity Downstream Products")
#     #     elif sector == list(sectors):
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("All Products")
#     #     elif sector == secondary_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Secondary Products")
#     #     elif sector == non_secondary_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Non-secondary  Products")
#     #     global_carbon_uplift = cba_ghg.sum()/((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions))).sum().sum()-1
#     #     carbon_gap_by_sector['Global']=global_carbon_uplift
#     #     carbon_uplift_by_sector = pd.concat([carbon_uplift_by_sector,carbon_gap_by_sector],axis=1,ignore_index=False)
#     # # 这里carbon_uplift_by_sector包含了每个sector贸易消失后的全球carbon uplift
#     # carbon_uplift_by_sector.to_csv(os.path.join(data_path,f"carbon_uplift_by_sector_{year}.csv"))
#     #无中美双边贸易
#     # 初始化一个DataFrame来存储每个产品的碳排放增加
#     # carbon_uplift_by_sector_bil = pd.DataFrame(index=['Global'] + list(regions))
#     # cba_x = L @ Y_reg.reindex(columns=regions)
#     # for sector in list(sectors)+[semi_products,upstream_semi_products,downstream_semi_products,list(sectors),secondary_products,non_secondary_products]:  # 假设sectors变量包含所有行业的列表
#     #     cba_ghg=pd.Series(dtype=float)
#     #     for country in regions:
#     #         S = GWP100['S'].copy()
#     #         if country in ['CN','US']:
#     #             for other_country in set(['CN','US']) - {country}: 
#     #                 new_s = S.loc[(other_country, sector), :]
#     #                 new_s.index = S.loc[(country, sector), :].index
#     #                 S.loc[(country, sector), :] = new_s
#     #         cba_ghg[country]=S[ghg] @ cba_x.loc[:,country]
#         
#     #     # 计算每个国家以及全球的碳排放增加
#     #     carbon_gap_by_sector = cba_ghg/ ((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions)))-1
#     #     if sector in list(sectors):
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename(f"{sector}")
#     #     elif sector == semi_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Semiconductivity Related Products")
#     #     elif sector == upstream_semi_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Semiconductivity Upstream Products")
#     #     elif sector == downstream_semi_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Semiconductivity Downstream Products")
#     #     elif sector == list(sectors):
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("All Products")
#     #     elif sector == secondary_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Secondary Products")
#     #     elif sector == non_secondary_products:
#     #         carbon_gap_by_sector = carbon_gap_by_sector.rename("Non-secondary Products")
#         
#     #     global_carbon_uplift = cba_ghg.sum()/((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions))).sum().sum()-1
#     #     carbon_gap_by_sector['Global']=global_carbon_uplift
#     #     carbon_uplift_by_sector_bil = pd.concat([carbon_uplift_by_sector_bil,carbon_gap_by_sector],axis=1,ignore_index=False)
#     # # 这里carbon_uplift_by_sector包含了每个sector贸易消失后的全球carbon uplift
#     # carbon_uplift_by_sector_bil.to_csv(os.path.join(data_path,f"bil_carbon_uplift_by_sector_{year}.csv"))
#     
#     carbon_uplift_by_sector_bil_gamed = pd.DataFrame(index=['Global'] + list(regions))
#     cba_x = L @ Y_reg.reindex(columns=regions)
#     for sector in list(sectors)+[semi_products,upstream_semi_products,downstream_semi_products,list(sectors),secondary_products,non_secondary_products]:  # 假设sectors变量包含所有行业的列表
#         cba_ghg=pd.Series(dtype=float)
#         for country in regions:
#             S = GWP100['S'].copy()
#             if country in ['CN','US']:
#                 for other_country in set(['CN','US']) - {country}: 
#                     new_s=GWP100['F'].loc[(list(set(regions) - {other_country}),sector),:].groupby(level=1).sum()[ghg]/(X.loc[(list(set(regions) - {other_country}),sector),:].groupby(level=1).sum()['indout']+1e-9)
#                     new_s.index = S.loc[(country, sector), :].index
#                     S.loc[(country, sector), :] = new_s
#             cba_ghg[country]=S[ghg] @ cba_x.loc[:,country]
# # 计算每个国家以及全球的碳排放增加
#         carbon_gap_by_sector = cba_ghg/ ((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions)))-1
#         if sector in list(sectors):
#             carbon_gap_by_sector = carbon_gap_by_sector.rename(f"{sector}")
#         elif sector == semi_products:
#             carbon_gap_by_sector = carbon_gap_by_sector.rename("Semiconductivity Related Products")
#         elif sector == upstream_semi_products:
#             carbon_gap_by_sector = carbon_gap_by_sector.rename("Semiconductivity Upstream Products")
#         elif sector == downstream_semi_products:
#             carbon_gap_by_sector = carbon_gap_by_sector.rename("Semiconductivity Downstream Products")
#         elif sector == list(sectors):
#             carbon_gap_by_sector = carbon_gap_by_sector.rename("All Products")
#         elif sector == secondary_products:
#             carbon_gap_by_sector = carbon_gap_by_sector.rename("Secondary Products")
#         elif sector == non_secondary_products:
#             carbon_gap_by_sector = carbon_gap_by_sector.rename("Non-secondary Products")
#         
#         global_carbon_uplift = cba_ghg.sum()/((GWP100['S'][ghg] @L @ Y_reg.reindex(columns=regions))).sum().sum()-1
#         carbon_gap_by_sector['Global']=global_carbon_uplift
#         carbon_uplift_by_sector_bil_gamed = pd.concat([carbon_uplift_by_sector_bil_gamed,carbon_gap_by_sector],axis=1,ignore_index=False)
#     # 这里carbon_uplift_by_sector包含了每个sector贸易消失后的全球carbon uplift
#     carbon_uplift_by_sector_bil_gamed.to_csv(os.path.join(data_path,f"bil_gamed_carbon_uplift_by_sector_{year}.csv"))
# =============================================================================
#%%Carbon Uplift Trends If No Trade(2000-2022).png
def to_percent(y, position):
    # 将y值转换为百分比字符串。你可以根据需要调整小数点后的位数。
    s = "{:.2f}%".format(100 * y)

    # 返回格式化的字符串
    return s
# E:/OneDrive/毕业论文/Data

path_template = os.path.join(data_path,"carbon_uplift_by_sector_{year}.csv")
all_data = pd.DataFrame()
countries = [ 'CN','US']#'Global',
# sectors_to_plot=list(set(semi_products)-{'Secondary other non-ferrous metals for treatment, Re-processing of secondary other non-ferrous metals into new other non-ferrous metals',
#                                     'Secondary preciuos metals for treatment, Re-processing of secondary preciuos metals into new preciuos metals'})
sectors_to_plot=["Semiconductivity Related Products","Semiconductivity Upstream Products",
                 "Semiconductivity Downstream Products"]#,,"All Products"
country_markers = {
    'US': ('#00487C', 'o'),  # 颜色和标记样式
    'CN': ('#FC440F', 's'),
    'Global': ('green', '^')  # 注意: matplotlib可能不直接支持"#"作为marker，需要选择matplotlib支持的marker
}
sector_linestyles = {
    sectors_to_plot[0]: 'solid',  # 实线
    sectors_to_plot[1]: '-.',  # 虚线
    sectors_to_plot[2]: ':',  # 点线
    # sectors_to_plot[3]: '-.'
}
# 加载2000年至2022年的数据
for year in range(2000, 2023):
    path = path_template.format(year=year)
    data=pd.read_csv(path,index_col=0)
    data['year']=year
    all_data = pd.concat([all_data, data],axis=0,ignore_index=False)


filtered_data = all_data.loc[countries,sectors_to_plot+['year']]
formatter = FuncFormatter(to_percent)
# 绘制趋势图
plt.figure(figsize=(15, 12))
plt.rcParams.update({'font.size': 14})  # 可以调整为你想要的大小
for sector in sectors_to_plot:
    for country in countries:
        color, marker = country_markers[country]
        sector_country_data = filtered_data.loc[country,:]
        if not sector_country_data.empty:
            plt.plot(sector_country_data['year'], sector_country_data[sector],
                     label=f"{country} - {sector}",  alpha=0.9,
                       linestyle=sector_linestyles[sector],
                      color=color,marker=marker,
                     linewidth=3)
plt.gca().yaxis.set_major_formatter(formatter)
# plt.title('Carbon Uplift Trends If No Trade(2000-2022)',weight='bold')
plt.xlabel('Year')
plt.ylabel('Carbon Uplift (%)')
plt.legend()
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.savefig(os.path.join(graph_path,"Carbon Uplift Trends If No Trade(2000-2022).png"), dpi=300)
plt.show()
#%%Carbon Uplift Trends If No Sino-US Trade(2000-2022).png
# E:/OneDrive/毕业论文/Data
path_template = "/Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Data/bil_carbon_uplift_by_sector_{year}.csv"
all_data = pd.DataFrame()
def to_percent(y, position):
    # 将y值转换为百分比字符串。你可以根据需要调整小数点后的位数。
    s = "{:.2f}%".format(100 * y)

    # 返回格式化的字符串
    return s
countries = [ 'CN','US']#'Global',
# sectors_to_plot=list(set(semi_products)-{'Secondary other non-ferrous metals for treatment, Re-processing of secondary other non-ferrous metals into new other non-ferrous metals',
#                                     'Secondary preciuos metals for treatment, Re-processing of secondary preciuos metals into new preciuos metals'})
sectors_to_plot=["Semiconductivity Related Products","Semiconductivity Upstream Products",
                 "Semiconductivity Downstream Products",]#"All Products"
country_markers = {
    'US': ('#00487C', 'o'),  # 颜色和标记样式
    'CN': ('#FC440F', 's'),
    'Global': ('green', '^')  # 注意: matplotlib可能不直接支持"#"作为marker，需要选择matplotlib支持的marker
}
sector_linestyles = {
    sectors_to_plot[0]: 'solid',  # 实线
    sectors_to_plot[1]: '-.',  # 虚线
    sectors_to_plot[2]: ':',  # 点线
    # sectors_to_plot[3]: '-.'
}
# 加载2000年至2022年的数据
for year in range(2000, 2023):
    path = path_template.format(year=year)
    data=pd.read_csv(path,index_col=0)
    data['year']=year
    all_data = pd.concat([all_data, data],axis=0,ignore_index=False)
filtered_data = all_data.loc[countries,sectors_to_plot+['year']]
formatter = FuncFormatter(to_percent)
# 绘制趋势图
plt.figure(figsize=(15, 12))
plt.rcParams.update({'font.size': 14})  # 可以调整为你想要的大小
for sector in sectors_to_plot:
    for country in countries:
        color, marker = country_markers[country]
        sector_country_data = filtered_data.loc[country,:]
        if not sector_country_data.empty:
            plt.plot(sector_country_data['year'], sector_country_data[sector],
                     label=f"{country} - {sector}",  alpha=0.9,
                       linestyle=sector_linestyles[sector],
                      color=color,marker=marker,
                     linewidth=3)
plt.gca().yaxis.set_major_formatter(formatter)
# plt.title('Carbon Uplift Trends If No Sino-US Trade(2000-2022)',weight='bold')
plt.xlabel('Year')
plt.ylabel('Carbon Uplift (%)')
plt.legend()
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.savefig(os.path.join(graph_path,"Carbon Uplift Trends If No Sino-US Trade(2000-2022).png"), dpi=300)
plt.show()
#%%Carbon Uplift Trends If No Sino-US with Gaming Trade(2000-2022).png
# E:/OneDrive/毕业论文/Data
path_template = "/Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Data/bil_gamed_carbon_uplift_by_sector_{year}.csv"
all_data = pd.DataFrame()
def to_percent(y, position):
    # 将y值转换为百分比字符串。你可以根据需要调整小数点后的位数。
    s = "{:.2f}%".format(100 * y)

    # 返回格式化的字符串
    return s
countries = [ 'CN','US']#'Global',
# sectors_to_plot=list(set(semi_products)-{'Secondary other non-ferrous metals for treatment, Re-processing of secondary other non-ferrous metals into new other non-ferrous metals',
#                                     'Secondary preciuos metals for treatment, Re-processing of secondary preciuos metals into new preciuos metals'})
sectors_to_plot=["Semiconductivity Related Products","Semiconductivity Upstream Products",
                 "Semiconductivity Downstream Products",]#"All Products"
country_markers = {
    'US': ('#00487C', 'o'),  # 颜色和标记样式
    'CN': ('#FC440F', 's'),
    'Global': ('green', '^')  # 注意: matplotlib可能不直接支持"#"作为marker，需要选择matplotlib支持的marker
}
sector_linestyles = {
    sectors_to_plot[0]: 'solid',  # 实线
    sectors_to_plot[1]: '-.',  # 虚线
    sectors_to_plot[2]: ':',  # 点线
    # sectors_to_plot[3]: '-.'
}
# 加载2000年至2022年的数据
for year in range(2000, 2023):
    path = path_template.format(year=year)
    data=pd.read_csv(path,index_col=0)
    data['year']=year
    all_data = pd.concat([all_data, data],axis=0,ignore_index=False)
filtered_data = all_data.loc[countries,sectors_to_plot+['year']]
formatter = FuncFormatter(to_percent)
# 绘制趋势图
plt.figure(figsize=(15, 12))
plt.rcParams.update({'font.size': 14})  # 可以调整为你想要的大小
for sector in sectors_to_plot:
    for country in countries:
        color, marker = country_markers[country]
        sector_country_data = filtered_data.loc[country,:]
        if not sector_country_data.empty:
            plt.plot(sector_country_data['year'], sector_country_data[sector],
                     label=f"{country} - {sector}",  alpha=0.9,
                       linestyle=sector_linestyles[sector],
                      color=color,marker=marker,
                     linewidth=3)
plt.gca().yaxis.set_major_formatter(formatter)
# plt.title('Carbon Uplift Trends If No Sino-US with Gaming Trade(2000-2022)',weight='bold')
plt.xlabel('Year')
plt.ylabel('Carbon Uplift (%)')
plt.legend()
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.savefig(os.path.join(graph_path,"Carbon Uplift Trends If No Sino-US with Gaming Trade(2000-2022).png"), dpi=300)
plt.show()
#%%Value of Carbon Uplift If No Trade Heatmap in 2022.png
# E:/OneDrive/毕业论文/Data
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Data
country_dict = matched.set_index("iso_3166_1_")["name"].to_dict()
country_dict["WA"]="RoW Asia and Pacific"
country_dict["WE"]="RoW Europe"
country_dict["WF"]="RoW Africa"
country_dict["WL"]="RoW America"
country_dict["WM"]="RoW Middle East"
path_template = os.path.join(data_path,"carbon_uplift_by_sector_{year}.csv")
all_data = pd.DataFrame()
for year in range(2000, 2023):
    path = path_template.format(year=year)
    data=pd.read_csv(path,index_col=0)
    data['year']=year
    all_data = pd.concat([all_data, data],axis=0,ignore_index=False)



# 重置索引，方便绘图
this_year_carbon_uplift=all_data[all_data['year'].squeeze()==2022]
this_year_carbon_uplift = this_year_carbon_uplift.drop('year', axis=1)

# ranked_df = this_year_carbon_stressor.rank(method='min', axis=1).astype(int)
# ranked_df=ranked_df.shape[1]-ranked_df+1
# ranked_df.columns = [country_dict.get(col, col) for col in ranked_df.columns]
ranked_df = this_year_carbon_uplift.T
ranked_df.columns = [country_dict.get(col, col) for col in ranked_df.columns]
no_trade_carbon_uplift_2022=ranked_df
fig, ax = plt.subplots(figsize=(35, 50))  # 指定图像大小
plt.subplots_adjust(left=0, right=1)  # 减少左右边距
colors = sns.color_palette("viridis", n_colors=256-1)#["red"] + 

# 创建一个自定义的颜色映射
colors = ['#1C0118','#42113C','#00487C', '#0F5257',"#6BD425","yellow" ,'#FC440F']  # 你可以自定义这些颜色以符合你的需求

# 创建自定义的颜色映射
custom_cmap = LinearSegmentedColormap.from_list("custom_blue_white_red", colors)

# 创建Normalize对象，设置色带的范围
norm = Normalize(vmin=-0.05, vmax=0.05)
# 绘制热力图，确保传入行和列的标签
heatmap= sns.heatmap(ranked_df, annot=False, fmt="d",cmap=custom_cmap,norm=norm,
            # cmap='viridis_r',
            xticklabels=ranked_df.columns,
            yticklabels=list(ranked_df.index), ax=ax,
            cbar_kws={"fraction": 0.03,"aspect":100})
cbar = heatmap.collections[0].colorbar

# 调整色带标签的位置
cbar.set_label('Carbon Uplift (%)', labelpad=-80, y=0.5, rotation=90)
cbar.ax.set_position([0.93, 0.15, 0.02, 0.7])  # 手动设置色带位置
marks = {"China": "+", "United States of America": "x", "Global": "-"}  # 确保列名与你的DataFrame匹配
for col, mark in marks.items():
    if col in ranked_df.columns:
        for row in range(ranked_df.shape[0]):
            ax.text(ranked_df.columns.get_loc(col) + 0.5, row + 0.5, mark,
                    horizontalalignment='center', verticalalignment='center', fontsize=8,
                    alpha=0.9)

# industry_boxes = {
#     "Semiconductivity Related Products": "solid",
#     "Semiconductivity Upstream Products": "dashed",
#     "Semiconductivity Downstream Products": "dotted"
# }
# for row_label, linestyle in industry_boxes.items():
#     if row_label in ranked_df.index:
#         for col in range(len(ranked_df.columns)):
#             rect = Rectangle((col, ranked_df.index.get_loc(row_label)), 1, 1, fill=False,
#                              edgecolor="yellow", linestyle=linestyle, linewidth=2)
#             ax.add_patch(rect)
industries_shapes = {
    "Semiconductivity Related Products": "square",
    "Semiconductivity Upstream Products": "circle",
    "Semiconductivity Downstream Products": "hexagon"
}
for industry, shape in industries_shapes.items():
    if industry in ranked_df.index:
        row = ranked_df.index.get_loc(industry)
        for col in range(ranked_df.shape[1]):
            if shape == "square":
                rect = Rectangle((col+0.15, row+0.15), 0.6, 0.6, fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(rect)
            elif shape == "circle":
                circ = Circle((col + 0.5, row + 0.5), 0.35, fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(circ)
            elif shape == "hexagon":
                hexagon = RegularPolygon((col + 0.5, row + 0.5), numVertices=6, radius=0.35, 
                                         orientation=np.radians(30), fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(hexagon)
def to_percentage(x, pos):
    return '{:.0f}%'.format(x * 100)
cbar = ax.collections[0].colorbar
# 设置色带的格式化器为百分比
cbar.formatter = FuncFormatter(to_percentage)

# 更新色带刻度
cbar.update_ticks()
# 将x轴标签移动到顶部
ax.xaxis.tick_top()  
ax.xaxis.set_label_position('top')  
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# 设置标题
# plt.title("Value of Carbon Uplift If No Trade Heatmap in 2022",weight='bold')
plt.rcParams.update({'font.size': 14})
plt.tight_layout()
# E:/OneDrive/毕业论文/Graph
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Graph
plt.savefig(os.path.join(graph_path,"Value of Carbon Uplift If No Trade Heatmap in 2022.png"), dpi=300)
# 显示图像
plt.show()

this_year_carbon_uplift.loc[['CN', 'US'], semi_products+sectors_to_plot].to_csv(os.path.join(data_path,"Carbon Uplift If No Trade in 2022.csv"))
selected_data=this_year_carbon_uplift.loc[['CN', 'US'], semi_products+sectors_to_plot]
#%%Value of Carbon Uplift If No Sino-US Trade Heatmap in 2022.png
# E:/OneDrive/毕业论文/Data/
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Data
country_dict = matched.set_index("iso_3166_1_")["name"].to_dict()
country_dict["WA"]="RoW Asia and Pacific"
country_dict["WE"]="RoW Europe"
country_dict["WF"]="RoW Africa"
country_dict["WL"]="RoW America"
country_dict["WM"]="RoW Middle East"
path_template = os.path.join(data_path,"bil_carbon_uplift_by_sector_{year}.csv")
all_data = pd.DataFrame()
for year in range(2000, 2023):
    path = path_template.format(year=year)
    data=pd.read_csv(path,index_col=0)
    data['year']=year
    all_data = pd.concat([all_data, data],axis=0,ignore_index=False)



# 重置索引，方便绘图
this_year_carbon_uplift=all_data[all_data['year'].squeeze()==2022]
this_year_carbon_uplift = this_year_carbon_uplift.drop('year', axis=1)

# ranked_df = this_year_carbon_stressor.rank(method='min', axis=1).astype(int)
# ranked_df=ranked_df.shape[1]-ranked_df+1
# ranked_df.columns = [country_dict.get(col, col) for col in ranked_df.columns]
ranked_df = this_year_carbon_uplift.T
ranked_df.columns = [country_dict.get(col, col) for col in ranked_df.columns]
no_bil_trade_carbon_uplift_2022=ranked_df
fig, ax = plt.subplots(figsize=(35, 50))  # 指定图像大小
colors = sns.color_palette("viridis", n_colors=256-1)#["red"] + 

# 创建一个自定义的颜色映射
colors = ['#1C0118','#42113C','#00487C', '#0F5257',"#6BD425","yellow" ,'#FC440F']  # 你可以自定义这些颜色以符合你的需求

# 创建自定义的颜色映射
custom_cmap = LinearSegmentedColormap.from_list("custom_blue_white_red", colors)

# 创建Normalize对象，设置色带的范围
norm = Normalize(vmin=-0.005, vmax=0.005)
# 绘制热力图，确保传入行和列的标签
heatmap=sns.heatmap(ranked_df, annot=False, fmt="d",cmap=custom_cmap,norm=norm,
            # cmap='viridis_r',
            xticklabels=ranked_df.columns,
            yticklabels=list(ranked_df.index), ax=ax,
            cbar_kws={"fraction": 0.03,"aspect":100,"label":"Carbon Uplift (%)"})
cbar = heatmap.collections[0].colorbar

# 调整色带标签的位置
cbar.set_label('Carbon Uplift (%)', labelpad=-100, y=0.5, rotation=90)
marks = {"China": "+", "United States of America": "x", "Global": "-"}  # 确保列名与你的DataFrame匹配
for col, mark in marks.items():
    if col in ranked_df.columns:
        for row in range(ranked_df.shape[0]):
            ax.text(ranked_df.columns.get_loc(col) + 0.5, row + 0.5, mark,
                    horizontalalignment='center', verticalalignment='center', fontsize=8,
                    alpha=0.9)

# industry_boxes = {
#     "Semiconductivity Related Products": "solid",
#     "Semiconductivity Upstream Products": "dashed",
#     "Semiconductivity Downstream Products": "dotted"
# }
# for row_label, linestyle in industry_boxes.items():
#     if row_label in ranked_df.index:
#         for col in range(len(ranked_df.columns)):
#             rect = Rectangle((col, ranked_df.index.get_loc(row_label)), 1, 1, fill=False,
#                              edgecolor="yellow", linestyle=linestyle, linewidth=2)
#             ax.add_patch(rect)
industries_shapes = {
    "Semiconductivity Related Products": "square",
    "Semiconductivity Upstream Products": "circle",
    "Semiconductivity Downstream Products": "hexagon"
}
for industry, shape in industries_shapes.items():
    if industry in ranked_df.index:
        row = ranked_df.index.get_loc(industry)
        for col in range(ranked_df.shape[1]):
            if shape == "square":
                rect = Rectangle((col+0.15, row+0.15), 0.6, 0.6, fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(rect)
            elif shape == "circle":
                circ = Circle((col + 0.5, row + 0.5), 0.35, fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(circ)
            elif shape == "hexagon":
                hexagon = RegularPolygon((col + 0.5, row + 0.5), numVertices=6, radius=0.35, 
                                         orientation=np.radians(30), fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(hexagon)
def to_percentage(x, pos):
    return '{:.2f}%'.format(x * 100)
cbar = ax.collections[0].colorbar
# 设置色带的格式化器为百分比
cbar.formatter = FuncFormatter(to_percentage)

# 更新色带刻度
cbar.update_ticks()
# 将x轴标签移动到顶部
ax.xaxis.tick_top()  
ax.xaxis.set_label_position('top')  
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# 设置标题
# plt.title("Value of Carbon Uplift If No Sino-US Trade Heatmap in 2022",weight='bold')
plt.rcParams.update({'font.size': 14})
plt.tight_layout()
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Graph
# E:/OneDrive/毕业论文/Graph
plt.savefig(os.path.join(graph_path,"Value of Carbon Uplift If No Sino-US Trade Heatmap in 2022.png"), dpi=300)
# 显示图像
plt.show()
selected_data=this_year_carbon_uplift.loc[['CN', 'US'], semi_products+sectors_to_plot]
#%%Value of Carbon Uplift If No Sino-US with Gaming Trade Heatmap in 2022.png
# E:/OneDrive/毕业论文/Data/
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Data
country_dict = matched.set_index("iso_3166_1_")["name"].to_dict()
country_dict["WA"]="RoW Asia and Pacific"
country_dict["WE"]="RoW Europe"
country_dict["WF"]="RoW Africa"
country_dict["WL"]="RoW America"
country_dict["WM"]="RoW Middle East"
path_template = os.path.join(data_path,"bil_gamed_carbon_uplift_by_sector_{year}.csv")
all_data = pd.DataFrame()
for year in range(2000, 2023):
    path = path_template.format(year=year)
    data=pd.read_csv(path,index_col=0)
    data['year']=year
    all_data = pd.concat([all_data, data],axis=0,ignore_index=False)



# 重置索引，方便绘图
this_year_carbon_uplift=all_data[all_data['year'].squeeze()==2022]
this_year_carbon_uplift = this_year_carbon_uplift.drop('year', axis=1)

# ranked_df = this_year_carbon_stressor.rank(method='min', axis=1).astype(int)
# ranked_df=ranked_df.shape[1]-ranked_df+1
# ranked_df.columns = [country_dict.get(col, col) for col in ranked_df.columns]
ranked_df = this_year_carbon_uplift.T
ranked_df.columns = [country_dict.get(col, col) for col in ranked_df.columns]
no_bil_trade_gamed_carbon_uplift_2022=ranked_df
fig, ax = plt.subplots(figsize=(35, 50))  # 指定图像大小
colors = sns.color_palette("viridis", n_colors=256-1)#["red"] + 

# 创建一个自定义的颜色映射
colors = ['#1C0118','#42113C','#00487C', '#0F5257',"#6BD425","yellow" ,'#FC440F']  # 你可以自定义这些颜色以符合你的需求

# 创建自定义的颜色映射
custom_cmap = LinearSegmentedColormap.from_list("custom_blue_white_red", colors)

# 创建Normalize对象，设置色带的范围
norm = Normalize(vmin=-0.005, vmax=0.005)
# 绘制热力图，确保传入行和列的标签
heatmap=sns.heatmap(ranked_df, annot=False, fmt="d",cmap=custom_cmap,norm=norm,
            # cmap='viridis_r',
            xticklabels=ranked_df.columns,
            yticklabels=list(ranked_df.index), ax=ax,
            cbar_kws={"fraction": 0.03,"aspect":100,"label":"Carbon Uplift (%)"})
cbar = heatmap.collections[0].colorbar

# 调整色带标签的位置
cbar.set_label('Carbon Uplift (%)', labelpad=-100, y=0.5, rotation=90)
marks = {"China": "+", "United States of America": "x", "Global": "-"}  # 确保列名与你的DataFrame匹配
for col, mark in marks.items():
    if col in ranked_df.columns:
        for row in range(ranked_df.shape[0]):
            ax.text(ranked_df.columns.get_loc(col) + 0.5, row + 0.5, mark,
                    horizontalalignment='center', verticalalignment='center', fontsize=8,
                    alpha=0.9)

# industry_boxes = {
#     "Semiconductivity Related Products": "solid",
#     "Semiconductivity Upstream Products": "dashed",
#     "Semiconductivity Downstream Products": "dotted"
# }
# for row_label, linestyle in industry_boxes.items():
#     if row_label in ranked_df.index:
#         for col in range(len(ranked_df.columns)):
#             rect = Rectangle((col, ranked_df.index.get_loc(row_label)), 1, 1, fill=False,
#                              edgecolor="yellow", linestyle=linestyle, linewidth=2)
#             ax.add_patch(rect)
industries_shapes = {
    "Semiconductivity Related Products": "square",
    "Semiconductivity Upstream Products": "circle",
    "Semiconductivity Downstream Products": "hexagon"
}
for industry, shape in industries_shapes.items():
    if industry in ranked_df.index:
        row = ranked_df.index.get_loc(industry)
        for col in range(ranked_df.shape[1]):
            if shape == "square":
                rect = Rectangle((col+0.15, row+0.15), 0.6, 0.6, fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(rect)
            elif shape == "circle":
                circ = Circle((col + 0.5, row + 0.5), 0.35, fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(circ)
            elif shape == "hexagon":
                hexagon = RegularPolygon((col + 0.5, row + 0.5), numVertices=6, radius=0.35, 
                                         orientation=np.radians(30), fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(hexagon)
def to_percentage(x, pos):
    return '{:.2f}%'.format(x * 100)
cbar = ax.collections[0].colorbar
# 设置色带的格式化器为百分比
cbar.formatter = FuncFormatter(to_percentage)

# 更新色带刻度
cbar.update_ticks()
# 将x轴标签移动到顶部
ax.xaxis.tick_top()  
ax.xaxis.set_label_position('top')  
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# 设置标题
# plt.title("Value of Carbon Uplift If No Sino-US with Gaming Trade Heatmap in 2022",weight='bold')
plt.rcParams.update({'font.size': 14})
plt.tight_layout()
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Graph
# E:/OneDrive/毕业论文/Graph
plt.savefig(os.path.join(graph_path,"Value of Carbon Uplift If No Sino-US with Gaming Trade Heatmap in 2022.png"), dpi=300)
# 显示图像
plt.show()
selected_data=this_year_carbon_uplift.loc[['CN', 'US','Global'], semi_products+sectors_to_plot]
#%%order generate
country_order=ranked_df.columns
sector_order=list(ranked_df.index)
#%%multi processing trial
# sns.kdeplot(GWP100['S'][ghg]*X['indout']-GWP100['F'][ghg])
# del exio3
# E:/OneDrive/毕业论文/Data
# =============================================================================
# def aggregate_data(original_data,vertical,aggregate_list,name):
#     aggregated_stressor=original_data.copy()
#     aggregated_stressor=aggregated_stressor.set_index(['region','sector'])
#     if vertical == 'sector':
#         aggregated_stressor=aggregated_stressor.loc[(slice(None),aggregate_list),:].groupby(level=0).sum()
#         aggregated_stressor[vertical] = name
#         aggregated_stressor.set_index(vertical, append=True, inplace=True)
#     else:
#         print("Not Valid vertical!")
#         return
#     aggregated_stressor=aggregated_stressor.swaplevel()
#     aggregated_stressor.reset_index(drop=False,inplace=True)
#     return aggregated_stressor
# for year in range(2000,2023):
# #     
# # def process_year(year):
#     #"/Users/bytedance/Documents/毕业论文/EXIO3
#     print(year, flush=True)
#     exio3 = pymrio.parse_exiobase3(path=f"E:/毕业论文数据/EXIO3/IOT_{year}_pxp.zip")
#     exio3.calc_all()
#     Y=exio3.Y
#     X=exio3.x
#     X_reg=X.groupby(level=0).sum()
#     A=exio3.A
#     L=exio3.L
#     Z=exio3.Z
#     Z_reg=Z.groupby(level=0).sum().sum(axis=1)
#     Y_reg=Y.groupby(level=0,axis=1).sum()
#     GWP100=exio3.impacts.get_row_data(row=ghg)
#     basic_carbon_data=pd.concat([GWP100['F'],X],axis=1,ignore_index=False).reset_index(drop=False)
#     upstream_carbon_data=aggregate_data(basic_carbon_data,'sector',upstream_semi_products,'Semiconductivity Upstream Products')
#     downstream_carbon_data=aggregate_data(basic_carbon_data,'sector',downstream_semi_products,"Semiconductivity Downstream Products")
#     secondary_carbon_data=aggregate_data(basic_carbon_data,'sector',secondary_products,"Secondary Products")
#     non_secondary_carbon_data=aggregate_data(basic_carbon_data,'sector',non_secondary_products,"Non Secondary Products")
#     semi_carbon_data=aggregate_data(basic_carbon_data,'sector',semi_products,"Semiconductivity Related Products")
#     all_carbon_data=aggregate_data(basic_carbon_data,'sector',list(sectors),"All Products")
#     sector_aggregated_carbon_data=pd.concat([basic_carbon_data,upstream_carbon_data,downstream_carbon_data,
#                                              secondary_carbon_data,non_secondary_carbon_data,semi_carbon_data,
#                                              all_carbon_data],ignore_index=True)
#     sector_aggregated_carbon_data=sector_aggregated_carbon_data.merge(matched,left_on='region',right_on='iso_3166_1_',how='left')
#     sector_aggregated_carbon_data=sector_aggregated_carbon_data.set_index(['continent','Region','region','iso3', 'name', 'sector'])
#     sector_carbon_stressor=sector_aggregated_carbon_data[[ghg,'indout']].groupby(level=[2,-1]).sum()
#     sector_carbon_stressor=sector_carbon_stressor[ghg]/(sector_carbon_stressor['indout']+1e-9)
#     Region_carbon_stressor=sector_aggregated_carbon_data[[ghg,'indout']].groupby(level=[1,-1]).sum()
#     Region_carbon_stressor=Region_carbon_stressor[ghg]/(Region_carbon_stressor['indout']+1e-9)
#     continent_carbon_stressor=sector_aggregated_carbon_data[[ghg,'indout']].groupby(level=[0,-1]).sum()
#     continent_carbon_stressor=continent_carbon_stressor[ghg]/(continent_carbon_stressor['indout']+1e-9)
#     Global_carbon_stressor=sector_aggregated_carbon_data[[ghg,'indout']].groupby(level=[-1]).sum()
#     Global_carbon_stressor=Global_carbon_stressor[ghg]/(Global_carbon_stressor['indout']+1e-9)
#     Global_carbon_stressor=Global_carbon_stressor.to_frame()
#     Global_carbon_stressor['region']='Global'
#     Global_carbon_stressor=Global_carbon_stressor.set_index('region',append=True)
#     Global_carbon_stressor=Global_carbon_stressor.swaplevel()
#     Carbon_stressor=pd.concat([sector_carbon_stressor,Global_carbon_stressor[0],continent_carbon_stressor,Region_carbon_stressor],ignore_index=False).rename('Carbon Stressor')
#     Carbon_stressor=Carbon_stressor.reset_index()
#     Carbon_stressor.to_csv(f"E:/OneDrive/毕业论文/Data/Carbon_stressor_{year}.csv")
# =============================================================================
# if __name__ == '__main__':  # This check is necessary for multiprocessing
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#         pool.map(process_year, list(range(2000, 2023)))
#%%Carbon Stressor Trends (2000-2022).png
# def parse_tuple(s):
#     # 使用正则表达式匹配括号内的内容
#     match = re.match(r"\('([^']*)', '([^']*)'\)", s)
#     if match:
#         return match.groups()
#     else:
#         return None, None  # 如果没有匹配到，返回两个None
# E:/OneDrive/毕业论文/Data

path_template = os.path.join(data_path,"Carbon_stressor_{year}.csv")
all_data = pd.DataFrame()
countries = ['CN', 'US', 'Global']
# sectors_to_plot=list(set(semi_products)-{'Secondary other non-ferrous metals for treatment, Re-processing of secondary other non-ferrous metals into new other non-ferrous metals',
                                    # 'Secondary preciuos metals for treatment, Re-processing of secondary preciuos metals into new preciuos metals'})
sectors_to_plot=["Semiconductivity Related Products","Semiconductivity Upstream Products",
                 "Semiconductivity Downstream Products","All Products"]
country_markers = {
    'US': ('#00487C', 'o'),  # 颜色和标记样式
    'CN': ('#FC440F', 's'),
    'Global': ("green", '^')  # 注意: matplotlib可能不直接支持"#"作为marker，需要选择matplotlib支持的marker
}
sector_linestyles = {
    sectors_to_plot[0]: '-',  # 实线
    sectors_to_plot[1]: '-.',  # 虚线
    sectors_to_plot[2]: ':',  # 点线
    sectors_to_plot[3]: '--'
}
# 加载2000年至2022年的数据
for year in range(2000, 2023):
    path = path_template.format(year=year)
    data=pd.read_csv(path,skiprows=1,header=None,index_col=0)
    data['year']=year
    all_data = pd.concat([all_data, data],axis=0,ignore_index=True)
# 拆分列
# all_data['region'], all_data['sector'] = zip(*all_data[0].apply(parse_tuple))

# all_data = all_data.drop(0, axis=1)

# 重置索引，方便绘图
all_data.columns=[['region','sector','Carbon Stressor','year']]
# 筛选指定的国家
filtered_data = all_data[(all_data['region'].squeeze().isin(countries))&(all_data['sector'].squeeze().isin(sectors_to_plot))]

# 绘制趋势图
plt.figure(figsize=(15, 12))
plt.rcParams.update({'font.size': 14})  # 可以调整为你想要的大小
for sector in sectors_to_plot:
    for country in countries:
        color, marker = country_markers[country]
        sector_country_data = filtered_data[(filtered_data['region'].squeeze() == country) & (filtered_data['sector'].squeeze() == sector)]
        if not sector_country_data.empty:
            plt.plot(sector_country_data['year'], sector_country_data['Carbon Stressor'],
                     label=f"{country} - {sector}",  alpha=0.8,
                     linestyle=sector_linestyles[sector],
                      color=color, marker=marker,
                     linewidth=3)

# plt.title('Carbon Stressor Trends (2000-2022)',weight='bold')
plt.xlabel('Year')
plt.ylabel('Carbon Stressor (kg CO2 eq./M.EUR)')
plt.legend(handlelength=3)
plt.grid(True)
plt.rcParams.update({'font.size': 14})
plt.tight_layout()
plt.savefig(os.path.join(graph_path,"Carbon Stressor Trends (2000-2022).png"), dpi=300)
plt.show()
#%%Rank of Carbon Stressor Heatmap in 2022
# E:/OneDrive/毕业论文/Data
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Data
matched=pd.read_csv(os.path.join(data_path,"Country Info.csv"))
country_dict = matched.set_index("iso_3166_1_")["name"].to_dict()
country_dict["WA"]="RoW Asia and Pacific"
country_dict["WE"]="RoW Europe"
country_dict["WF"]="RoW Africa"
country_dict["WL"]="RoW America"
country_dict["WM"]="RoW Middle East"
path_template = os.path.join(data_path,"Carbon_stressor_{year}.csv")
all_data = pd.DataFrame()
for year in range(2000, 2023):
    path = path_template.format(year=year)
    data=pd.read_csv(path,skiprows=1,header=None,index_col=[1,2])
    data['year']=year
    all_data = pd.concat([all_data, data],axis=0,ignore_index=False)

all_data = all_data.drop(0, axis=1)

# 重置索引，方便绘图
all_data.columns=[['Carbon Stressor','year']]
this_year_carbon_stressor=all_data[all_data['year'].squeeze()==2022]['Carbon Stressor']
this_year_carbon_stressor = this_year_carbon_stressor.rename_axis(index=['region', 'sector'])
this_year_carbon_stressor=this_year_carbon_stressor.unstack('region')
this_year_carbon_stressor.columns = this_year_carbon_stressor.columns.droplevel(0)
carbon_stressor_2022=this_year_carbon_stressor.T
ranked_df = this_year_carbon_stressor.rank(method='min', axis=1).astype(int)
ranked_df=ranked_df.shape[1]-ranked_df+1
ranked_df.columns = [country_dict.get(col, col) for col in ranked_df.columns]
sector_order[-1]='Non Secondary Products'
ranked_df = ranked_df.reindex(index=sector_order, columns=country_order)

fig, ax = plt.subplots(figsize=(35, 50))  # 指定图像大小
colors = ["red"] + sns.color_palette("viridis", n_colors=256-1)

# 创建一个自定义的颜色映射
custom_cmap = LinearSegmentedColormap.from_list("custom_viridis_red", colors)
# 绘制热力图，确保传入行和列的标签
heatmap=sns.heatmap(ranked_df, annot=False, fmt="d",cmap=custom_cmap,
            # cmap='viridis_r',
            xticklabels=ranked_df.columns,
            yticklabels=list(ranked_df.index), ax=ax,
            cbar_kws={"fraction": 0.03,"aspect":100,"label": "Rank of Carbon Intensity"})
cbar = heatmap.collections[0].colorbar

# 调整色带标签的位置
cbar.set_label('Rank of Carbon Intensity', labelpad=-80, y=0.5, rotation=90)
marks = {"China": "+", "United States of America": "x", "Global": "-"}  # 确保列名与你的DataFrame匹配
for col, mark in marks.items():
    if col in ranked_df.columns:
        for row in range(ranked_df.shape[0]):
            ax.text(ranked_df.columns.get_loc(col) + 0.5, row + 0.5, mark,
                    horizontalalignment='center', verticalalignment='center', fontsize=8,
                    alpha=0.9)

# industry_boxes = {
#     "Semiconductivity Related Products": "solid",
#     "Semiconductivity Upstream Products": "dashed",
#     "Semiconductivity Downstream Products": "dotted"
# }
# for row_label, linestyle in industry_boxes.items():
#     if row_label in ranked_df.index:
#         for col in range(len(ranked_df.columns)):
#             rect = Rectangle((col, ranked_df.index.get_loc(row_label)), 1, 1, fill=False,
#                              edgecolor="yellow", linestyle=linestyle, linewidth=2)
#             ax.add_patch(rect)
industries_shapes = {
    "Semiconductivity Related Products": "square",
    "Semiconductivity Upstream Products": "circle",
    "Semiconductivity Downstream Products": "hexagon"
}
for industry, shape in industries_shapes.items():
    if industry in ranked_df.index:
        row = ranked_df.index.get_loc(industry)
        for col in range(ranked_df.shape[1]):
            if shape == "square":
                rect = Rectangle((col+0.15, row+0.15), 0.6, 0.6, fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(rect)
            elif shape == "circle":
                circ = Circle((col + 0.5, row + 0.5), 0.35, fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(circ)
            elif shape == "hexagon":
                hexagon = RegularPolygon((col + 0.5, row + 0.5), numVertices=6, radius=0.35, 
                                         orientation=np.radians(30), fill=False, edgecolor="black", lw=0.5,alpha=0.9)
                ax.add_patch(hexagon)
# 将x轴标签移动到顶部
ax.xaxis.tick_top()  
ax.xaxis.set_label_position('top')  
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# 设置标题
# plt.title("Rank of Carbon Stressor Heatmap in 2022",weight='bold')
plt.rcParams.update({'font.size': 14})
plt.tight_layout()
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Graph
# E:/OneDrive/毕业论文/Graph
plt.savefig(os.path.join(graph_path,"Rank of Carbon Stressor Heatmap in 2022.png"), dpi=300)
# 显示图像
plt.show()
carbon_stressor_2022.to_csv(os.path.join(data_path,"Carbon Stressor(kg CO2 by M EUR) Heatmap in 2022.csv"))
#%%Carbon Uplift If No Trade 2022.png
sectors_to_plot = ["Semiconductivity Related Products", "Semiconductivity Upstream Products",
                   "Semiconductivity Downstream Products"]
country_dict = matched.set_index("name")["iso3"].to_dict()
country_dict["WA"]="RoW Asia and Pacific"
country_dict["WE"]="RoW Europe"
country_dict["WF"]="RoW Africa"
country_dict["WL"]="RoW America"
country_dict["WM"]="RoW Middle East"
country_dict["Global"]="Global"
# 转换为百分比的函数
def to_percent(y, position):
    s = "{:.0f}%".format(100 * y)
    return s

# 设置图形
fig, ax = plt.subplots(figsize=(20, 12))

# 数据准备
data = no_trade_carbon_uplift_2022.loc[sectors_to_plot, :].iloc[:, :-5].T
n = len(data.columns)  # 条形数量
width = 0.8 / n  # 条形宽度
ind = np.arange(len(data))  # x轴位置

# 自定义每组的样式
patterns = [ 'x', '-','o', '+', 'O', '.', '*','/', '\\',]

# 绘制条形图
for i, column in enumerate(data.columns):
    ax.bar(ind + i * width, data[column], width, label=column, hatch=patterns[i % len(patterns)])

# 设置Y轴格式化为百分比
formatter = FuncFormatter(to_percent)
ax.yaxis.set_major_formatter(formatter)

# 添加一些图形样式
# ax.set_title("Carbon Uplift If No Trade 2022")
ax.set_xticks(ind + width / 2)
ax.set_xticklabels([country_dict[x] for x in data.index], rotation='vertical')
plt.legend(title='Sectors',frameon=True, shadow=True, borderpad=1)
plt.ylabel('Carbon Uplift (%)')
plt.tight_layout()
plt.rcParams.update({'font.size': 14})
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Graph
# E:/OneDrive/毕业论文/Graph
plt.savefig(os.path.join(graph_path,"Carbon Uplift If No Trade 2022.png"), dpi=300)
# 显示图形
plt.show()
# E:/OneDrive/毕业论文/Data
data.to_csv(os.path.join(data_path,"Carbon Uplift If No Trade 2022.csv"),index=True)
#%%Comparison of Carbon Uplift between No Trade Scenario and No Sino-US Trade Scenario.png

# 将绘图代码封装为函数
def plot_data(ax, data, title):
    n = len(data.columns)  # 条形数量
    width = 0.8 / n  # 条形宽度
    ind = np.arange(len(data))  # x轴位置
    
    # 自定义每组的样式
    patterns = ['x', '-','o', '+', 'O', '.', '*','/', '\\']
    
    # 绘制条形图
    for i, column in enumerate(data.columns):
        ax.bar(ind + i * width, data[column], width, label=column, hatch=patterns[i % len(patterns)])
    
    # 设置Y轴格式化为百分比
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    
    # 图形样式设置
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(data.index, rotation='vertical')
    plt.legend(title='Sectors')
    plt.tight_layout()
    plt.rcParams.update({'font.size': 14})

# 转换为百分比的函数
def to_percent(y, position):
    s = "{:.2f}%".format(100 * y)
    return s

# 设置图形，现在是两个子图
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 12),sharey=True)

# 准备数据
data1 = no_trade_carbon_uplift_2022.loc[sectors_to_plot,['Global','China','United States of America']].T
data2 = no_bil_trade_carbon_uplift_2022.loc[sectors_to_plot,['Global','China','United States of America']].T
data3 = no_bil_trade_gamed_carbon_uplift_2022.loc[sectors_to_plot,['Global','China','United States of America']].T
# 在两个子图上绘图

plot_data(axs[0], data1, "Carbon Uplift If No Trade 2022")
plot_data(axs[1], data2, "Carbon Uplift If No Bilateral Trade 2022")
# plot_data(axs[2], data3, "Carbon Uplift If No Bilateral Trade with Gaming 2022")
plt.legend(title='Sectors',frameon=True, shadow=True, borderpad=1)
plt.rcParams.update({'font.size': 14})
axs[0].set_ylabel('Carbon Uplift (%)')
# plt.suptitle("Comparison of Carbon Uplift between No Trade Scenario and No Sino-US Trade Scenario",weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graph_path,"Comparison of Carbon Uplift between No Trade Scenario and No Sino-US Trade Scenario.png"), dpi=300)
plt.show()
#%%Comparison of Carbon Uplift between 3 Scenario.png

# 将绘图代码封装为函数
def plot_data(ax, data, title):
    n = len(data.columns)  # 条形数量
    width = 0.8 / n  # 条形宽度
    ind = np.arange(len(data))  # x轴位置
    
    # 自定义每组的样式
    patterns = ['x', '-','o', '+', 'O', '.', '*','/', '\\']
    
    # 绘制条形图
    for i, column in enumerate(data.columns):
        ax.bar(ind + i * width, data[column], width, label=column, hatch=patterns[i % len(patterns)])
    
    # 设置Y轴格式化为百分比
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    
    # 图形样式设置
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(data.index, rotation='vertical')
    # plt.legend(title='Sectors')
    plt.tight_layout()
    plt.rcParams.update({'font.size': 14})

# 转换为百分比的函数
def to_percent(y, position):
    s = "{:.2f}%".format(100 * y)
    return s

# 设置图形，现在是两个子图
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(24, 12),sharey=True)

# 准备数据
data1 = no_trade_carbon_uplift_2022.loc[sectors_to_plot,['Global','China','United States of America']].T
data2 = no_bil_trade_carbon_uplift_2022.loc[sectors_to_plot,['Global','China','United States of America']].T
data3 = no_bil_trade_gamed_carbon_uplift_2022.loc[sectors_to_plot,['Global','China','United States of America']].T
# 在两个子图上绘图

plot_data(axs[0], data1, "Carbon Uplift If No Trade 2022")
plot_data(axs[1], data2, "Carbon Uplift If No Bilateral Trade 2022")
plot_data(axs[2], data3, "Carbon Uplift If No Bilateral Trade with Gaming 2022")
axs[1].legend(title='Sectors',frameon=True, shadow=True, borderpad=1,loc='upper center')
plt.rcParams.update({'font.size': 14})
axs[0].set_ylabel('Carbon Uplift (%)')
# plt.suptitle("Comparison of Carbon Uplift between 3 Scenario",weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graph_path,"Comparison of Carbon Uplift between 3 Scenario.png"), dpi=300)
plt.show()
#%%Carbon Stressor in Different Sectors 2022.png
sectors_to_plot=["Semiconductivity Related Products","Semiconductivity Upstream Products",
                 "Semiconductivity Downstream Products"]
country_dict = matched.set_index("iso_3166_1_")["iso3"].to_dict()
country_dict["WA"]="RoW Asia and Pacific"
country_dict["WE"]="RoW Europe"
country_dict["WF"]="RoW Africa"
country_dict["WL"]="RoW America"
country_dict["WM"]="RoW Middle East"
carbon_stressor_2022.index = [country_dict.get(idx, idx) for idx in carbon_stressor_2022.index]
colors = plt.cm.tab10(np.linspace(0, 1, carbon_stressor_2022.shape[0]))
color_dict = dict(zip(carbon_stressor_2022.index, colors))
country_markers = {
    'USA': ('#00487C', '^'),  # 颜色和标记样式
    'CHN': ('#FC440F', 's'),
    'Global': ("green", 'D')
}
fig, ax = plt.subplots(figsize=(35, 50))  # Adjusting figure size for vertical layout
plt.rcParams.update({'font.size': 14})
texts = []
country_positions = {country: [] for country in country_markers}
# Iterate over sectors to plot each one
for i, sector in enumerate(sectors_to_plot):
    # For each sector, iterate over countries to plot their values
    for j, country in enumerate(carbon_stressor_2022.index):
        # Retrieve the value for the country and sector
        value = carbon_stressor_2022.loc[country, sector]
        # Check if value is greater than 0 to avoid issues with log scale
        if value > 0:
            # Convert the sector to a numeric x-position
            x_position = i
            if country in country_markers:
                color, marker = country_markers[country]
                zorder = 3
                country_positions[country].append((x_position, value))
                if i==0:
                    ax.scatter(x_position, value, s=30,color=color, marker=marker,zorder = zorder,alpha=0.8,label=country)  # s is the size of the point
                else:
                    ax.scatter(x_position, value, s=30,color=color, marker=marker,zorder = zorder,alpha=0.8)  # s is the size of the point
            else:
                color = color_dict[country]  # Use the color from the color_dict
                marker = 'o'  # Default marker
                zorder = 2
                ax.scatter(x_position, value, s=30,color=color, marker=marker,zorder = zorder,alpha=0.7)  # s is the size of the point
            # Plot the value on a vertical axis
            
            # Add text label for the country next to the point
            # ax.text(x_position, value, get_iso3_or_keep_country(country, matched), 
            #         rotation=0, ha='right', va='bottom')
            text = ax.text(x_position, value, country, 
                           rotation=0, ha='right', va='bottom',alpha=0.9)
            texts.append(text)

for country, positions in country_positions.items():
    if len(positions) > 1:  # 确保至少有两个点可以连接
        # 解压位置列表到两个分开的列表x和y
        x, y = zip(*positions)
        # 使用相应的颜色绘制折线
        line, = ax.plot(x, y, color=country_markers[country][0], lw=2,linestyle='-.',alpha=0.5,label=f'Line for {country}')
        ax.legend(handles=[line], loc='best')
# 添加图例
ax.legend()

# Set the x-axis to have one tick per sector, with labels
ax.set_xticks(range(len(sectors_to_plot)))
ax.set_xticklabels([fill(label, width=20) for label in sectors_to_plot], rotation=0, ha='right')
ax_xlims = ax.get_xlim()
new_xlims = (ax_xlims[0] - 1, ax_xlims[1] + 1)  # 在两边各增加1的间距，根据需要调整这个值
ax.set_xlim(new_xlims)
# Set the y-axis to a logarithmic scale
ax.set_yscale('log')
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'),alpha=0.5,
            avoid_self=True,
            only_move={'text': 'x','explode':'xy','pull':'y','static':'xy'},  # 仅在y方向上移动
            force_text=(0.8,2),  # 增加水平方向的排斥力，减少垂直方向的
            force_explode=(0.07, 0.3),  # 减少爆炸力，尤其是垂直方向的
            force_pull=(0.5, 3.5),  # 适当增加吸引力
            force_static=(0.08, 0.05),  # 适当增加静态力
            explode_radius=5.5,
            iter_lim=100
        )
for text in texts:
    if (text.get_text() == "KOR") and  (text._x>1):
        text.set_position((text.get_position()[0]+0.3, text.get_position()[1] - 1.2e3))  # 向上移动
    if (text.get_text() == "IDN") and  (text._x>1):
        text.set_position((text.get_position()[0]+0.2, text.get_position()[1] - 1e3))  # 向上移动
# Set labels for axes
ax.set_xlabel('Sector')
ax.set_ylabel('Carbon Stressor (kg CO2 eq./M.EUR)')

# Add a title to the plot
# ax.set_title('Carbon Stressor in Different Sectors 2022',weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graph_path,"Carbon Stressor in Different Sectors 2022.png"), dpi=300)
# Show the plot
plt.show()

filtered_data=carbon_stressor_2022[sectors_to_plot]
#%%Net Semi-Conductor Export Ratio vs. Carbon Uplift If No Sino-US Trade with Gaming.png
def aggregate_data(original_data,vertical,aggregate_list,name):
    aggregated_stressor=original_data.copy()
    # aggregated_stressor=aggregated_stressor.set_index(['region','sector'])
    if vertical == 'sector':
        aggregated_stressor=aggregated_stressor.loc[(slice(None),aggregate_list),:].groupby(level=0).sum()
        aggregated_stressor[vertical] = name
        aggregated_stressor.set_index(vertical, append=True, inplace=True)
    else:
        print("Not Valid vertical!")
        return
    aggregated_stressor=aggregated_stressor.swaplevel()
    aggregated_stressor.reset_index(drop=False,inplace=True)
    return aggregated_stressor
def to_percent(y, position):
    s = "{:.0f}%".format(100 * y)
    return s
country_markers = {
    'US': ('#00487C', 'h'),  # 颜色和标记样式
    'CN': ('#FC440F', 's')
}
country_dict = matched.set_index("iso_3166_1_")["name"].to_dict()
country_dict["WA"]="RoW Asia and Pacific"
country_dict["WE"]="RoW Europe"
country_dict["WF"]="RoW Africa"
country_dict["WL"]="RoW America"
country_dict["WM"]="RoW Middle East"
country_dict_3 = matched.set_index("iso_3166_1_")["iso3"].to_dict()
country_dict_3["WA"]="RoW Asia and Pacific"
country_dict_3["WE"]="RoW Europe"
country_dict_3["WF"]="RoW Africa"
country_dict_3["WL"]="RoW America"
country_dict_3["WM"]="RoW Middle East"
import_and_export=pd.concat([VA['D_exp'],VA['D_imp'],VA['F']],axis=1,ignore_index=False)
import_and_export.columns=['export','import','value added']
upstream_netexport_data=aggregate_data(import_and_export,'sector',upstream_semi_products,'Semiconductivity Upstream Products')
downstream_netexport_data=aggregate_data(import_and_export,'sector',downstream_semi_products,"Semiconductivity Downstream Products")
secondary_netexport_data=aggregate_data(import_and_export,'sector',secondary_products,"Secondary Products")
non_secondary_netexport_data=aggregate_data(import_and_export,'sector',non_secondary_products,"Non Secondary Products")
semi_netexport_data=aggregate_data(import_and_export,'sector',semi_products,"Semiconductivity Related Products")
all_netexport_data=aggregate_data(import_and_export,'sector',list(sectors),"All Products")
netexport_data=pd.concat([import_and_export.reset_index(),upstream_netexport_data,downstream_netexport_data,secondary_netexport_data,non_secondary_netexport_data,
                          semi_netexport_data,all_netexport_data],axis=0,ignore_index=False)
netexport_data=netexport_data.set_index(['region','sector'])
netexport_data['NE_ratio']=(netexport_data['export']-netexport_data['import'])/(netexport_data['value added']+1e-9)
plt.figure(figsize=(15, 12))

# 散点大小的缩放因子
size_scale = 2e-3

# 文本对象列表，将被用于adjust_text
texts = []
legend_handles=[]
for idx in netexport_data.loc[(regions, "Semiconductivity Related Products"), :]['NE_ratio'].index:
    scatter_size = netexport_data['value added'][idx] * size_scale
    if idx[0] in country_markers:
        color, marker = country_markers[idx[0]]
        plt.scatter(netexport_data['NE_ratio'][idx], no_bil_trade_gamed_carbon_uplift_2022.loc[idx[1], country_dict[idx[0]]],
                    s=scatter_size, color=color, marker=marker, alpha=0.7, edgecolors='w')
        legend_handles.append(Line2D([0], [0], color=color, marker=marker, label=country_dict_3[idx[0]],
                                  markersize=14,alpha=0.7,linestyle='none'))
    else:
        plt.scatter(netexport_data['NE_ratio'][idx], no_bil_trade_gamed_carbon_uplift_2022.loc[idx[1], country_dict[idx[0]]],
                    s=scatter_size, alpha=0.7)
    
    # 为每个点添加文本标签，并加入到texts列表中
    text = plt.text(netexport_data['NE_ratio'][idx], no_bil_trade_gamed_carbon_uplift_2022.loc[idx[1], country_dict[idx[0]]],
                    country_dict_3[idx[0]], fontsize=12, ha='right')
    texts.append(text)

# 设置y轴为百分比格式
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'),alpha=0.5,
            avoid_self=True,
            only_move={'text': 'xy','explode':'y','pull':'y','static':'xy'},  # 仅在y方向上移动
            force_text=(0.1,0.3),  # 增加水平方向的排斥力，减少垂直方向的
            force_explode=(0.2, 0.2),  # 减少爆炸力，尤其是垂直方向的
            force_pull=(0.3, 0.8),  # 适当增加吸引力
            force_static=(0.1, 0.1),  # 适当增加静态力
            explode_radius=3,
            iter_lim=100
        )

# 添加代表特定value added值的散点图例条目
legend_handles.append(plt.scatter([], [], s=1e4 * size_scale, color='gray', alpha=0.7, edgecolors='w', label=f"Value Added = 1e4 {VA['unit'].values[0][0]}"))
legend_handles.append(plt.scatter([], [], s=1e5 * size_scale, color='gray', alpha=0.7, edgecolors='w', label=f"Value Added = 1e5 {VA['unit'].values[0][0]}"))
# 添加标题和轴标签
# plt.title('Net Semi-Conductor Export Ratio vs. Carbon Uplift If No Sino-US Trade with Gaming',weight='bold')
plt.xlabel('Net Semi-Conductor Export Ratio')
plt.ylabel('Carbon Uplift')
plt.rcParams.update({'font.size': 14})  # 可以调整为你想要的大小
# 添加图例（使用自定义图例句柄和调整边框大小）
plt.legend(handles=legend_handles, loc='upper left', title='Notes', 
           fontsize='small', frameon=True, shadow=True, borderpad=1)
plt.savefig(os.path.join(graph_path,"Net Semi-Conductor Export Ratio vs. Carbon Uplift If No Sino-US Trade with Gaming.png"), dpi=300)
plt.show()
filtered_data=netexport_data.loc[(regions, 'Semiconductivity Downstream Products'), :]['NE_ratio'].to_frame()
filtered_data2=no_bil_trade_gamed_carbon_uplift_2022.loc["Semiconductivity Related Products"]
filtered_data2=filtered_data2.drop("Global").rename("Carbon Uplift If No Sino-US Trade with Gaming")
filtered_data.index=filtered_data2.index
filtered_data=pd.concat([filtered_data,filtered_data2],axis=1,ignore_index=False)
# =============================================================================
# #%%
# fig, axes = plt.subplots(7, 1, figsize=(10, 20))
# axes=axes.flatten()
# 
# for i,semi_product in enumerate(semi_products):
#     texts = []  # 用于收集所有的文本对象
#     no_semitrade_by_country = L @ Y_reg.reindex(columns=regions)
#     S=GWP100['S'].copy()
#     for country in regions:
#         new_s=S.loc[(country, semi_product),:]
#         for other_country in set(regions)-set(country): 
#             new_s.index=S.loc[(other_country, semi_product),:].index
#             S.loc[(other_country, semi_product),:]=new_s
#         no_semitrade_by_country.loc[:,country]=no_semitrade_by_country.loc[:,country]*S[ghg]
#     carbon_gap_notrade=no_semitrade_by_country.sum(axis=1).groupby(level=0).sum()/ GWP100['F'].sum(axis=1).groupby(level=0).sum()
#     carbon_gap_notrade=carbon_gap_notrade.rename("Carbon Gap No Trade")
#     # (carbon_gap_notrade-1).plot(kind='bar',figsize=(15,12))
#     carbon_gap_adjusted=carbon_gap_notrade-1
#     net_semi_export=(VA['D_exp']-VA['D_imp']).loc[(regions, semi_product),:].groupby(level=0).sum()
#     for idx in net_semi_export.index:
#         if idx not in ['WA','WE','WF','WL']:
#             x = net_semi_export['Value Added'][idx]
#             y = carbon_gap_adjusted[idx]
#             axes[i].scatter(x, y)
#             # 为每个散点添加文本对象到列表，以便后续调整
#             texts.append(axes[i].text(x, y, idx))
#     # 添加图例
#     # axes[i].legend()
#     # 添加标题和轴标签
#     axes[i].set_title(semi_product)
#     plt.xlabel('Net Semi Export')
#     plt.ylabel('Carbon Gap Notrade - 1')
#     adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'),
#             force_explode=(0.001,0.002),
#             force_pull=(0.06,0.06),force_static=(0.06,0.01),
#             force_text=(0.06,0.06),ax=axes[i])    
#     # 显示图形
# plt.show()
# =============================================================================
#%%Comparison of Carbon Emission (GHG) between PBA and CBA.png
# /Users/bytedance/Documents/毕业论文/
# E:/OneDrive/毕业论文/world-administrative-boundaries
world_map = gpd.read_file(os.path.join(root_path,"world-administrative-boundaries/world-administrative-boundaries.shp"))
map_order_dict=dir(world_map)
world_map.columns
world_map = world_map.merge(GWP100['D_cba_reg'],how='outer', left_on='iso_3166_1_', right_index=True)
world_map = world_map.merge(GWP100['D_pba_reg'],how='outer', left_on='iso_3166_1_', right_index=True)
colors = colors = ['#42113C','#274156', "#17BEBB","#B96D40" ,'#E5446D']
# colors = ["pink","red"]  # 这里可以根据需要调整颜色
n_bins = 100  # 这里可以增加以得到更平滑的颜色过渡
cmap_name = "my_custom_cmap"
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
# 绘制地图
plt.rcParams.update({
    'font.size': 14, # 设置图表中的字体大小
    'font.family': 'sans-serif' # 可以选用你喜欢的字体风格
})
fig, axes = plt.subplots(2, 1, figsize=(30, 18))
axes=axes.flatten()
for ax in axes:
    ax.set_facecolor('lightblue')
missing_kwds = {
    "color": "lightgrey",  # 没有数据的地方使用的颜色
    "edgecolor": "black",   # 边界颜色
    # "hatch": "///",        # 可选，如果你想要有图案的话
    # "label": "Missing data"  # 图例项的标签
}
world_map.plot(column=ghg+'_x', cmap=cm,
               linewidth=0.6, ax=axes[0], edgecolor="black", legend=False,
               norm=LogNorm(vmin=world_map[ghg+'_x'].min() , vmax=world_map[ghg+'_x'].max()),
               missing_kwds=missing_kwds,facecolor='lightblue')
sm_cba = ScalarMappable(norm=LogNorm(vmin=world_map[ghg + '_x'].min(), vmax=world_map[ghg + '_x'].max()), cmap=cm)
sm_cba._A = []
cbar_cba = plt.colorbar(sm_cba, ax=axes[0])
cbar_cba.set_label(f"Carbon Emission by CBA ({GWP100['unit'].values[0][0]})", rotation=270, labelpad=20)
# 设置标题
axes[0].set_title('Carbon Emission by CBA', pad=20)
# 隐藏坐标轴
axes[0].set_axis_off()
norm = Normalize(vmin=-0.005, vmax=0.005)
world_map.plot(column=ghg+'_x', cmap=cm,
               linewidth=0.6, ax=axes[1], edgecolor="black", legend=False,
               norm=LogNorm(vmin=world_map[ghg+'_y'].min() , vmax=world_map[ghg+'_y'].max()),
               missing_kwds=missing_kwds,facecolor='lightblue'
               ) 
sm_pba = ScalarMappable(norm=LogNorm(vmin=world_map[ghg + '_y'].min(), vmax=world_map[ghg + '_y'].max()), cmap=cm)
sm_pba._A = []
cbar_pba = plt.colorbar(sm_pba, ax=axes[1])
cbar_pba.set_label(f"Carbon Emission by PBA ({GWP100['unit'].values[0][0]})", rotation=270, labelpad=20)
axes[1].set_title('Carbon Emission by PBA', pad=20)
# 隐藏坐标轴
axes[1].set_axis_off()
# 显示地图
# plt.suptitle("Comparison of Carbon Emission (GHG) between PBA and CBA",weight='bold')
plt.tight_layout()
# E:/OneDrive/毕业论文/Graph
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Graph
plt.savefig(os.path.join(graph_path,"Comparison of Carbon Emission (GHG) between PBA and CBA.png"), dpi=300)
plt.show()
result=world_map[['name',ghg+'_x',ghg+'_y']][world_map.notna().all(axis=1)]
result.columns=['country','CBA','PBA']
# E:/OneDrive/毕业论文/Data/
result.to_csv(os.path.join(data_path,f"CBA and PBA ({GWP100['unit'].values[0][0]}).csv",index=False))
#%%Comparison of Carbon Emission (GHG) by CBA and Carbon Uplift.png
world_map = gpd.read_file(os.path.join(root_path,"world-administrative-boundaries/world-administrative-boundaries.shp"))
map_order_dict=dir(world_map)
world_map.columns
world_map = world_map.merge(GWP100['D_cba_reg'],how='outer', left_on='iso_3166_1_', right_index=True)
world_map = world_map.merge(no_trade_carbon_uplift_2022.stack().loc["Semiconductivity Related Products"].rename("Carbon Uplift If No Trade"),how='outer', left_on='name', right_index=True)
colors = colors = ['#42113C','#274156', "#17BEBB","#B96D40" ,'#E5446D']
# colors = ["pink","red"]  # 这里可以根据需要调整颜色
n_bins = 100  # 这里可以增加以得到更平滑的颜色过渡
cmap_name = "my_custom_cmap"
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
# 绘制地图
plt.rcParams.update({
    'font.size': 14, # 设置图表中的字体大小
    'font.family': 'sans-serif' # 可以选用你喜欢的字体风格
})
fig, axes = plt.subplots(2, 1, figsize=(30, 18))
axes=axes.flatten()
for ax in axes:
    ax.set_facecolor('lightblue')
missing_kwds = {
    "color": "lightgrey",  # 没有数据的地方使用的颜色
    "edgecolor": "black",   # 边界颜色
    # "hatch": "///",        # 可选，如果你想要有图案的话
    # "label": "Missing data"  # 图例项的标签
}
world_map.plot(column='GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)', cmap=cm,
               linewidth=0.6, ax=axes[0], edgecolor="black", legend=False,
               norm=LogNorm(vmin=world_map[ghg].min() , vmax=world_map[ghg].max()),
               missing_kwds=missing_kwds,facecolor='lightblue')
sm_cba = ScalarMappable(norm=LogNorm(vmin=world_map[ghg].min(), vmax=world_map[ghg].max()), cmap=cm)
sm_cba._A = []
cbar_cba = plt.colorbar(sm_cba, ax=axes[0])
cbar_cba.set_label(f"Carbon Emission by CBA ({GWP100['unit'].values[0][0]})", rotation=270, labelpad=20)
# 设置标题
axes[0].set_title('Carbon Emission by CBA', pad=20)
# 隐藏坐标轴
axes[0].set_axis_off()
norm = Normalize(vmin=-0.005, vmax=0.005)
world_map.plot(column="Carbon Uplift If No Trade", cmap=cm, linewidth=0.6, ax=axes[1],
               edgecolor='black',norm=norm,legend=False,
               missing_kwds=missing_kwds,facecolor='lightblue'
               ) 
def to_percentage(x, pos):
    return '{:.2f}%'.format(x * 100)
sm = ScalarMappable(cmap=cm, norm=norm)
sm._A = []
cbar = plt.colorbar(sm, ax=axes[1], format=FuncFormatter(to_percentage))
cbar.ax.set_ylabel('Carbon Uplift If No Trade (%)', rotation=270, labelpad=15)
axes[1].set_title('Carbon Uplift If No Trade', pad=20)
# 隐藏坐标轴
axes[1].set_axis_off()
# 显示地图
# plt.suptitle("Comparison of Carbon Emission (GHG) by CBA and Carbon Uplift",weight='bold')
plt.tight_layout()
# E:/OneDrive/毕业论文/Graph
# /Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/Graph
plt.savefig(os.path.join(graph_path,"Comparison of Carbon Emission (GHG) by CBA and Carbon Uplift.png"), dpi=300)
plt.show()
result=world_map[['name',ghg,"Carbon Uplift If No Trade"]][world_map.notna().all(axis=1)]
result.columns=['country','CBA',"Carbon Uplift If No Trade"]
# E:/OneDrive/毕业论文/Data
result.to_csv(os.path.join(data_path,f"CBA({GWP100['unit'].values[0][0]}) and Carbon Uplift If No Trade.csv"),index=False)
#%%test
print(exio3.impacts.unit.loc["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"])
print(exio3.impacts.D_cba_reg.loc["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"])
GWP100['D_cba'].plot(kind='bar',figsize=(15,12))
exio3.impacts.D_cba_reg.loc["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"].plot(kind='bar',figsize=(15,12))
#%%test
with plt.style.context("ggplot"):
    exio3.impacts.plot_account(["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"], figsize=(15, 10))
    plt.show()