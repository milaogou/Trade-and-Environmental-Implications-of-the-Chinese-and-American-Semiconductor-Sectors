# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:20:57 2024

@author: Administrator
"""
import pandas as pd
import numpy as np
import time
import os
from mpmath import mp
from multiprocessing import Pool
from scipy.io import loadmat
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
#%%
# root_path="E:\OneDrive\毕业论文\WIOTS_in_EXCEL"
root_path="/Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/WIOTS_in_EXCEL/"
xlsb=os.path.join(root_path,"WIOT2014Data.xlsb")
df_Z=pd.read_excel(xlsb,sheet_name="Z",header=None,engine='pyxlsb')
df_X=pd.read_excel(xlsb,sheet_name="X",header=None,engine='pyxlsb')
df_Y=pd.read_excel(xlsb,sheet_name="Y",header=None,engine='pyxlsb')
Country=pd.read_excel(xlsb,sheet_name="Country",header=None,engine='pyxlsb')
Industry=pd.read_excel(xlsb,sheet_name="Industry",header=None,engine='pyxlsb')
indices = [(country, industry) for country in Country[0] for industry in Industry[0]]
indices = pd.MultiIndex.from_tuples(indices, names=['Country', 'Industry'])
Y_category=pd.read_excel(xlsb,sheet_name="Y_category",header=None,engine='pyxlsb')
#%%
# # 使用Pandas的ExcelFile类来处理多个sheet
# xlsx = pd.ExcelFile("E:\OneDrive\毕业论文\WIOTS_in_EXCEL\CO2 emissions.xlsx")
# # 初始化一个空的DataFrame来存储所有数据
# df_carbon = pd.DataFrame()

# # 从第三个sheet开始遍历（索引从0开始，所以是从2开始）
# for sheet_name in xlsx.sheet_names[2:]:
#     # 读取当前sheet
#     df = pd.read_excel(xlsx, sheet_name=sheet_name,index_col=None,header=0)
#     df['Country']=sheet_name
#     df['Industry']=df['Unnamed: 0']
#     df_2014 = df[['Country','Industry','2014']]
#     df_2014.columns=['Country','Industry','Carbon_emission']
#     df_carbon = pd.concat([df_carbon, df_2014], axis=0,ignore_index=True)
# df_carbon['Country']=df_carbon['Country'].apply(lambda x:'ROW' if x=='RoW' else x)
# df_carbon.to_csv(r'E:\OneDrive\毕业论文\WIOTS_in_EXCEL\2014_carbon_emissions.csv',index=False)
#%%
df_carbon=pd.read_csv(os.path.join(root_path,"2014_carbon_emissions.csv"))
df_carbon.groupby('Country').count()
df_carbon.groupby('Industry').count()
print(set(df_carbon['Industry'].unique())-set(Industry[0]))
print(df_carbon.loc[df_carbon['Industry']=='FC_HH',['Carbon_emission']].sum())
print(df_carbon.loc[df_carbon['Industry']!='FC_HH',['Carbon_emission']].sum())
print(set(df_carbon['Country'].unique())-set(Country[0]))
print(set(Country[0])-set(df_carbon['Country'].unique()))
#%%
carbon_data=df_carbon[df_carbon['Industry']!='FC_HH']
carbon_data=carbon_data.set_index(['Country','Industry'])
carbon_data=carbon_data.reindex(indices)
carbon_data.fillna(0, inplace=True)
carbon_data=carbon_data.values
#%%
Z = df_Z.to_numpy()
X = df_X.to_numpy()
Y = df_Y.to_numpy()
epsilon = 1e-9#np.finfo(X.dtype).tiny
X = X + epsilon
f = carbon_data.T/X
# 计算技术系数矩阵A
A = Z / X
np.isclose(X, 0).mean()
np.isnan(f).mean()
np.isnan(A).mean()
# 构造单位矩阵I
I = np.eye(Z.shape[0])

# 计算里昂惕夫逆矩阵L
# L = np.linalg.inv(I - A)
L = np.linalg.solve(I - A, I)
A_check = np.linalg.solve(I - L, I)
np.isnan(L).mean()
#%%
Y_total = Y.sum(axis=1).reshape(-1, 1)
X_check =  L @ Y_total 
# 计算X_check与X的比值
column_sums = A.sum(axis=0)
result_check = X_check.T / X
result_check.mean()
#%%
Y_Country = np.sum(Y.reshape(-1,44, 5), axis=2)
Carbon_CBA=  (f @ L @ Y_Country).T
Carbon_CBA.shape
Carbon_PBA=np.sum(carbon_data.reshape(44,56), axis=1).reshape(-1,1)
Carbon_PBA.shape
Carbon_CBA_df=pd.DataFrame(Carbon_CBA,index=list(Country[0]),columns=["Carbon_CBA"])
Carbon_PBA_df=pd.DataFrame(Carbon_PBA,index=list(Country[0]),columns=["Carbon_PBA"])
Carbon_delta=Carbon_CBA-Carbon_PBA
Carbon_delta_df=pd.DataFrame(Carbon_delta,index=list(Country[0]),columns=["Carbon_delta"])
#%%
Y_leveled=Y_Country.reshape(44,56,44)
mask = np.zeros_like(Y_leveled)

# 步骤2: 设置mask中对应位置为1
for i in range(min(mask.shape[0], mask.shape[2])):
    mask[i, :, i] = 1
Y_leveled_masked = Y_leveled * mask
Y_leveled_masked=Y_leveled_masked.reshape(-1,44)
Y_local=(f @ L @ Y_leveled_masked).T
#%%
Y_leveled=Y_Country.reshape(44,56,44)
mask = np.zeros_like(Y_leveled)

# 步骤2: 设置mask中对应位置为1
for i in range(min(mask.shape[0], mask.shape[2])):
    mask[i, :, i] = 1
mask=1-mask
Y_leveled_masked = Y_leveled * mask
Y_leveled_masked=Y_leveled_masked.reshape(-1,44)
Y_inter=(f @ L @ Y_leveled_masked).T
#%%
Y_local+Y_inter-Carbon_CBA
#%%
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
# 读取中国各省份的地理信息数据
world_map = gpd.read_file("/Users/bytedance/Library/CloudStorage/OneDrive-个人/毕业论文/world-administrative-boundaries/world-administrative-boundaries.shp")
# world_map = gpd.read_file("E:\OneDrive\毕业论文\world-administrative-boundaries\world-administrative-boundaries.shp")
# pd.DataFrame(china_map['省']).to_excel("E:\毕业论文数据\Province MRIO\province_in_shp.xlsx",encoding='utf-8')
# 将地图和碳排放量数据合并
world_map = world_map.merge(Carbon_CBA_df,how='outer', left_on='iso3', right_index=True)
world_map = world_map.merge(Carbon_PBA_df,how='outer', left_on='iso3', right_index=True)
world_map = world_map.merge(Carbon_delta_df,how='outer', left_on='iso3', right_index=True)
NA_countries=world_map['Carbon_CBA'].isna().count()
# world_map['Carbon_CBA']=world_map['Carbon_CBA'].fillna(Carbon_CBA_df.loc['ROW'].values[0]/NA_countries)
# world_map['Carbon_PBA']=world_map['Carbon_PBA'].fillna(Carbon_PBA_df.loc['ROW'].values[0]/NA_countries)
world_map[world_map["Carbon_CBA"].isna()]['iso3']
#%%
colors = ['#EBB6B4','#800526']
# colors = ["pink","red"]  # 这里可以根据需要调整颜色
n_bins = 10  # 这里可以增加以得到更平滑的颜色过渡
cmap_name = "my_custom_cmap"
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
# 绘制地图
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes=axes.flatten()
world_map.plot(column='Carbon_CBA', cmap=cm, linewidth=0.8, ax=axes[0], edgecolor='0.8', legend=True,
               norm=LogNorm(vmin=world_map['Carbon_CBA'].min() , vmax=world_map['Carbon_CBA'].max()))
# 设置标题
axes[0].set_title('Carbon_CBA')
# 隐藏坐标轴
axes[0].set_axis_off()
world_map.plot(column='Carbon_PBA', cmap=cm, linewidth=0.8, ax=axes[1], edgecolor='0.8', legend=True,
               norm=LogNorm(vmin=world_map['Carbon_CBA'].min() , vmax=world_map['Carbon_CBA'].max()))
# 设置标题
axes[1].set_title('Carbon_PBA')
# 隐藏坐标轴
axes[1].set_axis_off()
# 显示地图
plt.tight_layout()
plt.show()
#%%
colors = ['#EBB6B4','#800526']
# colors = ["pink","red"]  # 这里可以根据需要调整颜色
n_bins = 10  # 这里可以增加以得到更平滑的颜色过渡
cmap_name = "my_custom_cmap"
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
# 绘制地图
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# axes=axes.flatten()
world_map.plot(column='Carbon_delta', cmap=cm, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
               # norm=LogNorm(vmin=world_map['Carbon_CBA'].min() , vmax=world_map['Carbon_CBA'].max())
               )
# 设置标题
ax.set_title('Carbon_delta')
# 隐藏坐标轴
ax.set_axis_off()
# 显示地图
plt.tight_layout()
plt.show()
