# NOTE: 
# While this is a py file, it is meant to be used like a jupyter notebook. 
# Each new cell is demarcated by '# %%'

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
"""
This script contains the code to generate plots that compare the measured performance with the actual distance.

"""
# %%
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats

# %% [markdown]
# ## XCode Data Processing

# %%
df = pd.read_excel("XCodeData.xlsx")
df

# %%
def standardize(df):
    df_1 = df[['Device Pair', 'Type', '1.0m']]
    df_1 = df_1.rename({"1.0m":"Measured Distance [m]"}, axis=1)
    df_1['True Distance [m]'] = 1.0
    
    df_15 = df[['Device Pair', 'Type', '1.5m']]
    df_15 = df_15.rename({"1.5m":"Measured Distance [m]"}, axis=1)
    df_15['True Distance [m]'] = 1.5

    df_2 = df[['Device Pair', 'Type', '2.0m']]
    df_2 = df_2.rename({"2.0m":"Measured Distance [m]"}, axis=1)
    df_2['True Distance [m]'] = 2.0

    df_new = pd.concat([df_1, df_15, df_2], ignore_index=True)
    df_new.reset_index(drop=True)
    return df_new
    


# %%
df_standard = standardize(df)
df_standard


# %%
types = list(set(df_standard['Type']))
types

# %%
pairs = list(set(df_standard['Device Pair']))
pairs


# %%
distances = list(set(df_standard['True Distance [m]']))
distances


# %%
def pruneOutliers(df):

    types = list(set(df['Type']))
    pairs = list(set(df['Device Pair']))
    distances = list(set(df['True Distance [m]']))

    all_dfs = []

    for p in pairs:
        for t in types:
            for d in distances:
                df_min = df[((df['Device Pair'] == p) & (df['Type'] == t) & (df['True Distance [m]'] == d))]
                if len(df_min) > 0:
                    # prune anything more than 0.75m away from the true value
                    # note, the prunning done for the live app is slightly different.

                    trimmed = df_min[np.abs(df_min['Measured Distance [m]'] - d) < 0.75] 

                    tm = np.mean(trimmed['Measured Distance [m]'])
                    s = np.std(trimmed['Measured Distance [m]'])

                    print(p,t)
                    print(f'    {d:.2f} \t {tm:.3f} \t {s:3f} \t {(1-len(trimmed)/len(df_min))*100:.1f}%')
                    print("")
                                    
                    all_dfs.append(trimmed)

    df_new = pd.concat(all_dfs)

    df_new = df_new[df_new['Measured Distance [m]'] < 3]

    return df_new.reset_index(drop=True)


# %%
df_clean = pruneOutliers(df_standard)
df_clean

# %%
perc = 1- len(df_clean)/len(df_standard)
print(f"Percentage of Pruned Data Points: {perc*100:.1f}%")



# %%
#sns.swarmplot(data=df_clean, x='True Distance [m]', y='Measured Distance [m]', hue="Type")


# %%
set(df_clean['Device Pair'])

# %%
# sns.catplot(data=df_clean, x='True Distance [m]', y='Measured Distance [m]', hue="Device Pair", kind='box', legend_out=True, whis=1, aspect=1, height=10)
# plt.grid()
# plt.ylim([0,3])


# %%
with sns.plotting_context("notebook",font_scale=1.25):
    sns.catplot(data=df_clean, x='True Distance [m]', y='Measured Distance [m]', hue="Type", kind='boxen', legend_out=False, aspect=1, height=10, hue_order = ["Speaker to Speaker", "Pocket to Speaker", "Other Pocket to Speaker"])
    plt.grid()
    plt.ylim([0,3])
    sns.despine(top=False, right=False)
    plt.savefig('images/xcodePerformance.pdf')

# %%
df_clean['Abs Error [cm]'] = 100*np.abs(df_clean['Measured Distance [m]'] - df_clean['True Distance [m]'])
df_clean



# %%
np.mean(df_clean['Abs Error [cm]'])


# %%
np.median(df_clean['Abs Error [cm]'])


# %%
df_s2s = df_clean[df_clean['Type'] == 'Speaker to Speaker']
df_s2s_1 = df_s2s[df_s2s['True Distance [m]'] == 1]
df_s2s_15 = df_s2s[df_s2s['True Distance [m]'] == 1.5]
df_s2s_2 = df_s2s[df_s2s['True Distance [m]'] == 2]

MAD_S2S_A = np.median(df_s2s['Abs Error [cm]'])
MAD_S2S_1 = np.median(df_s2s_1['Abs Error [cm]'])
MAD_S2S_15 = np.median(df_s2s_15['Abs Error [cm]'])
MAD_S2S_2 = np.median(df_s2s_2['Abs Error [cm]'])


print(f"Speaker to Speaker MAD [ALL ]: {MAD_S2S_A:.2f} cm")
print(f"Speaker to Speaker MAD [1  m]: {MAD_S2S_1:.2f} cm")
print(f"Speaker to Speaker MAD [1.5m]: {MAD_S2S_15:.2f} cm")
print(f"Speaker to Speaker MAD [2  m]: {MAD_S2S_2:.2f} cm")
print("")



df_p2s = df_clean[df_clean['Type'] == 'Pocket to Speaker']
df_p2s_1 = df_p2s[df_p2s['True Distance [m]'] == 1]
df_p2s_15 = df_p2s[df_p2s['True Distance [m]'] == 1.5]
df_p2s_2 = df_p2s[df_p2s['True Distance [m]'] == 2]


MAD_P2S_A = np.median(df_p2s['Abs Error [cm]'])
MAD_P2S_1 = np.median(df_p2s_1['Abs Error [cm]'])
MAD_P2S_15 = np.median(df_p2s_15['Abs Error [cm]'])
MAD_P2S_2 = np.median(df_p2s_2['Abs Error [cm]'])

print(f"Pocket to Speaker MAD [ALL ]: {MAD_P2S_A:.2f} cm")
print(f"Pocket to Speaker MAD [1  m]: {MAD_P2S_1:.2f} cm")
print(f"Pocket to Speaker MAD [1.5m]: {MAD_P2S_15:.2f} cm")
print(f"Pocket to Speaker MAD [2  m]: {MAD_P2S_2:.2f} cm")
print("")


df_o2s = df_clean[df_clean['Type'] == 'Other Pocket to Speaker']
df_o2s_1 = df_o2s[df_o2s['True Distance [m]'] == 1]
df_o2s_15 = df_o2s[df_o2s['True Distance [m]'] == 1.5]
df_o2s_2 = df_o2s[df_o2s['True Distance [m]'] == 2]

MAD_O2S_A = np.median(df_o2s['Abs Error [cm]'])
MAD_O2S_1 = np.median(df_o2s_1['Abs Error [cm]'])
MAD_O2S_15 = np.median(df_o2s_15['Abs Error [cm]'])
MAD_O2S_2 = np.median(df_o2s_2['Abs Error [cm]'])

print(f"Other Pocket to Speaker MAD [ALL ]: {MAD_O2S_A:.2f} cm")
print(f"Other Pocket to Speaker MAD [1  m]: {MAD_O2S_1:.2f} cm")
print(f"Other Pocket to Speaker MAD [1.5m]: {MAD_O2S_15:.2f} cm")
print(f"Other Pocket to Speaker MAD [2  m]: {MAD_O2S_2:.2f} cm")
print("")



df_a_1 = df_clean[df_clean['True Distance [m]'] == 1]
df_a_15 = df_clean[df_clean['True Distance [m]'] == 1.5]
df_a_2 = df_clean[df_clean['True Distance [m]'] == 2]


MAD_A = np.median(df_clean['Abs Error [cm]'])
MAD_A_1 = np.median(df_a_1['Abs Error [cm]'])
MAD_A_15 = np.median(df_a_15['Abs Error [cm]'])
MAD_A_2 = np.median(df_a_2['Abs Error [cm]'])


print(f"All MAD [ALL ]: {MAD_A:.2f} cm")
print(f"All MAD [1  m]: {MAD_A_1:.2f} cm")
print(f"All MAD [1.5m]: {MAD_A_15:.2f} cm")
print(f"All MAD [2  m]: {MAD_A_2:.2f} cm")
print("")


# %%
#print the latex code
print(f"""
Type & 1~m & 1.5~m & 2~m & All\\\\
Speaker to Speaker & {MAD_S2S_1:.2f} & {MAD_S2S_15:.2f} & {MAD_S2S_2:.2f} & {MAD_S2S_A:.2f} \\\\
Pocket to Speaker & {MAD_P2S_1:.2f} & {MAD_P2S_15:.2f} & {MAD_P2S_2:.2f} & {MAD_P2S_A:.2f} \\\\
Other Pocket to Speaker & {MAD_O2S_1:.2f} & {MAD_O2S_15:.2f} & {MAD_O2S_2:.2f} & {MAD_O2S_A:.2f} \\\\
All & {MAD_A_1:.2f} & {MAD_A_15:.2f} & {MAD_A_2:.2f} & {MAD_A:.2f} \\\\

""")
# %%
