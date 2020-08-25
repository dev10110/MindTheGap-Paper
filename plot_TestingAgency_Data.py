# NOTE: 
# While this is a py file, it is meant to be used like a jupyter notebook. 
# Each new cell is demarcated by '# %%'

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# %%
fileName = 'testing_agency_data.xlsx'
sheetsMap = pd.read_excel(fileName, sheet_name=None)

# %%
# load each sheet
s2s = sheetsMap['Speaker to Speaker']
p2s = sheetsMap['Pocket to Speaker']
p2p = sheetsMap['Pocket to Pocket']


# %%

def devicePair(row):
    # get the pair of devices used
    a = row['Device A']
    b = row['Device B']
    if a <= b:
        return a + " - " + b
    else:
        return b + " - " + a

def unroll(df):
    # convert each set of readings into its own row, for easy processing.
    # also remove:
    #   Tests by Tester 13 (did not follow protocol)
    #   Tests which claimed the alert distance was <0.5m. Something must be wrong.

    # generate dataframes for each repeat
    dfs = []
    for i in range(1,7):
        df_1 = df[['Test ID',"Tester ID", 'Device A', 'Device A OS', "Device B", "Device B OS", f"Reading {i}"]]
        df_1.rename({f"Reading {i}": "Alert Distance [m]"}, axis=1, inplace=True)
        dfs.append(df_1)

    # concatenate all results into one dataframe
    df_unroll = pd.concat(dfs, ignore_index=True)

    #Drop missing data points
    df_unroll_1 = df_unroll.dropna()

    # Drop results from tester 13
    df_unroll_2 = df_unroll_1[df_unroll_1['Tester ID'] != "MTG-2 Tester 13"]

    # Remove tests with alert distance < 0.5m
    df_unroll_3 = df_unroll_2[df_unroll_2['Alert Distance [m]'] >= 0.5]

    # print number of removed results
    print("Number removed: ", len(df_unroll_2) - len(df_unroll_3))
    print("Total originally: ", len(df_unroll_2))

    # sort
    df_unroll_4 = df_unroll_3.sort_values("Test ID", inplace=False)
    df_unroll_5 = df_unroll_4.reset_index(drop=True)
    
    # create column to indicate the device pair used
    df_unroll_5['Device Pair'] = df_unroll_5.apply(devicePair, axis=1)

    return df_unroll_5


# %%
# unrol data frames

s2s_unroll = unroll(s2s)
s2s_unroll['Type'] = "Speaker-Speaker"
s2s_unroll

# %%
p2s_unroll = unroll(p2s)
p2s_unroll['Type'] = "Pocket-Speaker"
p2s_unroll

# %%
p2p_unroll = unroll(p2p)
p2p_unroll['Type'] = "Pocket-Pocket"
p2p_unroll


# %%

# create dataframe of all tests

allTests = pd.concat([s2s_unroll, p2s_unroll, p2p_unroll])
allTests.sort_values(by="Test ID", inplace=True)
allTests = allTests.reset_index(drop=True)
allTests


# %%
def getDevicePerformance(testsDF):
    devicePairs = set(allTests['Device Pair'])
    dicts = []
    for d in devicePairs:
        df = testsDF[testsDF['Device Pair'] == d]
        ad = df['Alert Distance [m]']
        dictionary = {"Device Pair": d, "mean": np.mean(ad), "std": np.std(ad), "count": len(ad), "min": min(ad), "max": max(ad)}
        dicts.append(dictionary)
    df_perf = pd.DataFrame(dicts)
    df_sorted = df_perf.sort_values(by="mean", ascending=False)

    df_new = df_sorted.reset_index(drop=True)

    return df_new


# %%
perf_df = getDevicePerformance(allTests)
perf_df

# %%

# get number of unique devices
devices = set(allTests['Device A']).union(set(allTests['Device B']))
print("Number of unique Devices: ", len(devices))
print(devices)


# %%
devicePairOrder = list(perf_df['Device Pair'])
print("Number of pairs: ", len(devicePairOrder))
devicePairOrder



# %%
# fig = plt.figure(figsize=(12,12))
# sns.catplot(x = "Alert Distance [m]", y = 'Device Pair', data=allTests, kind="bar", hue='Type', aspect=2, height=10, order=devicePairOrder)
# plt.xlim([0,3])


# %%
# fig = plt.figure(figsize=(12,12))
# sns.catplot(x = "Alert Distance [m]", y = 'Device Pair', data=allTests, kind="box", hue='Type', aspect=2, height=10, order=devicePairOrder)
# plt.xlim([0,3])


# %%
# with sns.plotting_context("notebook",font_scale=1.25):
#     fig = plt.figure(figsize=(12,12))
#     ax = plt.gca()
#     sns.swarmplot(x = "Alert Distance [m]", y = 'Device Pair', data=allTests, hue='Type',  order=devicePairOrder, size=5, ax=ax)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.xlim([0,3])
#     plt.grid()


# %%
with sns.plotting_context("notebook",font_scale=1.25):
    
    sns.catplot(x = "Alert Distance [m]", y="Type", data=allTests, kind="boxen", hue='Type', aspect=1.5, height=6, legend=False)
    plt.axvline(2.0, color='r')
    plt.axvline(1.5, color='r')
    plt.xlim([0,2.5])
    plt.grid(True)
    plt.box(True)
    sns.despine(top=False, right=False, bottom=False, left=False)
    plt.savefig('images/testingAgencySpread.pdf')


# %%
# fig = plt.figure(figsize=(12,12))
# ax = sns.swarmplot(x = "Alert Distance [m]", y="Type", data=allTests, hue='Device Pair')
# ax = sns.boxplot(x = "Alert Distance [m]", y="Type", data=allTests, whis=1, fliersize=0,color=[1,1,1])
# ax.set_aspect(0.5)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
# ax.grid()
# plt.xlim([0,3])



# %% [markdown]
"""
Now I want to see a breakdown of median alerting distance for each type of test
"""

# %%
allTests['Alert Distance [m]'].describe()

# %%

a1 = np.median(allTests['Alert Distance [m]'])
print(f"Median Alert Distance across all tests: {a1:.2f} m")

# %%
a2 = np.mean(allTests['Alert Distance [m]'])
print(f"Mean Alert Distance across all tests: {a2:.2f} m")


# %%
a3 = np.median(s2s_unroll['Alert Distance [m]'])
a4 = np.median(p2s_unroll['Alert Distance [m]'])
a5 = np.median(p2p_unroll['Alert Distance [m]'])

print(f"Median Alert Distance Speaker to Speaker: {a3:.2f} m")
print(f"Median Alert Distance Pocket to Speaker:  {a4:.2f} m")
print(f"Median Alert Distance Pocket to Pocket:   {a5:.2f} m")
