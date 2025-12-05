import numpy as np
import pandas as pd
import scipy
import pathlib
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import h5py

HDF5_CYCLE_DSET_INDEX_DICT = {
    "time": 0,
    "TLD Temp": 1,
    "TFC Temp": 2,
    "Tboard Temp": 3,
    "ITEC LD Current": 4,
    "ITEC FC Current": 5,
    "ILD Current": 6,
    "FC1": 7,
    "FC1_amp": 8,
    "FC2": 9,
    "FC2_amp": 10,
    "FC3    ": 11,
    "FC3_amp": 12,
    "FC4": 13,
    "FC4_amp": 14,
}


def read_cyclesData():
    path_to_file = pathlib.Path("data/example/cyclesData.h5")
    f = h5py.File(path_to_file, "r")
    # print("Number of keys in cyclesData = ",len(list(f.keys())))
    # print("First 10 keys = ",list(f.keys())[0:10])
    # dset=f["0019b18b-ba44-40f2-82eb-0398d97fc77d.0"]
    # print(dset)
    return


def read_project_xml():
    path_to_file = pathlib.Path("data/example/_project.cx3")
    xml_tree = ET.parse(path_to_file)
    return xml_tree


def get_h5_data_for_guid(guid):
    path_to_file = pathlib.Path("data/example/cyclesData.h5")
    f = h5py.File(path_to_file, "r")
    datasets = [f[guid + "." + str(n)] for n in range(0, 15)]
    return datasets


def get_signal_tail_mean(df, end_fraction=0.1):
    i_end = int(np.floor(end_fraction * len(df)))
    signal_tail_mean = np.mean(df.iloc[:, -1][-i_end:])
    return signal_tail_mean


good_fit_data = pd.read_excel(
    pathlib.Path("data/good_fit_data.xlsx"), sheet_name="adjusted sample sensogram"
)
okay_fit_data = pd.read_excel(
    pathlib.Path("data/okay_fit_data.xlsx"), sheet_name="adjusted sample sensogram"
)
bad_fit_data = pd.read_excel(
    pathlib.Path("data/bad_fit_data.xlsx"), sheet_name="adjusted sample sensogram"
)
print(good_fit_data.head())

data_to_plot = good_fit_data

# signal_tail_mean = get_signal_tail_mean(data_to_plot)
# print("Mean of tail:  ",signal_tail_mean)
# zero_thresh = 0.05
# print("    so final dissoc complete?  ",bool(signal_tail_mean <= zero_thresh))
# plt.figure()
# plt.plot(data_to_plot.iloc[:,-2],data_to_plot.iloc[:,-1],".")

# nparray = data_to_plot.to_numpy()
# dRbydt = np.gradient(nparray[:,-1],nparray[:,-2])
# # plt.plot(nparray[:,-2],dRbydt)

# peaks, _ = scipy.signal.find_peaks(nparray[:,-1],prominence=1)
# cwt_peaks = scipy.signal.find_peaks_cwt(nparray[:,-1],np.arange(1,50))
# plt.plot(nparray[peaks,-2], nparray[peaks,-1], "x", color="green")
# #plt.plot(nparray[cwt_peaks,-2], nparray[cwt_peaks,-1], "o", color="red")
# plt.show()

h5_data = read_cyclesData()
xml = read_project_xml()

root = xml.getroot()
for serie in root.iter("Serie"):
    if serie.find("Type").text == "Pulse":
        pulse_serie = serie
        break

cycles = pulse_serie.iter("Cycle")  # iterator
sample_idx_to_guid = {}
for cycle in cycles:
    if cycle.find("CycleType").text == "Sample":
        idx = cycle.find("Index").text
        guid = cycle.find("Guid").text
        print(f"Sample: Index: {idx}, GUID: {guid}")
        sample_idx_to_guid[idx] = guid

sets_to_plot = [get_h5_data_for_guid(guid) for guid in sample_idx_to_guid.values()][
    0:20
]

fig = plt.figure()
ax1 = fig.add_subplot(111)
for d, dset in enumerate(sets_to_plot):
    t = dset[HDF5_CYCLE_DSET_INDEX_DICT["time"]][:]
    sig1 = dset[HDF5_CYCLE_DSET_INDEX_DICT["FC1"]][:]
    sig2 = dset[HDF5_CYCLE_DSET_INDEX_DICT["FC2"]][:]

    ax1.scatter(t, sig2 - sig1, label="Sample: " + str(d))
plt.legend(loc="upper left")
plt.show()

example_sample_idxs = {
    1: "Startup",
    11: "DMSO_Calib",
    12: "Blank",
    15: "Sample",
}

cycles = pulse_serie.iter("Cycle")  # iterator
sample_idx_to_guid = {}
for cycle in cycles:
    if int(cycle.find("Index").text) in example_sample_idxs:
        idx = cycle.find("Index").text
        guid = cycle.find("Guid").text
        print(f"Sample: Index: {idx}, GUID: {guid}")
        sample_idx_to_guid[idx] = guid

example_dsets_to_print = [
    get_h5_data_for_guid(guid) for guid in sample_idx_to_guid.values()
]

data_labels = [
    "Time",
    "TLD Temp",
    "TFC Temp",
    "Tboard Temp",
    "ITEC LD Current",
    "ITEC FC Current",
    "ILD Current",
    "FC1",
    "FC1_Amp",
    "FC2",
    "FC2_Amp",
    "FC3",
    "FC3_Amp",
    "FC4",
    "FC4_Amp",
]
pd.options.display.float_format = "{:.3f}".format
for isample in example_sample_idxs:
    dset = get_h5_data_for_guid(sample_idx_to_guid[str(isample)])
    df = pd.DataFrame([dset[n][:] for n in range(0, 15)], index=data_labels).transpose()
    df.style.format(precision=3)
    df.to_csv(
        "cycle_" + str(isample) + "_" + example_sample_idxs[isample] + "_raw_data.csv"
    )

    print(df)
    # print([dset[n][0] for n in range(0,15)])
