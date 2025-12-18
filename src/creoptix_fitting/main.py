import numpy as np
import pandas as pd
import scipy
import pathlib
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import h5py  # pandas HDF5 support is limited to specific structures -- need h5py

CREOPTIX_TIME_STEP = 0.1  # timestep seems to be fixed by Creoptix

HDF5_CYCLE_DSET_COLS = {
    "Time": 0,
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
    "FC3": 11,
    "FC3_amp": 12,
    "FC4": 13,
    "FC4_amp": 14,
}

sensogram_quality_criteria = {
    "blank_trough_depth": 0,  # pg/mm^2
    "blank_tail_height": 0,  # pg/mm^2
    "relative_response_peak_height": 5,  # pg/mm^2
}


class CreoptixCycle:
    def __init__(
        self,
        dataset,
        type,
        cycle_num,
        guid,
        t_injection=25.0,
        t_dissociation=60.0,
        t_baseline=45.0,
    ):
        self.dataset = dataset
        self.type = type
        self.cycle_num = int(cycle_num)
        self.guid = guid
        self.t_injection = float(t_injection)
        self.t_disssociation = float(t_dissociation)
        self.t_baseline = float(t_baseline)

        self.baseline_values = {
            "FC1": None,
            "FC2": None,
            "FC3": None,
            "FC4": None,
        }  # For each flow channel

    def calculate_baselines(self):
        if self.type == "TubingWash":
            # No concept of baseline for TubingWash
            return
        for fc in self.baseline_values.keys():
            dt = CREOPTIX_TIME_STEP
            iend = int(np.floor(self.t_baseline / dt))
            self.baseline_values[fc] = np.mean(self.dataset[fc][0:iend])
        return

    def returned_to_baseline(self, fc="FC1", epilogue_time=10.0, thresh=0.1):
        if self.type == "TubingWash":
            # No concept of baseline for TubingWash
            return False
        dt = CREOPTIX_TIME_STEP
        epilogue_len = int(np.floor(epilogue_time / dt))
        epilogue_signal_avg = np.mean(self.dataset[fc][-epilogue_len:])
        # print(epilogue_signal_avg,"  vs  ",self.baseline_values[fc])
        rel_delta = (
            epilogue_signal_avg - self.baseline_values[fc]
        ) / self.baseline_values[fc]
        allowed_baseline_change = thresh  # fraction of original baseline
        if np.abs(rel_delta) > allowed_baseline_change:
            return False
        else:
            return True

    def has_flat_tail(self, fc="FC1", epilogue_time=20.0, slope_thresh=0.02):
        if self.type == "TubingWash":
            # No concept of baseline for TubingWash
            return False
        dt = CREOPTIX_TIME_STEP
        # # simple measure of slope between epilogue start and end points
        it1 = int(np.floor(epilogue_time / dt))
        it2 = -1
        # epilogue_dy = (self.dataset[fc][it2] - self.dataset[fc][it1])  # should be -ve or 0
        # # is epilogue_dx useful?
        # epilogue_rel_slope = epilogue_dy/self.dataset[fc][it1]
        # # Linear regression
        lin_reg = scipy.stats.linregress(
            self.dataset["Time"][it1:], self.dataset[fc][it1:]
        )
        print("LIN_REG slope", lin_reg.slope)
        if -slope_thresh < lin_reg.slope < slope_thresh:
            return True
        else:
            return False

    def find_peaks_and_troughs(self, fc="FC1"):
        peaks, peak_props = scipy.signal.find_peaks(self.dataset[fc], prominence=1)
        # cwt_peaks = scipy.signal.find_peaks_cwt(nparray[:,-1],np.arange(1,50))
        return peaks, peak_props


def read_project_xml(pathstr):
    path_to_file = pathlib.Path(pathstr)
    xml_tree = ET.parse(path_to_file)
    return xml_tree


def get_h5_data_for_guid(h5_file, guid):
    datasets = [h5_file[guid + "." + str(n)] for n in range(0, 15)]
    return datasets


def get_signal_tail_mean(df, end_fraction=0.1):
    i_end = int(np.floor(end_fraction * len(df)))
    signal_tail_mean = np.mean(df.iloc[:, -1][-i_end:])
    return signal_tail_mean


def plot_sensogram(cycle, blank_cycle=None, diff=True, labels=None, name=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    tdata = cycle.dataset["Time"]
    ydata1 = cycle.dataset["FC1"]
    ydata2 = cycle.dataset["FC2"]

    window = min(
        len(ydata1), len(ydata2), len(blank_cycle.dataset["FC1"])
    )  # get shorter length, assume start at 0
    if blank_cycle:
        ydata1 = ydata1[:window] - blank_cycle.dataset["FC1"][:window]
        ydata2 = ydata2[:window] - blank_cycle.dataset["FC2"][:window]
        tdata = tdata[:window]

    if diff:
        ax1.plot(tdata, ydata2 - ydata1, label="Cycle ID: " + str(cycle.cycle_num))
    else:
        ax1.plot(tdata, ydata1, label="Cycle ID: " + str(cycle.cycle_num) + " FC1")
        ax1.plot(tdata, ydata2, label="Cycle ID: " + str(cycle.cycle_num) + " FC2")
    peak_data1, _ = cycle.find_peaks_and_troughs(fc="FC1")
    peak_data2, _ = cycle.find_peaks_and_troughs(fc="FC2")
    print(peak_data1)
    ax1.scatter(tdata[peak_data1], ydata1[peak_data1], marker="o", color="red")
    ax1.scatter(tdata[peak_data2], ydata2[peak_data2], marker="o", color="green")

    rtb, flat_tail = classify_sensogram(cycle.cycle_num, cycle.dataset)
    quality = "Good" if rtb and flat_tail else "Bad"

    quality_text = f"{quality} \n Baseline? {rtb} \n Flat tail? {flat_tail}"

    # these are matplotlib.patch.Patch properties
    color = {"Good": "palegreen", "Bad": "lightcoral"}
    props = dict(boxstyle="round", facecolor=color[quality], alpha=0.5)
    ax1.text(
        0.05,
        0.95,
        quality_text,
        transform=ax1.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    ax1.legend()
    plotfile = f"cycle{cycle.cycle_num}_{name}.png"
    plt.title(f"cycle{cycle.cycle_num}_{name}")
    plt.savefig(plotfile, bbox_inches="tight")
    plt.close()
    return


def classify_sensogram(cycle_id, dataset, channelA="FC1", channelB="FC2", name=None):
    rtb = my_cycles[cycle_id - 1].returned_to_baseline()
    flat_tail = my_cycles[cycle_id - 1].has_flat_tail()
    return rtb, flat_tail


# nparray = data_to_plot.to_numpy()
# dRbydt = np.gradient(nparray[:,-1],nparray[:,-2])
# # plt.plot(nparray[:,-2],dRbydt)

# peaks, _ = scipy.signal.find_peaks(nparray[:,-1],prominence=1)
# cwt_peaks = scipy.signal.find_peaks_cwt(nparray[:,-1],np.arange(1,50))
# plt.plot(nparray[peaks,-2], nparray[peaks,-1], "x", color="green")
# #plt.plot(nparray[cwt_peaks,-2], nparray[cwt_peaks,-1], "o", color="red")
# plt.show()


def ingest_project_data(xml_path, h5_path):
    cycles = []
    xml = read_project_xml(xml_path)
    root = xml.getroot()
    for serie in root.iter("Serie"):
        if serie.find("Type").text == "Pulse":
            pulse_serie = serie
            break
    serie_cycles = pulse_serie.iter("Cycle")  # iterator
    sample_idx_to_guid = {}
    with h5py.File(h5_path, "r") as h5_file:
        for cycle in serie_cycles:
            cycle_type = cycle.find("CycleType").text
            cycle_number = cycle.find("Index").text
            guid = cycle.find("Guid").text
            injection = cycle.find("ContactTime").text
            dissociation = cycle.find("TimeAfterInjection").text
            baseline = cycle.find("BaselineDuration").text

            dataset = get_h5_data_for_guid(h5_file, guid)
            data_to_keep = ["Time", "FC1", "FC2", "FC3", "FC4"]
            # cols_to_keep = [HDF5_CYCLE_DSET_COLS[lab] for lab in data_labels]
            cycle_data = {
                lab: dataset[HDF5_CYCLE_DSET_COLS[lab]][:] for lab in data_to_keep
            }

            cycles.append(
                CreoptixCycle(
                    cycle_data,
                    cycle_type,
                    cycle_number,
                    guid,
                    t_injection=float(injection),
                    t_dissociation=float(dissociation),
                    t_baseline=float(baseline),
                )
            )
    return cycles


xml_path = pathlib.Path("data/example/_project.cx3")
h5_path = pathlib.Path("data/example/cyclesData.h5")
my_cycles = ingest_project_data(xml_path, h5_path)
print(my_cycles[0].dataset)


# with h5py.File(path_to_file, "r") as h5_file:
#     for cycle in serie_cycles:
#         cycle_type = cycle.find("CycleType").text
#         cycle_number = cycle.find("Index").text
#         guid = cycle.find("Guid").text
#         injection = cycle.find("ContactTime").text
#         dissociation = cycle.find("TimeAfterInjection").text
#         baseline = cycle.find("BaselineDuration").text

#         dataset = get_h5_data_for_guid(h5_file, guid)
#         cols_to_keep = [HDF5_CYCLE_DSET_COLS["Time"],HDF5_CYCLE_DSET_COLS["FC1"],HDF5_CYCLE_DSET_COLS["FC2"],HDF5_CYCLE_DSET_COLS["FC3"],HDF5_CYCLE_DSET_COLS["FC4"]]
#         data_labels = list(HDF5_CYCLE_DSET_COLS.keys())[c for c in cols_to_keep]
#         cycle_data = pd.DataFrame(
#             [dataset[n][:] for n in cols_to_keep], columns=data_labels
#         )

#         my_cycles.append(
#             CreoptixCycle(
#                 cycle_type,
#                 cycle_data,
#                 cycle_number,
#                 guid,
#                 t_injection=float(injection),
#                 t_dissociation=float(dissociation),
#                 t_baseline=float(baseline),
#             )
#         )

for cyc in my_cycles[0:30]:
    cyc.calculate_baselines()
    print(cyc.cycle_num, " returned to baseline?  ", cyc.returned_to_baseline())
    print(cyc.cycle_num, " has flat tail?  ", cyc.has_flat_tail())
    most_recent_blank = max(
        [int(c.cycle_num) for c in my_cycles[0 : cyc.cycle_num] if c.type == "Blank"],
        default=None,
    )
    if most_recent_blank:
        # my_cycles indexed from 0 but cycle_num indexed from 1
        blank_cycle = my_cycles[most_recent_blank - 1]
    else:
        blank_cycle = None
    print(cyc.cycle_num, "has Most recent blank :", most_recent_blank)
    if cyc.type != "Sample":
        continue  # don't plot
    plot_sensogram(
        cyc,
        name=cyc.type.replace(" ", "_").strip("."),
        # blank_dataset = None,
        blank_cycle=blank_cycle,
        diff=False,
    )

exit()
# plt.legend(loc='upper left')
# plt.show()

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

with h5py.File(path_to_file, "r") as h5_file:
    example_dsets_to_print = [
        get_h5_data_for_guid(h5_file, guid) for guid in sample_idx_to_guid.values()
    ]

    data_labels = HDF5_CYCLE_DSET_COLS.keys()
    pd.options.display.float_format = "{:.3f}".format
    for isample in example_sample_idxs:
        dset = get_h5_data_for_guid(h5_file, sample_idx_to_guid[str(isample)])
        df = pd.DataFrame(
            [dset[n][:] for n in range(0, 15)], index=data_labels
        ).transpose()
        df.style.format(precision=3)
        # df.to_csv("cycle_"+str(isample)+"_"+example_sample_idxs[isample]+"_raw_data.csv")
        print(df)
    # print([dset[n][0] for n in range(0,15)])

    example_dsets_to_plot = example_dsets_to_print[0:10]
    for d, dset in enumerate(example_dsets_to_plot):
        plot_sensogram(dataset, sampleid=d, name="Test2")
