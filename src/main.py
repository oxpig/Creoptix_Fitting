import numpy as np
import scipy
import pathlib
import matplotlib.pyplot as plt
import time

from creoptix_tools.data_io import ingest_project_data

def plot_sensogram(
    cycle, correct_for_blank=False, blank_cycle=None, diff=True, labels=None, name=None
):
    if correct_for_blank and not blank_cycle:
        raise Exception("correct_for_blank requested but no blank cycle provided!")
    if cycle.adjusted_dataset and blank_cycle.adjusted_dataset:
        dataset = cycle.adjusted_dataset
        blank_dataset = blank_cycle.adjusted_dataset
    else:
        dataset = cycle.dataset
        blank_dataset = blank_cycle.dataset
    fig = plt.figure()
    if diff:
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    else:
        ax1 = fig.add_subplot(1, 1, 1)
    # assuming two relevant flow channels are 1 and 2
    tdata = dataset["Time"]
    ydata1 = dataset["FC1"]
    ydata2 = dataset["FC2"]

    window = min(
        len(ydata1), len(ydata2), len(blank_dataset["FC1"])
    )  # get shorter length, assume start at 0
    if correct_for_blank and blank_cycle:
        ydata1 = ydata1[:window] - blank_dataset["FC1"][:window]
        ydata2 = ydata2[:window] - blank_dataset["FC2"][:window]
        tdata = tdata[:window]

    ax1.plot(tdata, ydata1, label="Cycle ID: " + str(cycle.cycle_num) + " FC1")
    ax1.plot(tdata, ydata2, label="Cycle ID: " + str(cycle.cycle_num) + " FC2")
    peak_data1, _ = scipy.signal.find_peaks(ydata1, prominence=1)
    peak_data2, _ = scipy.signal.find_peaks(ydata2, prominence=1)
    ax1.scatter(
        tdata[peak_data1],
        ydata1[peak_data1],
        facecolors="none",
        marker="o",
        color="red",
    )
    ax1.scatter(
        tdata[peak_data2],
        ydata2[peak_data2],
        facecolors="none",
        marker="o",
        color="green",
    )
    if diff:
        ax2.plot(
            tdata,
            ydata2 - ydata1,
            label="Cycle ID: " + str(cycle.cycle_num) + " FC(2-1)",
            color="black",
        )

    rtb, flat_tail = classify_sensogram(cycle)
    quality = "Good" if rtb[0] and flat_tail[0] else "Bad"

    quality_text = f"{quality} \n Baseline? {rtb[0]} \n Flat tail? {flat_tail[0]}"
    # Plot lin reg of tail for FC(2-1)
    LR_result = scipy.stats.linregress(tdata[-200:], ydata2[-200:]-ydata1[-200:])
    ax2.plot(
        tdata[-200:],
        LR_result.intercept + LR_result.slope * tdata[-200:],
        color="magenta",
        linestyle="--",
    )

    # these are matplotlib.patch.Patch properties
    color = {"Good": "palegreen", "Bad": "lightcoral"}
    props = dict(boxstyle="round", facecolor=color[quality], alpha=0.5)
    # ax1.text(
    #     0.05,
    #     0.95,
    #     quality_text,
    #     transform=ax1.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )

    ax1.legend()
    ax2.legend()
    plotfile = f"cycle{cycle.cycle_num}_{name}.png"
    fig.suptitle(f"cycle{cycle.cycle_num}_{name}")
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close()
    return


def classify_sensogram(cycle, channelA="FC1", channelB="FC2", name=None):
    rtb = cycle.returned_to_baseline()
    flat_tail = cycle.has_flat_tail()
    return rtb, flat_tail


# nparray = data_to_plot.to_numpy()
# dRbydt = np.gradient(nparray[:,-1],nparray[:,-2])
# # plt.plot(nparray[:,-2],dRbydt)

# peaks, _ = scipy.signal.find_peaks(nparray[:,-1],prominence=1)
# cwt_peaks = scipy.signal.find_peaks_cwt(nparray[:,-1],np.arange(1,50))
# plt.plot(nparray[peaks,-2], nparray[peaks,-1], "x", color="green")
# #plt.plot(nparray[cwt_peaks,-2], nparray[cwt_peaks,-1], "o", color="red")
# plt.show()


xml_path = pathlib.Path("data/example/_project.cx3")
h5_path = pathlib.Path("data/example/cyclesData.h5")
cycles_data = ingest_project_data(xml_path, h5_path)

cycles_data = cycles_data[0:50]  # CRT DEBUG

t_start = time.time()

for cyc in cycles_data:
    cyc.calculate_baselines()
    cyc.apply_corrections()
    most_recent_blank = cyc.get_most_recent_blank(cycles_data)
    # if last_blank:
    #     # cycles_data indexed from 0 but cycle_num indexed from 1
    #     blank_cycle = cycles_data[most_recent_blank - 1]
    # else:
    #     blank_cycle = None

    # print(cyc.cycle_num, " returned to baseline?  ", cyc.returned_to_baseline())
    # print(cyc.cycle_num, " has flat tail?  ", cyc.has_flat_tail())
    if cyc.type in ["Startup", "DMSO Cal.", "TubingWash"]:
        continue  # don't plot
    plot_sensogram(
        cyc,
        name=cyc.type.replace(" ", "_").strip("."),
        correct_for_blank=True,
        blank_cycle=most_recent_blank,
        diff=True,
    )

t_end = time.time()

print(f"Plotted sensograms for {len([c for c in cycles_data if c.type in ["Blank","Sample","ControlSample"]])} cycles in {t_end-t_start} seconds.")

exit()
# # plt.legend(loc='upper left')
# # plt.show()

# example_sample_idxs = {
#     1: "Startup",
#     11: "DMSO_Calib",
#     12: "Blank",
#     15: "Sample",
# }

# cycles = pulse_serie.iter("Cycle")  # iterator
# sample_idx_to_guid = {}
# for cycle in cycles:
#     if int(cycle.find("Index").text) in example_sample_idxs:
#         idx = cycle.find("Index").text
#         guid = cycle.find("Guid").text
#         print(f"Sample: Index: {idx}, GUID: {guid}")
#         sample_idx_to_guid[idx] = guid

# with h5py.File(path_to_file, "r") as h5_file:
#     example_dsets_to_print = [
#         get_h5_data_for_guid(h5_file, guid) for guid in sample_idx_to_guid.values()
#     ]

#     data_labels = HDF5_CYCLE_DSET_COLS.keys()
#     pd.options.display.float_format = "{:.3f}".format
#     for isample in example_sample_idxs:
#         dset = get_h5_data_for_guid(h5_file, sample_idx_to_guid[str(isample)])
#         df = pd.DataFrame(
#             [dset[n][:] for n in range(0, 15)], index=data_labels
#         ).transpose()
#         df.style.format(precision=3)
#         # df.to_csv("cycle_"+str(isample)+"_"+example_sample_idxs[isample]+"_raw_data.csv")
#         print(df)
#     # print([dset[n][0] for n in range(0,15)])

#     example_dsets_to_plot = example_dsets_to_print[0:10]
#     for d, dset in enumerate(example_dsets_to_plot):
#         plot_sensogram(dataset, sampleid=d, name="Test2")
