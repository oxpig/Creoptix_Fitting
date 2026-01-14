import numpy as np
import scipy
import pathlib
import matplotlib.pyplot as plt
import time
import datetime

from creoptix_tools.data_io import ingest_project_data
from creoptix_tools.plotting import plot_sensogram

# nparray = data_to_plot.to_numpy()
# dRbydt = np.gradient(nparray[:,-1],nparray[:,-2])
# # plt.plot(nparray[:,-2],dRbydt)

# peaks, _ = scipy.signal.find_peaks(nparray[:,-1],prominence=1)
# cwt_peaks = scipy.signal.find_peaks_cwt(nparray[:,-1],np.arange(1,50))
# plt.plot(nparray[peaks,-2], nparray[peaks,-1], "x", color="green")
# #plt.plot(nparray[cwt_peaks,-2], nparray[cwt_peaks,-1], "o", color="red")
# plt.show()

today = datetime.datetime.now()
project_name = "_".join(["TestProject", today.strftime("%Y-%m-%d")])
xml_path = pathlib.Path("data/example/_project.cx3")
h5_path = pathlib.Path("data/example/cyclesData.h5")
cycles_data = ingest_project_data(xml_path, h5_path)

# cycles_data = cycles_data[0:50]  # CRT DEBUG

t_start = time.time()

results_path = pathlib.Path("results_" + project_name)
results_path.mkdir(exist_ok=True)

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
    elif cyc.type == "Blank":
        plot_sensogram(
            cyc,
            name=cyc.type.replace(" ", "_").strip("."),
            diff=True,
            directory=results_path,
        )
    else:
        pass
        plot_sensogram(
            cyc,
            name=cyc.type.replace(" ", "_").strip("."),
            correct_for_blank=most_recent_blank,
            diff=True,
            directory=results_path,
        )

t_end = time.time()

print(
    f"Plotted sensograms for {len([c for c in cycles_data if c.type in ['Blank', 'Sample', 'ControlSample']])} cycles in {t_end - t_start} seconds."
)

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
