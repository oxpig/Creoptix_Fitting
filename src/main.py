import numpy as np
import pandas as pd
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
data_dir = pathlib.Path("C:/Users/stat0524-admin/Creoptix_Fitting/data/example/")
xml_path = pathlib.Path(data_dir / "_project.cx3")
h5_path = pathlib.Path(data_dir / "cyclesData.h5")
cycles_data = ingest_project_data(xml_path, h5_path)

# cycles_data = cycles_data[0:30]  # CRT DEBUG

t_start = time.time()

results_path = pathlib.Path("results_" + project_name)
results_path.mkdir(exist_ok=True)

cycle_binding_type = {}

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
        _ = plot_sensogram(
            cyc,
            name=cyc.type.replace(" ", "_").strip("."),
            diff=True,
            directory=results_path,
        )
    else:
        binding_type = plot_sensogram(
            cyc,
            name=cyc.type.replace(" ", "_").strip("."),
            correct_for_blank=most_recent_blank,
            diff=True,
            directory=results_path,
        )
        cycle_binding_type[cyc.cycle_num] = binding_type


t_end = time.time()

n_samples = [c for c in cycles_data if c.type in ["Blank", "Sample", "ControlSample"]]
print(f"Plotted sensograms for {len(n_samples)} cycles in {t_end - t_start} seconds.")

# Create a summary table of binding type for each cycle
result_df = pd.DataFrame(
    [
        [i, cycles_data[i - 1].type, cycle_binding_type[i]]
        for i in cycle_binding_type.keys()
    ]
)
result_df.to_csv(
    pathlib.Path(results_path / "binding_table.csv"),
    header=["CycleID", "CycleType", "BindingType"],
    index=False,
)

print(f"Of {len(n_samples)} samples:")
for bind_type in set(cycle_binding_type.values()):
    cases = [i for i in cycle_binding_type.keys() if cycle_binding_type[i] == bind_type]
    print(f"    {bind_type} : {len(cases)}")
print(f"See binding_table.csv alongside sensogram plots for more details.")
exit()
