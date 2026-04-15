import pandas as pd
import numpy as np
import pathlib
import zipfile
import time
import datetime
import os
from creoptix_tools.data_io import ingest_project_data
from creoptix_tools.plotting import plot_sensogram


def setup_directories(
    creoptix_zipfile,
    project_name="TestProject",
    data_dir="UnzippedData",
    results_dir="PlottingResults",
):
    today = datetime.datetime.now()
    project_name = "_".join([project_name, today.strftime("%Y-%m-%d")])

    parent_path = pathlib.Path(creoptix_zipfile).parents[0]

    data_path = pathlib.Path(parent_path / (data_dir + "_" + project_name))
    data_path.mkdir(exist_ok=True)

    results_path = pathlib.Path(parent_path / (results_dir + "_" + project_name))
    results_path.mkdir(exist_ok=True)
    return data_path, results_path


def unzip_data(creoptix_zipfile, data_path):
    zf_obj = zipfile.ZipFile(creoptix_zipfile)
    members = ["cyclesData.h5", "_project.cx3"]
    zf_obj.extractall(members=members, path=data_path)
    return


def remove_working_files(data_path, filenames=["_project.cx3", "cyclesData.h5"]):
    for f in filenames:
        os.remove(pathlib.Path(data_path, f))
    os.rmdir(data_path)
    return


def perform_creoptix_fitting(
    creoptix_zipfile,
    project_name="TestProject",
    data_dir="UnzippedData",
    results_dir="PlottingResults",
):
    t_start = time.time()

    data_path, results_path = setup_directories(
        creoptix_zipfile=creoptix_zipfile,
        project_name=project_name,
        data_dir=data_dir,
        results_dir=results_dir,
    )
    unzip_data(creoptix_zipfile=creoptix_zipfile, data_path=data_path)
    xml_path = pathlib.Path(data_path / "_project.cx3")
    h5_path = pathlib.Path(data_path / "cyclesData.h5")
    cycles_data = ingest_project_data(xml_path, h5_path, results_path)

    result_data = {
        "SerieID": [],
        "CycleID": [],
        "CycleType": [],
        "SampleID": [],
        "BindingType_FC2-1": [],
        "Response_FC2-1": [],
        "kA_FC2-1": [],
        "kD_FC2-1": [],
        "bigKD_FC2-1": [],
        "Rmax_FC2-1": [],
        "BindingType_FC3-1": [],
        "Response_FC3-1": [],
        "kA_FC3-1": [],
        "kD_FC3-1": [],
        "bigKD_FC3-1": [],
        "Rmax_FC3-1": [],
        "BindingType_FC4-1": [],
        "Response_FC4-1": [],
        "kA_FC4-1": [],
        "kD_FC4-1": [],
        "bigKD_FC4-1": [],
        "Rmax_FC4-1": [],
    }
    n_samples = 0
    for cyc in cycles_data:
        if cyc.cycle_num > 19:
            break
        if cyc.type in ["Blank", "DMSO Cal.", "Sample", "ControlSample"]:
            n_samples += 1
        cyc.calculate_baselines()
        cyc.apply_corrections()
        cyc.check_noise()
        if cyc.type == "Sample":
            cyc.get_conc_wrt_time(cycles_data)
        # blank = cyc.get_most_recent_sample_of_type(cycles_data, sample_type="Blank")

        # if last_blank:
        #     # cycles_data indexed from 0 but cycle_num indexed from 1
        #     blank_cycle = cycles_data[most_recent_blank - 1]
        # else:
        #     blank_cycle = None

        # print(cyc.cycle_num, " returned to baseline?  ", cyc.returned_to_baseline())
        # print(cyc.cycle_num, " has flat tail?  ", cyc.has_flat_tail())
        if cyc.type in ["Startup", "TubingWash"]:
            continue  # don't plot
        elif cyc.type not in ["Sample", "ControlSample"]:
            plot_sensogram(
                cyc,
                name=cyc.type.replace(" ", "_").strip("."),
                diff=True,
                directory=results_path,
            )
        else:
            blank = cyc.get_blank_cycle(cycles_data)
            plot_sensogram(
                cyc,
                name=cyc.type.replace(" ", "_").strip("."),
                correct_for_blank=blank,
                diff=True,
                directory=results_path,
            )
            result_data["SerieID"].append(cyc.subdir.split("_")[-1])  # extract series number from subdir name
            result_data["CycleID"].append(cyc.cycle_num)
            result_data["CycleType"].append(cyc.type)
            result_data["SampleID"].append(cyc.sampleid)
            for channel in ["FC2", "FC3", "FC4"]:
                result_data[f"BindingType_{channel}-1"].append(cyc.binding_values[f"{channel}-1"])
                result_data[f"Response_{channel}-1"].append(int(np.floor(cyc.response_values[f"{channel}-1"])))
                ka = cyc.fitted_params[f"{channel}-1"][0]
                kd = cyc.fitted_params[f"{channel}-1"][1]
                result_data[f"kA_{channel}-1"].append(ka)
                result_data[f"kD_{channel}-1"].append(kd)
                if ka is not None and kd is not None:
                    bigKD = kd/ ka
                else:
                    bigKD = None
                result_data[f"bigKD_{channel}-1"].append(bigKD)
                result_data[f"Rmax_{channel}-1"].append(cyc.fitted_params[f"{channel}-1"][2])


    t_end = time.time()

    print(
        f"Plotted sensograms for {n_samples} cycles in {t_end - t_start} seconds."
    )

    # Export a summary table of binding type and response for each channel of each cycle
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(
        pathlib.Path(results_path /  "binding_table.csv"),
        index=False,
    )

    print(f"Of {n_samples} samples:")
    for bind_type in set(result_data["BindingType_FC2-1"]):
        cases = [
            cycle_id for i, cycle_id in enumerate(result_data["CycleID"])
            if result_data["BindingType_FC2-1"][i] == bind_type
            and result_data["BindingType_FC3-1"][i] == bind_type
            and result_data["BindingType_FC4-1"][i] == bind_type
        ]
        print(f"    {bind_type} in all 3 active channels : {len(cases)}")
    print("See binding_table.csv alongside sensogram plots for more details.")
    print("The following cycles have at least one active channel classifed as BINDING:")
    binding_cycles = [
        cycle_id for i, cycle_id in enumerate(result_data["CycleID"])
        if result_data["BindingType_FC2-1"][i] == "Binding"
        or result_data["BindingType_FC3-1"][i] == "Binding"
        or result_data["BindingType_FC4-1"][i] == "Binding"
    ]
    for i in binding_cycles:
        print(f"    {i:5d}")
    print("Removing temporary data files...")
    remove_working_files(data_path=data_path)
    return result_data


def main():
    project_name = "TestProject"
    data_dir = "C:/Users/stat0524-admin/Creoptix_Fitting/data/example/"
    results_name = "PlottingResults"
    perform_creoptix_fitting(
        "C:\\Users\\stat0524-admin\\Creoptix_Fitting\\data\\20250826_DENV-2 NS2B3 Binding Assay.cxw",
        project_name=project_name, data_dir=data_dir, results_dir=results_name
    )


if __name__ == "__main__":
    main()
