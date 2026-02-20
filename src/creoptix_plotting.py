import pandas as pd
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
    cycles_data = ingest_project_data(xml_path, h5_path)

    cycle_binding_type = {}
    for cyc in cycles_data:
        cyc.calculate_baselines()
        cyc.apply_corrections()
        most_recent_blank = cyc.get_most_recent_sample_of_type(cycles_data, type="Blank")
        # if last_blank:
        #     # cycles_data indexed from 0 but cycle_num indexed from 1
        #     blank_cycle = cycles_data[most_recent_blank - 1]
        # else:
        #     blank_cycle = None

        # print(cyc.cycle_num, " returned to baseline?  ", cyc.returned_to_baseline())
        # print(cyc.cycle_num, " has flat tail?  ", cyc.has_flat_tail())
        if cyc.type in ["Startup", "TubingWash"]:
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

    n_samples = [
        c for c in cycles_data if c.type in ["Blank", "`Sample`", "ControlSample"]
    ]
    print(
        f"Plotted sensograms for {len(n_samples)} cycles in {t_end - t_start} seconds."
    )

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
        cases = [
            i for i in cycle_binding_type.keys() if cycle_binding_type[i] == bind_type
        ]
        print(f"    {bind_type} : {len(cases)}")
    print("See binding_table.csv alongside sensogram plots for more details.")
    print("The following cycles have been classifed as BINDING:")
    binding_cycles = [
        i for i, binding in cycle_binding_type.items() if binding == "Binding"
    ]
    for i in binding_cycles:
        print(f"    {i:5d}")
    print("Removing temporary data files...")
    remove_working_files(data_path=data_path)
    return cycle_binding_type


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
