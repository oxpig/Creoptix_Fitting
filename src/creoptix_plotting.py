import pandas as pd
import numpy as np
import pathlib
import zipfile
import time
import datetime
import os
from creoptix_tools.data_io import ingest_project_data
from creoptix_tools.plotting import plot_sensogram
from dataclasses import dataclass, asdict

@dataclass
class FitData:
    R2: float
    RMSE: float
    sqrt_chi2: float
    red_chi2: float

@dataclass
class RowData:
    SerieID: int
    CycleID: int
    CycleType: str
    SampleID: str
    channel: str
    BindingType: str
    Response: float
    kA: float
    kD: float
    bigKD: float
    Rmax: float
    fit: FitData

def _flatten_row(row):
    d = asdict(row)
    fit = d.pop("fit")          # remove nested dict
    return {**d, **fit}         # flatten fit into the same row

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

    data_rows = []
    
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
            for fc in ["FC1","FC2", "FC3", "FC4"]:
                cyc.get_conc_wrt_time(cycles_data,fc=fc)
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
            SerieID = cyc.subdir.split("_")[-1]  # extract series number from subdir name
            for channel in ["FC2", "FC3", "FC4"]:
                diff_channel = f"{channel}-1"
                kA = cyc.fitted_params[diff_channel][0]
                kD = cyc.fitted_params[diff_channel][1]
                if kA is not None and kD is not None:
                    bigKD = kD/ kA
                else:
                    bigKD = None
                if cyc.fitting_metrics[diff_channel] is not None:
                    R2 = cyc.fitting_metrics[diff_channel]["R2"]
                    RMSE = cyc.fitting_metrics[diff_channel]["RMSE"]
                    sqrt_chi2 = cyc.fitting_metrics[diff_channel]["Sqrt(Chi2)"]
                    red_chi2 = cyc.fitting_metrics[diff_channel]["Chi2_red"]
                else: # no fitting done
                    R2 = None
                    RMSE = None
                    sqrt_chi2 = None
                    red_chi2 = None
                fit_data = FitData(R2,RMSE,sqrt_chi2,red_chi2)

                data_rows.append(RowData(
                        SerieID=SerieID,
                        CycleID=cyc.cycle_num,
                        CycleType=cyc.type,
                        SampleID=cyc.sampleid,
                        channel=diff_channel,
                        BindingType=cyc.binding_values[diff_channel],
                        Response=int(np.floor(cyc.response_values[diff_channel])),
                        kA=kA,
                        kD=kD,
                        bigKD=bigKD,
                        Rmax=cyc.fitted_params[diff_channel][2],
                        fit=fit_data
                    )
                )


    t_end = time.time()

    print(
        f"Plotted sensograms for {n_samples} cycles in {t_end - t_start} seconds."
    )

    # Export a summary table of binding type and response for each channel of each cycle
    # result_df = pd.DataFrame(data_rows)
    result_df = pd.DataFrame(_flatten_row(r) for r in data_rows)
    
    result_df.to_csv(
        pathlib.Path(results_path /  "binding_table.csv"),
        index=False,
    )

    # fitting_df.to_csv(
    #     pathlib.Path(results_path /  "fitting_table.csv"),
    #     index=False,
    # )

    print(f"Of {n_samples} samples:")
    for bind_type in set([row.BindingType for row in data_rows]):
        cycle_ids = (
            result_df.groupby("CycleID")
            .filter(lambda g: g["BindingType"].eq(bind_type).all())["CycleID"]
            .unique()
            .tolist()
        )
        print(f"    {bind_type} in all 3 active channels : {len(cycle_ids)}")
    print("See binding_table.csv alongside sensogram plots for more details.")

    print("The following cycles have at least one active channel classifed as BINDING:")
    binding_cycle_ids = (
        result_df.groupby("CycleID")
        .filter(lambda g: g["BindingType"].eq("Binding").any())["CycleID"]
        .unique()
        .tolist()
    )
    for i in binding_cycle_ids:
        print(f"    {i:5d}")
    print("Removing temporary data files...")
    remove_working_files(data_path=data_path)
    return data_rows


def main():
    project_name = "TestProject"
    data_dir = "C:/Users/stat0524-admin/Creoptix_Fitting/data/example_4FC/"
    results_name = "PlottingResults"
    perform_creoptix_fitting(
        "C:\\Users\\stat0524-admin\\Creoptix_Fitting\\data\\example_4FC\\20260323_EV71 2A Binding assay.cxw",
        #         "C:\\Users\\stat0524-admin\\Creoptix_Fitting\\data\\20250826_DENV-2 NS2B3 Binding Assay.cxw",
        project_name=project_name, data_dir=data_dir, results_dir=results_name
    )


if __name__ == "__main__":
    main()
