import shutil
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd

from creoptix_plotting import perform_creoptix_fitting

start = datetime.now()

files = glob("*/*.cxw")
assert len(files) > 0, (
    "No .cxw files found in subdirectories. Please check the path and try again."
)

if Path("processed_files.csv").exists():
    processed_files = pd.read_csv("processed_files.csv")
else:
    processed_files = {"FileName": [], "DateProcessed": []}
    processed_files = pd.DataFrame(processed_files)

new_processed_files = {processed_files.columns[0]: [], processed_files.columns[1]: []}

error_counts = 0
for file in files:
    if file not in processed_files["FileName"].values:
        print(f"\n\nProcessing file: {file}...\n\n")
        file_path = Path(file)
        project_name = f"{file_path.name.split('.')[0]}".replace(" ", "_")
        try:
            _ = perform_creoptix_fitting(
                file,
                project_name=f"{project_name}",
                data_dir="Unzipped_Data",
                results_dir="Plotting_Results",
            )
            new_processed_files["FileName"].append(file)
            new_processed_files["DateProcessed"].append(
                datetime.now().strftime("%d-%m-%Y (%H:%M:%S)")
            )
            print(f"\n{file} has been successfully processed.\n")
        except Exception as e:
            error_counts += 1
            print(f"\nError processing file {file}: {e}\n")
            today = datetime.now().strftime("%Y-%m-%d")
            parent_dir = Path(file).parent
            if Path(f"{parent_dir}/Unzipped_Data_{project_name}_{today}").exists():
                shutil.rmtree(
                    f"{parent_dir}/Unzipped_Data_{project_name}_{today}",
                    ignore_errors=True,
                )
    else:
        date_time = processed_files.loc[
            processed_files["FileName"] == file, "DateProcessed"
        ].values[0]
        print(f"\nFile {file} has already been processed at {date_time}. Skipping...\n")

end = datetime.now()
elapsed = end - start

print(
    f"\n\nAll new {len(files)} files have been processed in {elapsed.total_seconds():.3f} s. There were {error_counts} errors encountered (please check the .log file)!"
)

processed_files = pd.concat(
    [processed_files, pd.DataFrame(new_processed_files)], ignore_index=True
)
processed_files.to_csv("processed_files.csv", index=False)
