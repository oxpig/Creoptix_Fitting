## _WORKFLOW for Windows users_:

Make sure that the current directory follows the strucure below:
my_creoptix_directory/
│							# The virtual environment directory
├── Data_Folder_1
│   ├── zipped_project_file_1.cxw
│   ├── zipped_project_file_2.cxw
│   ├── ...							# All the zipped_project_file (\*.cxw) of Data_Folder_1 
│   ├── zipped_project_file_n.cxw
├── Data_Folder_2
│   ├── zipped_project_file_1.cxw
│   ├── zipped_project_file_2.cxw
│   ├── ...							# All the zipped_project_file (\*.cxw) of Data_Folder_2
│   ├── zipped_project_file_n.cxw
├── ...
├── Data_Folder_n
│   ├── zipped_project_file_1.cxw
│   ├── zipped_project_file_2.cxw
│   ├── ...							# All the zipped_project_file (\*.cxw) of Data_Folder_n
│   ├── zipped_project_file_n.cxw
│   
├── run_creoptix_plotting.py		# The python script running `perform_creoptix_fitting` for plotting
├── run_creoptix_plotting.bat		# The batch script running `run_creoptix_plotting.py` on a Windows computer (just need to double click on the file)
├── README.md						# This README

# 1. Clone the repo and create the python virtual environment (only need to do it once):

If .venv is not in the directory, follow the procedure below:

- Clone the GitHub repository (ideally in another directory)
Open a command prompt, cd to de chosen directory (e.g. `cd C:\Users\_USERNAME_\Documents`) and run `git clone https://github.com/oxpig/Creoptix_Fitting.git`

- Dowload uv
`winget install --id=astral-sh.uv  -e` 

- Create the virtual environment
cd to the GitHub directory (e.g. `cd C:\Users\_USERNAME_\Documents\Creoptix_Fitting`) & run `uv sync`.
	- If uv complains about the name of the LICENSE file, copy it to LICENCE as well (e.g. `copy LICENSE LICENCE` or use the file browser and right-click menus to copy).
	- run `uv sync` again

- Make sure that the .venv path in `run_creoptix_plotting.bat` is correct (e.g. change _USERNAME_ to your username in `call C:\Users\_USERNAME_\Documents\Git\Creoptix_Fitting\.venv\Scripts\activate.bat >> run_creoptix_plotting.log 2>&1` line 10)

# 2. Run Creoptix plotting script

Once your virtual environment (.venv) is set you can run the following commands depending on which system you are using:
Run `run_creoptix_plotting.bat` by double clicking on the file (make sure the file is in your Creoptix project directory and that your directory has the right structure - see above).

# 3. Check if your run was successful
Make sure that you find the following files after the run:
- A "Plotting_Results_zipped_project_file" directory should be created in each Data_Folder_\* directories
	- Each "Plotting_Results_zipped_project_file" should contain all the plots (saved as \*.png) and a "binding_table.csv" file
- A "processed_file.csv" file should be created into your Creoptix project directory and should contain the path to all the \*.cxw files that has been processed (this file is important as it keeps track of all the files that has been processed, so one can add more data into the Creoptix project directory, run the same script, and only the new data will be processed).
- A "run_creoptix_plotting.log" files file should be created into your Creoptix project directory and should contain the stdout of the script (this is where you can see if any errors have occured).

# 4. Keep Creoptix_Fitting and .venv up to date
Regurlarly check if your Creoptix_Fitting is up to date:
- cd to the Git directory `cd C:\Users\_yourfedID_\Documents\Creoptix_Fitting`, then run `git pull`
- update .venv if needed with `uv sync`
