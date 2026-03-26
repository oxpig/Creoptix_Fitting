# Package for plotting Creoptix sensograms

## Installation
The following assumes installation on a NoMachine VM:
1. [_only necessary first time_] Install uv (a Python package and virtual environment manager): \
  `curl -LsSf https://astral.sh/uv/install.sh | sh`

2. [_only necessary first time or when code is updated_] Pull the latest copy of Creoptix_Fitting from the GitHub: \
   https://github.com/oxpig/Creoptix_Fitting
   - If doing this by downloading a ZIP, extract the code to the appropriate directory on the NoMachine (should be something like \
   `/users/<username>/Creoptix_Fitting-main/`)

**The following steps need to be run within the directory (folder) where Creoptix_Fitting is located**, i.e. the folder containing the pyproject.toml file.

3. [_only necessary first time or when code is updated_] Ensure uv has synchronised (updated) any dependencies required by the Creoptix_Fitting package, which might change with a code update, by running\
`uv sync`
   - If uv complains about the name of the LICENCE file, copy it to LICENSE as well (e.g. `cp LICENCE LICENSE` or use the file browser and right-click menus to copy).
4. Run the Jupyter notebook server within this environment\
`uv run --with jupyter jupyter lab`
5. Within the Jupyter notebook server interface (should open in browser window after previous step), open\
`Creoptix_Fitting-main/creoptix_plotting.ipynb`
6. Change file paths/names within the creoptix_plotting.ipynb notebook as appropriate, remembering that NoMachine works as if a Linux host, not Windows,
so does not understand the meaning of e.g. the V: drive -- e.g. if a Creoptix project file to be analysed is in `V://<username>/20260202` on Windows,
then the path must be given in the notebook as `/users/<username>/20260202`.  If confused, navigate to the appropriate folder within NoMachine to confirm and copy the path from the file browser there.


## Known Issues
- A Creoptix project with no data in any of the 3 active channels (FC2, FC3 or FC4) will have no valid blank cycle for blank corrections as the blank cycle for the missing channel(s) will most likely be rejected (rejecting of 3 blanks for all channels for a given cycle)


