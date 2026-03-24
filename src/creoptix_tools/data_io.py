import numpy as np
import scipy
import pathlib
import xml.etree.ElementTree as ET
import h5py  # pandas HDF5 support is limited to specific structures -- need h5py
import copy

CREOPTIX_TIME_STEP = 0.1  # timestep seems to be fixed by Creoptix
CREOPTIX_TIME_TO_LOAD_CHIP = 1.4  # seconds

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

CREOPTIX_SAMPLE_TYPES = [
    "Startup",
    "Sample",
    "ControlSample",
    "Blank",
    "DMSO Cal.",
    "TubingWash"
]

quality_criteria = {
    "blank_trough_depth": 0,  # pg/mm^2
    "blank_tail_height": 0,  # pg/mm^2
    "relative_response_peak_height": 5,  # pg/mm^2
}



def read_project_xml(pathstr):
    path_to_file = pathlib.Path(pathstr)
    # Remove specified encoding as this causes problems with ElementTree
    with open(path_to_file, "r") as f:
        lines = f.readlines()
    enc_string = ' encoding="utf-16"'
    if enc_string in lines[0]:
        lines[0] = lines[0].replace(enc_string, "")
        with open(path_to_file, "w") as f:
            for line in lines:
                f.write(line)
    xml_tree = ET.parse(path_to_file)
    return xml_tree


def get_h5_data_for_guid(h5_file, guid):
    datasets = [h5_file[guid + "." + str(n)] for n in range(0, 15)]
    return datasets


class CreoptixCycle:
    def __init__(
        self,
        dataset,
        type,
        cycle_num,
        guid,
        t_contact=25.0,
        t_dissociation=60.0,
        t_baseline=45.0,
        adjusted_dataset=None,
        reagent=None,
        sampleid=None
    ):
        self.dataset = dataset
        self.type = type
        self.cycle_num = int(cycle_num)
        self.guid = guid
        self.t_contact = float(t_contact)
        self.t_disssociation = float(t_dissociation)
        self.t_baseline = float(t_baseline)
        self.adjusted_dataset = adjusted_dataset
        self.reagent=reagent
        self.sampleid=sampleid
        self.binding_dict = {"FC2-1": None, "FC3-1": None, "FC4-1": None}  # to be filled in after plotting/classification
        self.response_dict = {"FC2-1": None, "FC3-1": None, "FC4-1": None}  # to be filled in after plotting/classification

        self.baseline_values = {
            "FC1": None,
            "FC2": None,
            "FC3": None,
            "FC4": None,
        }  # For each flow channel

        if self.reagent and self.type == "Sample":
            rgnt_conc_value, rgnt_conc_unit = self.reagent["Concentration"].split()
            if rgnt_conc_unit[0] == "n":  # nanomoles
                mag = -9
            elif rgnt_conc_unit[0] == "μ":   # micromoles
                mag = -6
            else:
                raise Exception(f"Unknown unit for reagent concentration: {rgnt_conc_unit}")
            self.reagent["Conc_Value"] = float(rgnt_conc_value) * 10.0**mag
            print(f"Normalised concentration: {self.reagent["Conc_Value"]}")
            

    def get_most_recent_sample_of_type(self, cycles_data, sample_type=None):
        if sample_type not in CREOPTIX_SAMPLE_TYPES:
            raise Exception(f"Unexpected most recent sample type requested ({sample_type}).")
        # cycle_num starts from 1, so these bounds will get all matches 
        # up to and including self as cycles_data starts from 0
        valids = [c for c in cycles_data[0 : self.cycle_num] if c.type == sample_type]
        if not valids:
            return None
        else:
            return valids[-1]

    def apply_corrections(self, yofftime=40.0, yavgwindow=2.0):
        self.adjusted_dataset = copy.deepcopy(self.dataset)
        # zero to value at t=0
        fcs = ["FC1", "FC2", "FC3", "FC4"]
        for fc in fcs:
            self.adjusted_dataset[fc] = (
                self.adjusted_dataset[fc] - self.adjusted_dataset[fc][0]
            )
        # apply y-offset
        for fc in fcs:
            iyoff = int(
                yofftime / CREOPTIX_TIME_STEP
            )  # should start from 0.0 anyway but just in case
            win_width = int(yavgwindow / CREOPTIX_TIME_STEP)
            iyoff_lower = iyoff - int(np.floor(0.5 * win_width))
            iyoff_higher = iyoff + int(np.floor(0.5 * win_width))
            y_offset = np.mean(
                self.adjusted_dataset[fc][iyoff_lower : iyoff_higher + 1]
            )
            self.adjusted_dataset[fc] = self.adjusted_dataset[fc] - y_offset
        # apply x-offset
        ixoff = self.t_baseline + self.t_contact
        self.adjusted_dataset["Time"] = self.adjusted_dataset["Time"] - ixoff
        # subtract the blank response
        for fc in fcs:
            pass

        return

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
        rel_delta = (
            epilogue_signal_avg - self.baseline_values[fc]
        ) / self.baseline_values[fc]
        allowed_baseline_change = thresh  # fraction of original baseline
        if np.abs(rel_delta) > allowed_baseline_change:
            return False, rel_delta
        else:
            return True, rel_delta

    def tail_linreg(self, fc="FC1", epilogue_time=20.0):
        """Fits a linear-regression line of best fit to epilogue of sensogram to determine if tail is flat"""
        if self.type == "TubingWash":
            # No concept of baseline for TubingWash
            return False
        dt = CREOPTIX_TIME_STEP
        it1 = int(np.floor(epilogue_time / dt))
        lin_reg = scipy.stats.linregress(
            self.adjusted_dataset["Time"][it1:], self.adjusted_dataset[fc][it1:]
        )
        return lin_reg

    # This needs updating to work with a window (i.e. slice) of dataset
    def find_peaks_and_troughs(
        self,
        fc="FC1",
    ):
        peaks, peak_props = scipy.signal.find_peaks(self.dataset[fc], prominence=1)
        # cwt_peaks = scipy.signal.find_peaks_cwt(nparray[:,-1],np.arange(1,50))
        return peaks, peak_props


def ingest_project_data(xml_path, h5_path):
    cycles = []
    xml = read_project_xml(xml_path)
    root = xml.getroot()
    for serie in root.iter("Serie"):
        if serie.find("Type").text == "Pulse":
            pulse_serie = serie
            break
    serie_cycles = pulse_serie.iter("Cycle")  # iterator

    with h5py.File(h5_path, "r") as h5_file:
        for cycle in serie_cycles:
            cycle_type = cycle.find("CycleType").text
            cycle_number = cycle.find("Index").text
            guid = cycle.find("Guid").text
            contact = cycle.find("ContactTime").text
            dissociation = cycle.find("TimeAfterInjection").text
            baseline = cycle.find("BaselineDuration").text
            reagent_ref = cycle.find("ReagentRef")
            sampleid = cycle.find("Name").text.split(" - ")[2]

            dataset = get_h5_data_for_guid(h5_file, guid)
            data_to_keep = ["Time", "FC1", "FC2", "FC3", "FC4"]
            # cols_to_keep = [HDF5_CYCLE_DSET_COLS[lab] for lab in data_labels]
            cycle_data = {
                lab: dataset[HDF5_CYCLE_DSET_COLS[lab]][:] for lab in data_to_keep
            }
            # Add reagent data
            side = reagent_ref.find("AutosamplerLocation").get("Side")
            slot = reagent_ref.find("AutosamplerLocation").get("Slot")
            for rgnt in pulse_serie.findall("AutosamplerVM/"+side+"Rack/Reagents/Reagent"):
                if rgnt.attrib["Slot"] == slot:
                    reagent_info = copy.copy(rgnt.attrib)

            cycles.append(
                CreoptixCycle(
                    cycle_data,
                    cycle_type,
                    cycle_number,
                    guid,
                    t_contact=float(contact),
                    t_dissociation=float(dissociation),
                    t_baseline=float(baseline),
                    reagent=reagent_info,
                    sampleid=sampleid
                )
            )
    return cycles
