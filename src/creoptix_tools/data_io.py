import numpy as np
import scipy
import pathlib
import xml.etree.ElementTree as ET
import h5py  # pandas HDF5 support is limited to specific structures -- need h5py
import copy
from collections import namedtuple

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

Pulse = namedtuple("Pulse",["duration","gap"])

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


def blank_is_valid(cycle, neg_thresh=-5, max_thresh=50):
    assert cycle.type == "Blank"
    fcs = ["FC2", "FC3", "FC4"]
    for fc in fcs:
        if cycle.noise_values[fc]:
            return False    # Not valid if blank channel is noisy
        dt = CREOPTIX_TIME_STEP
        ibaseline_end = int(np.floor(cycle.t_baseline / dt))
        if not cycle.adjusted_dataset:
            cycle.apply_corrections()
        blank_dataset = cycle.adjusted_dataset[fc][ibaseline_end:] - cycle.adjusted_dataset["FC1"][ibaseline_end:]
        if np.min(blank_dataset) < neg_thresh:
            return False    # Not valid if blank has negative response after injection
        if np.max(blank_dataset) > max_thresh:
            return False    # Not valid if blank has very high response after injection
    return True


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
        conc_wrt_time=None,
        pulses=None,
        sampleid=None,
        rinse=0.0,
        subdir=None,
        acquisition_rate=None,   # acquisition rate in Hz
    ):
        self.dataset = dataset
        self.type = type
        self.cycle_num = int(cycle_num)
        self.guid = guid
        self.t_contact = float(t_contact)
        self.t_dissociation = float(t_dissociation)
        self.t_baseline = float(t_baseline)
        self.t_rinse = float(rinse)
        self.adjusted_dataset = adjusted_dataset
        self.reagent=reagent
        self.sampleid=sampleid
        self.subdir=subdir
        self.acquisition_rate=acquisition_rate
        self.conc_wrt_time=conc_wrt_time
        self.pulses=pulses
        self.dissoc_phases=None
        self.baseline_values = {
            "FC1": None,
            "FC2": None,
            "FC3": None,
            "FC4": None,
        }  # For each flow channel
        self.noise_values = {
            "FC1": None,
            "FC2": None,
            "FC3": None,
            "FC4": None}  # to track noise in each channel

        self.binding_values = {
            "FC2-1": None,
            "FC3-1": None,
            "FC4-1": None
        }  # to be filled in after plotting/classification
        self.response_values = {
            "FC2-1": None,
            "FC3-1": None,
            "FC4-1": None
        }  # to be filled in after plotting/classification
        self.fitted_params = {  # ka, kd, rmax
            "FC2-1": [None, None, None],
            "FC3-1": [None, None, None],
            "FC4-1": [None, None, None],
        }  # to be filled in after fitting

        if self.reagent and self.type == "Sample":
            rgnt_conc_value, rgnt_conc_unit = self.reagent["Concentration"].split()
            if rgnt_conc_unit[0] == "n":  # nanomoles
                mag = -9
            elif rgnt_conc_unit[0] == "μ":   # micromoles
                mag = -6
            else:
                raise Exception(f"Unknown unit for reagent concentration: {rgnt_conc_unit}")
            self.reagent["Conc_Value"] = float(rgnt_conc_value) * 10.0**mag
            ##print(f"Normalised concentration: {self.reagent["Conc_Value"]}")
    
    def get_dissociation_phases(self):
        if self.dissoc_phases:
            return
        else:
            self.dissoc_phases=[]
            t0 = 2.0
            self.pulses.append(self.t_dissociation) # final dissociation phase length
            pulses = self.pulses
            if len(pulses) % 2 != 0:
                raise Exception(f"Error, number of time points for pulses is not even."
                                "At least one dissociation phase is unbounded.")
            ti_starts = np.array([
                t0+sum(pulses[:n+1]) for n in range(0,len(pulses),2)
            ])
            ti_ends = np.array([
                t0+sum(pulses[:n+2]) for n in range(0,len(pulses),2)
            ])
            self.dissoc_phases=list(zip(ti_starts,ti_ends))
            print(f"Dissociation phases = {self.dissoc_phases}")
            return
           
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
        
    def get_blank_cycle(self, cycles_data):
        valids = [c for c in cycles_data if c.type == "Blank" and blank_is_valid(cycle=c)]
        most_recent = [c for c in valids if c.cycle_num < self.cycle_num]
        if most_recent:
            return most_recent[-1]
        else:
            if valids:
                blank = valids[0]
                print(f"Warning: no valid blank cycle found before cycle {self.cycle_num}. Using first valid blank from cycle {blank.cycle_num}.")
                return blank
            else:
                print(f"Warning: no valid blank cycles found in dataset!")
                return None            

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
    
    def check_noise(self, noise_threshold=5.0):
        # Check if the signal is noisy by looking at the standard deviation of the baseline region
        fcs = ["FC1", "FC2", "FC3", "FC4"]
        for fc in fcs:
            dt = CREOPTIX_TIME_STEP
            ibaseline_end = int(np.floor(self.t_baseline / dt))
            if self.adjusted_dataset:
                baseline_std = np.std(self.adjusted_dataset[fc][0:ibaseline_end])
            else:
                baseline_std = np.std(self.dataset[fc][0:ibaseline_end])
            if baseline_std > noise_threshold:
                self.noise_values[fc] = True
            else:
                self.noise_values[fc] = False
        return

    def get_conc_wrt_time(self,cycles_data,fc="FC2"):
        most_recent_dmso = self.get_most_recent_sample_of_type(cycles_data, sample_type="DMSO Cal.")
        input_conc = self.reagent["Conc_Value"]
        #print(f"INPUT_CONC {input_conc} and nearest DMSO {most_recent_dmso.cycle_num}")
        dmso_response = most_recent_dmso.dataset[fc]
        min_response, max_response = (min(dmso_response), max(dmso_response))
        normalised_response = (most_recent_dmso.dataset[fc]-min_response) / (max_response-min_response)
        #print(f"Max response: {max_response:.3f} so Normalised Response:{normalised_response}")
        conc_wrt_time = normalised_response * input_conc
        # print(f"Length of conc_wrt_time: {len(conc_wrt_time)}")
        self.conc_wrt_time=conc_wrt_time
        return


def ingest_project_data(xml_path, h5_path, results_path=pathlib.Path(".")):
    cycles = []
    xml = read_project_xml(xml_path)
    root = xml.getroot()
    pulse_series = []
    acquisition_rates = []
    for serie in root.iter("Serie"):
        if serie.find("Type").text == "Pulse":
            pulse_series.append(serie)
            rate = serie.find("AcquisitionRate").text
            acquisition_rates.append(rate)

    for i, ps in enumerate(pulse_series):
        subdir = pathlib.Path(results_path / f"Serie_{i+1:02d}")
        subdir.mkdir(parents=True, exist_ok=True)
        if "RAPID Kinetics: " not in ps.find("Name").text:
            print(f"Warning: pulse serie {i+1:02d} name is different from 'RAPID Kinetics: ' ({ps.find('Name').text}).")

        serie_cycles = ps.iter("Cycle")  # iterator
        try:
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
                    for marker in cycle.iter("Marker"):
                        if marker.find("Type").text == "Rinse":
                            rinse = marker.find("X").text

                    dataset = get_h5_data_for_guid(h5_file, guid)
                    data_to_keep = ["Time", "FC1", "FC2", "FC3", "FC4"]
                    # cols_to_keep = [HDF5_CYCLE_DSET_COLS[lab] for lab in data_labels]
                    cycle_data = {
                        lab: dataset[HDF5_CYCLE_DSET_COLS[lab]][:] for lab in data_to_keep
                    }

                    # add pulses data
                    pulse_data=cycle.findall("PulseDurations/double")
                    pulses = []
                    for ip,p in enumerate(pulse_data):
                        pulse_data[ip] = float(p.text)
                    print(f"Pulse data for cycle {cycle_number}: {pulse_data}")
                    # pulses = zip(pulse_data[0::2],pulse_data[1::2])
                    # pulses = [Pulse(duration=x,gap=y) for (x,y) in pulses]
                    pulses = pulse_data

                    # Add reagent data
                    side = reagent_ref.find("AutosamplerLocation").get("Side")
                    slot = reagent_ref.find("AutosamplerLocation").get("Slot")
                    for rgnt in ps.findall("AutosamplerVM/"+side+"Rack/Reagents/Reagent"):
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
                            pulses=pulses,
                            sampleid=sampleid,
                            rinse=rinse,
                            subdir=subdir.name,
                            acquisition_rate=float(acquisition_rates[i])
                        )
                    )
        except Exception as e:
            print(f"Error occurred while processing pulse serie {i+1:02d}: {e}")
    return cycles
