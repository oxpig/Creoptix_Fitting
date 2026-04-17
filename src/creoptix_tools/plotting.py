import numpy as np
import scipy
import matplotlib.pyplot as plt
import pathlib
from creoptix_tools.data_io import CREOPTIX_TIME_STEP
from .analysis import CreoptixModel, BindingParams, perform_fitting, fit_metrics

dt = CREOPTIX_TIME_STEP
DELTA_DISS = int(np.floor(1.5 / dt)) # default response in dissociation phase 1.5 s after rinse time
DELTA_BIND = int(np.floor(0.5 / dt)) # default response in binding phase 0.5 s before rinse time

def plot_sensogram(
    cycle,
    correct_for_blank=None,
    diff=True,
    labels=None,
    name=None,
    directory=None,
):
    if cycle.acquisition_rate:
        dt = 1.0 / cycle.acquisition_rate
        DELTA_DISS = int(np.floor(1.5 / dt))
        DELTA_BIND = int(np.floor(0.5 / dt))
    if cycle.adjusted_dataset:
        dataset = cycle.adjusted_dataset
    else:
        dataset = cycle.dataset
    fig = plt.figure(layout="constrained", figsize=(14, 6))
    fcs = ["FC2", "FC3", "FC4"]
    n_channels_in_cycle = sum(1 for fc in fcs if fc in dataset.keys())
    subfigs = fig.subfigures(1, n_channels_in_cycle, wspace=0.07)
    for i in range(n_channels_in_cycle):
        current_channel = fcs[i]
        tdata = dataset["Time"]
        ydata1 = dataset["FC1"]
        ydata2 = dataset[current_channel]

        window = min(len(ydata1), len(ydata2))  # get shorter length, assume start at 0
        if correct_for_blank:
            if cycle.adjusted_dataset:
                blank_dataset = correct_for_blank.adjusted_dataset
            else:
                blank_dataset = correct_for_blank.dataset
            window = min(window, len(blank_dataset["FC1"]))
            ydata1 = ydata1[:window] - blank_dataset["FC1"][:window]
            ydata2 = ydata2[:window] - blank_dataset[current_channel][:window]
            tdata = tdata[:window]
            ax1 = subfigs[i].add_subplot(3, 1, 1)
            ax2 = subfigs[i].add_subplot(3, 1, 2, sharex=ax1)
            ax3 = subfigs[i].add_subplot(3, 1, 3, sharex=ax1)
            ax3.plot(
                tdata,
                blank_dataset[current_channel][:window] - blank_dataset["FC1"][:window],
                label="Blank ID: " + str(correct_for_blank.cycle_num) + f" FC({i+2}-1)",
                color="grey",
                linestyle=":",
            )
            ax3.legend()
        else:
            if diff:
                ax1 = subfigs[i].add_subplot(2, 1, 1)
                ax2 = subfigs[i].add_subplot(2, 1, 2, sharex=ax1)
            else:
                ax1 = subfigs[i].add_subplot(1, 1, 1)


        ax1.plot(tdata, ydata1, label="Cycle ID: " + str(cycle.cycle_num) + " FC1")
        ax1.plot(tdata, ydata2, label="Cycle ID: " + str(cycle.cycle_num) + f" FC{i+2}")
        peak_data1, _ = scipy.signal.find_peaks(ydata1, prominence=1)
        peak_data2, _ = scipy.signal.find_peaks(ydata2, prominence=1)
        # ax1.scatter(
        #     tdata[peak_data1],
        #     ydata1[peak_data1],
        #     facecolors="none",
        #     marker="o",
        #     color="red",
        # )
        # ax1.scatter(
        #     tdata[peak_data2],
        #     ydata2[peak_data2],
        #     facecolors="none",
        #     marker="o",
        #     color="green",
        # )
        irinse = (int(np.floor(cycle.t_rinse / dt)))
        ax1.scatter(
            tdata[irinse+DELTA_DISS],
            ydata1[irinse+DELTA_DISS],
            facecolors="none",
            marker="o",
            color="red",
        )
        ax1.scatter(
            tdata[irinse-DELTA_BIND],
            ydata1[irinse-DELTA_BIND],
            facecolors="none",
            marker="o",
            color="red",
        )
        ax1.scatter(
            tdata[irinse-DELTA_BIND],
            ydata2[irinse-DELTA_BIND],
            facecolors="none",
            marker="o",
            color="green",
        )
        if diff:
            ax2.plot(
                tdata,
                ydata2 - ydata1,
                label="Cycle ID: " + str(cycle.cycle_num) + f" FC({i+2}-1)",
                color="black",
            )
            peak_data_diff, _ = scipy.signal.find_peaks(ydata2 - ydata1, prominence=1)
            peak_data_diff = [p - (int(np.floor(0.5 / dt))) for p in peak_data_diff]  # adjust to look for value at 0.5 s before peak
            # max_binding_response = max((ydata2 - ydata1)[peak_data_diff], default=0.0)
            max_binding_response = (ydata2 - ydata1)[irinse-DELTA_BIND]
            cycle.response_values[f"{current_channel}-1"] = max_binding_response
            # ax2.scatter(
            #     tdata[peak_data_diff],
            #     (ydata2 - ydata1)[peak_data_diff],
            #     facecolors="none",
            #     marker="o",
            #     color="purple",
            # )
            ax2.scatter(
            tdata[irinse-DELTA_BIND],
            (ydata2 - ydata1)[irinse-DELTA_BIND],
            facecolors="none",
            marker="o",
            color="purple",
        )
        # print(f"Peaks of response diff for {cycle.cycle_num} are at times: {tdata[peak_data_diff]}")
        # Get linear regression of tail of sensogram before classifying binding type
        LR_result = scipy.stats.linregress(tdata[-200:], ydata2[-200:] - ydata1[-200:])
        binding_type = classify_sensogram(
            cycle,
            fc=current_channel,
            LR=LR_result,
            max_response=max_binding_response
        )
        if binding_type == "Binding":
            if cycle.is_noisy["FC1"] == True or cycle.is_noisy[current_channel] == True:
                binding_type += ":Noisy"
            if not (-0.05 < LR_result.slope < 0.05):
                binding_type += ":Drift"
        cycle.binding_values[f"{current_channel}-1"] = binding_type


        # quality_text = f"{quality} \n Baseline? {rtb[0]} \n Flat tail? {flat_tail[0]}"
        # Plot lin reg of tail for FC(2-1)
        LR_result = scipy.stats.linregress(tdata[-200:], ydata2[-200:] - ydata1[-200:])
        ax2.plot(
            tdata[-200:],
            LR_result.intercept + LR_result.slope * tdata[-200:],
            color="magenta",
            linestyle="--",
        )

        fitting_candidates = [
                            "Binding",
                            "Binding:Noisy", 
                            "Binding:Drift",
                            "Binding:Noisy:Drift",
                            "Check",
                            "Non-specific"
                              ]

        if binding_type in fitting_candidates and cycle.conc_wrt_time[current_channel] is not None:
            cycle.get_dissociation_phases()
            window=min(window,len(cycle.conc_wrt_time[current_channel]))
            Cdata = cycle.conc_wrt_time[current_channel][:window]
            Rdata = (ydata2 - ydata1)[:window]
            # ax_conc = ax3.secondary_yaxis("right")
            # ax_conc.set_label("Conc.")
            # ax3.plot(
            #     tdata,
            #     cycle.conc_wrt_time[current_channel][:window],
            #     label="Blank ID: " + str(correct_for_blank.cycle_num) + " FC(2-1)",
            #     color="purple",
            #     linestyle=":",
            # )
            # ax3.fill_between(tdata,cycle.conc_wrt_time[current_channel][:window],color="purple",alpha=0.2)
            conc_peaks, _ = scipy.signal.find_peaks(cycle.conc_wrt_time[current_channel][:window], prominence=1.0e-6)
            print(f"Conc peaks {conc_peaks} occur at times: {tdata[conc_peaks]}")

            print(f"Fitting for Cycle {cycle.cycle_num}...")
            bootstrap = False
            model_all = perform_fitting(cycle,tdata[:window],Cdata,Rdata,bootstrap=bootstrap)
            ka,kd,rmax = model_all.final_params
            # assume we want to keep fitted params from all data, not dissoc_only
            cycle.fitted_params[f"{current_channel}-1"] = [ka, kd, rmax]

            markersize=4
            ax2.plot(
                model_all.t_data,
                model_all.R_fit,
                '-', markersize=markersize, markerfacecolor="none",
                color="turquoise",
                label=f'All data (ka = {ka:.2f}, kd = {kd:.2f}, Rmax = {rmax:.2f})'
            )
            if bootstrap:
                # print("LOWER INTERVAL",model_all.lower)
                # print("UPPER INTERVAL",model_all.upper)
                ax2.fill_between(model_all.t_data,
                                 model_all.lower,
                                 model_all.upper, 
                                 color="lightblue",alpha=0.7,label="95% CI")
            # ax2.plot(
            #     model_all.t_data[0::5],
            #     model_all.R_0[0::5],
            #     'x',markersize=markersize,
            #     color="turquoise",
            #     label=f'Initial guess',
            # )
            ax2.plot(
                model_all.t_data[0::5],
                model_all.residuals[0::5],
                # 'x',markersize=markersize,
                color="darkred",
                label=f'Residuals',
            )

            # assume uncertainty sigma is normally distributed from baseline noise
            sigma = np.sqrt(
                            cycle.noise_values[current_channel]**2
                            +cycle.noise_values["FC1"]**2
                            )
            cycle.fitting_metrics[f"{current_channel}-1"] = fit_metrics(model_all.R_data,model_all.R_fit,3,sigma=sigma)
            print(f" cycle.fitting_metrics: {cycle.fitting_metrics[f"{current_channel}-1"]}")

            # print(f"    Pulse data suggests dissoc intervals are at: f{cycle.dissoc_phases}")
            # model_dissoc = perform_fitting(cycle,tdata[:window],Cdata,Rdata,dissoc_only=True)
            # ka,kd,rmax = model_dissoc.final_params
            # ax2.plot(
            #     model_dissoc.t_data,
            #     model_dissoc.R_fit,
            #     '-',markersize=markersize,markerfacecolor="none",
            #     color="purple",
            #     label=f'Dissoc data (ka = {ka:.2f}, kd = {kd:.2f}, Rmax = {rmax:.2f})'
            # )
            # ax2.plot(
            #     model_dissoc.t_data[0::5],
            #     model_dissoc.R_0[0::5],
            #     'x',markersize=markersize,
            #     color="purple",label=f'Dissoc initial guess'
            # ) 


            for ax in [ax1, ax2]:
                for start, end in cycle.dissoc_phases:
                    ax.axvspan(start, end, color="gray",alpha=0.05, linewidth=None, linestyle="None")

        # these are matplotlib.patch.Patch properties
        color = {
            "Binding": "palegreen",
            "Binding:Noisy": "darkseagreen",
            "Binding:Drift": "darkseagreen",
            "Binding:Noisy:Drift": "darkseagreen",
            "No Binding": "lightcoral",
            "Check": "wheat",
            "Non-specific": "khaki",
            "Unknown": "lightgrey",
            "Ignore": "darkgrey"
        }
        props = dict(boxstyle="round", facecolor=color[binding_type], alpha=0.5)
        if "Sample" in cycle.type: # No need to label binding type for blank or DMSO cycles.
            ax1.text(
                0.05,
                0.95,
                binding_type,
                transform=ax1.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )
        # ax2.text(
        #     0.05,
        #     0.95,
        #     "Some text here",
        #     transform=ax1.transAxes,
        #     fontsize=8,
        #     verticalalignment="top",
        #     bbox=props,
        # )

        ax1.legend()
        ax2.legend()

    plotfile = f"cycle{cycle.cycle_num}_{name}.png"
    fig.suptitle(f"cycle{cycle.cycle_num}_{name}_{cycle.sampleid}")
    fig.savefig(pathlib.Path(directory / cycle.subdir / plotfile), bbox_inches="tight")
    plt.close(fig)
    return


def classify_sensogram_old(
    cycle, fc="FC1", tail_slope_tol=0.05, max_response=None, response_thresh=5.0
):
    # is the binding response difference between channels < threshold?  (non-binding)
    print(f"Sensogram for Cycle: {cycle.cycle_num}, {cycle.sampleid}, channel {fc}", end="    ")
    bare_channel_ok, reason = check_bare_channel_ok(
        cycle, fc=fc, tail_slope_tol=tail_slope_tol, response_thresh=response_thresh
    )
    if not bare_channel_ok:
        print(f"Bare channel is not valid: {reason}")
        if reason == "Tail is above threshold":
            binding_type = "Non-specific"
        else:
            binding_type = "Unknown"
    else:
        binding_type = check_relative_response(max_response, response_thresh=5.0)
    return binding_type

def classify_sensogram(
    cycle, fc="FC1", LR=None, max_response=None, response_thresh=5.0
):
    # is the binding response difference between channels < threshold?  (non-binding)
    print(f"Sensogram for Cycle: {cycle.cycle_num}, {cycle.sampleid}, channel {fc}", end="    ")
    blank_channel_ok, reason = check_blank_channel_ok(cycle, dissociation_thresh=response_thresh)
    if not blank_channel_ok:
        print(f"Blank channel is not valid: {reason}")
        if reason == "Negative responses":
            binding_type = "Ignore"
        elif reason == "Non-specific interactions":
            binding_type = "Non-specific"
        return binding_type
    active_channel_ok, reason = check_active_channel_ok(cycle, fc=fc)
    if not active_channel_ok:
        print(f"Active channel is not valid: {reason}")
        binding_type = "Unknown"
        return binding_type
    diff_ok, reason = check_diff_ok(cycle, LR=LR, tail_thresh=response_thresh, max_response=max_response, response_thresh=response_thresh)
    if not diff_ok:
        print(f"Difference between active and blank channels is not valid: {reason}")
        if reason == "Max response is below threshold":
            binding_type = "No Binding"
        elif reason == "Tail is above threshold":
            binding_type = "Check"
        return binding_type
    binding_type = "Binding"
    return binding_type


def check_bare_channel_ok(cycle, fc="FC1", tail_slope_tol=0.05, response_thresh=5.0):
    # is the bare ("blank") channel binding in tail of sensogram?
    tail_linreg = cycle.tail_linreg(fc=fc)
    if not (-tail_slope_tol < tail_linreg.slope < tail_slope_tol):
        # print(f"Bare channel tail slope ({tail_linreg.slope:.2f}) is not flat -- anomaly?")
        return False, "Tail slope not flat"
    elif tail_linreg.intercept < 0.0:
        # print(f"Bare channel response {tail_linreg.intercept:.2f} < 0.0 -- anomaly?")
        return False, "Tail of response is negative"
    elif tail_linreg.intercept > response_thresh:
        # print(f"Bare channel response {tail_linreg.intercept:.2f} > {response_thresh:.2f} -- non-specific")
        return False, "Tail is above threshold"
    else:
        return True, "Tail is flat and within thresholds"
    
def check_blank_channel_ok(cycle, dissociation_thresh=5.0):
    # does the blank channel show negative responses?
    # is the blank channel showing response during dissociation phase above threshold?
    if not cycle.adjusted_dataset:
        cycle.apply_corrections()
    injection_start = (int(np.floor(cycle.t_baseline / dt)))
    blank_dataset = cycle.adjusted_dataset["FC1"]
    if min(blank_dataset[injection_start:]) <= -5.0:
        return False, "Negative responses"
    irinse= (int(np.floor(cycle.t_rinse / dt)))
    print(f"Blank channel dissociation response after rinse time: {blank_dataset[irinse + DELTA_DISS]:.3f}")
    if blank_dataset[irinse + DELTA_DISS] > dissociation_thresh:  # look for response in dissociation phase 0.5 s after rinse time
        return False, f"Non-specific interactions"
    return True, "Blank channel is ok!"

def check_active_channel_ok(cycle, fc="FC2"):
    # does the active channel show negative responses?
    if not cycle.adjusted_dataset:
        cycle.apply_corrections()
    injection_start = (int(np.floor(cycle.t_baseline / dt)))
    active_dataset = cycle.adjusted_dataset[fc]
    if min(active_dataset[injection_start:]) <= -5.0:
        return False, "Negative responses"
    return True, "Active channel is ok!"

def check_diff_ok(cycle, LR=None, tail_thresh=5.0, max_response=None, response_thresh=5.0):
    # does the tail of the difference between active and blank channels goes back to baseline?
    # is the difference between active and blank channels showing binding response above threshold?
    if max_response < response_thresh:
        return False, "Max response is below threshold"
    if LR.intercept > tail_thresh:
        return False, "Tail is above threshold"
    return True, "Difference between active and blank channels is ok, Binding!"

def check_relative_response(max_response, response_thresh=5.0):
    if max_response < 0.0:
        print(f"Non-specific -- max response {max_response:.2f} < 0.0")
        binding_type = "Non-specific"
    elif 0.0 < max_response < response_thresh:
        print(
            f"No binding -- max response {max_response:.2f} below {response_thresh:.2f}"
        )
        binding_type = "No Binding"
    else:
        print(f"max response = {max_response:.2f} Binding!")
        binding_type = "Binding"
    return binding_type
