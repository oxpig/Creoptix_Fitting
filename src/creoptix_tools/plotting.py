import scipy
import matplotlib.pyplot as plt
import pathlib


def plot_sensogram(
    cycle,
    correct_for_blank=None,
    diff=True,
    labels=None,
    name=None,
    directory=None,
):
    if cycle.adjusted_dataset:
        dataset = cycle.adjusted_dataset
    else:
        dataset = cycle.dataset
    fig = plt.figure(layout="constrained")
    if diff:
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    else:
        ax1 = fig.add_subplot(1, 1, 1)
    # assuming two relevant flow channels are 1 and 2
    tdata = dataset["Time"]
    ydata1 = dataset["FC1"]
    ydata2 = dataset["FC2"]

    window = min(len(ydata1), len(ydata2))  # get shorter length, assume start at 0
    if correct_for_blank:
        if cycle.adjusted_dataset:
            blank_dataset = correct_for_blank.adjusted_dataset
        else:
            blank_dataset = correct_for_blank.dataset
        window = min(window, len(blank_dataset["FC1"]))
        ydata1 = ydata1[:window] - blank_dataset["FC1"][:window]
        ydata2 = ydata2[:window] - blank_dataset["FC2"][:window]
        tdata = tdata[:window]
        plt.close(fig)
        fig = plt.figure(layout="constrained")
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
        ax3.plot(
            tdata,
            blank_dataset["FC2"][:window] - blank_dataset["FC1"][:window],
            label="Blank ID: " + str(correct_for_blank.cycle_num) + " FC(2-1)",
            color="grey",
            linestyle=":",
        )
        ax3.legend()

    ax1.plot(tdata, ydata1, label="Cycle ID: " + str(cycle.cycle_num) + " FC1")
    ax1.plot(tdata, ydata2, label="Cycle ID: " + str(cycle.cycle_num) + " FC2")
    peak_data1, _ = scipy.signal.find_peaks(ydata1, prominence=1)
    peak_data2, _ = scipy.signal.find_peaks(ydata2, prominence=1)
    ax1.scatter(
        tdata[peak_data1],
        ydata1[peak_data1],
        facecolors="none",
        marker="o",
        color="red",
    )
    ax1.scatter(
        tdata[peak_data2],
        ydata2[peak_data2],
        facecolors="none",
        marker="o",
        color="green",
    )
    if diff:
        ax2.plot(
            tdata,
            ydata2 - ydata1,
            label="Cycle ID: " + str(cycle.cycle_num) + " FC(2-1)",
            color="black",
        )
        peak_data_diff, _ = scipy.signal.find_peaks(ydata2 - ydata1, prominence=1)
        max_binding_response = max((ydata2 - ydata1)[peak_data_diff], default=0.0)
        ax2.scatter(
            tdata[peak_data_diff],
            (ydata2 - ydata1)[peak_data_diff],
            facecolors="none",
            marker="o",
            color="purple",
        )

    binding_type = classify_sensogram(cycle, max_response=max_binding_response)
    # quality = "Good" if rtb[0] and flat_tail[0] else "Bad"

    # quality_text = f"{quality} \n Baseline? {rtb[0]} \n Flat tail? {flat_tail[0]}"
    # Plot lin reg of tail for FC(2-1)
    LR_result = scipy.stats.linregress(tdata[-200:], ydata2[-200:] - ydata1[-200:])
    ax2.plot(
        tdata[-200:],
        LR_result.intercept + LR_result.slope * tdata[-200:],
        color="magenta",
        linestyle="--",
    )

    # these are matplotlib.patch.Patch properties
    color = {
        "Binding": "palegreen",
        "No Binding": "lightcoral",
        "Non-specific": "khaki",
        "Unknown": "lightgrey",
    }
    props = dict(boxstyle="round", facecolor=color[binding_type], alpha=0.5)
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
    fig.savefig(pathlib.Path(directory / plotfile), bbox_inches="tight")
    plt.close(fig)
    return binding_type


def classify_sensogram(
    cycle, tail_slope_tol=0.05, max_response=None, response_thresh=5.0
):
    # is the binding response difference between channels < threshold?  (non-binding)
    print(f"Sensogram for Cycle: {cycle.cycle_num}, {cycle.sampleid}", end="    ")
    bare_channel_ok, reason = check_bare_channel_ok(
        cycle, fc="FC1", tail_slope_tol=tail_slope_tol, response_thresh=response_thresh
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
