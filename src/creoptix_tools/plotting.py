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
    fig = plt.figure()
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

    rtb, flat_tail = classify_sensogram(cycle)
    quality = "Good" if rtb[0] and flat_tail[0] else "Bad"

    quality_text = f"{quality} \n Baseline? {rtb[0]} \n Flat tail? {flat_tail[0]}"
    # Plot lin reg of tail for FC(2-1)
    LR_result = scipy.stats.linregress(tdata[-200:], ydata2[-200:] - ydata1[-200:])
    ax2.plot(
        tdata[-200:],
        LR_result.intercept + LR_result.slope * tdata[-200:],
        color="magenta",
        linestyle="--",
    )

    # these are matplotlib.patch.Patch properties
    color = {"Good": "palegreen", "Bad": "lightcoral"}
    props = dict(boxstyle="round", facecolor=color[quality], alpha=0.5)
    # ax1.text(
    #     0.05,
    #     0.95,
    #     quality_text,
    #     transform=ax1.transAxes,
    #     fontsize=14,
    #     verticalalignment="top",
    #     bbox=props,
    # )

    ax1.legend()
    ax2.legend()
    plotfile = f"cycle{cycle.cycle_num}_{name}.png"
    fig.suptitle(f"cycle{cycle.cycle_num}_{name}")
    fig.savefig(pathlib.Path(directory / plotfile), bbox_inches="tight")
    plt.close()
    return


def classify_sensogram(cycle):
    rtb = cycle.returned_to_baseline()
    flat_tail = cycle.has_flat_tail()
    return rtb, flat_tail
