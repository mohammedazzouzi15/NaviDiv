import os

import matplotlib.pyplot as plt
import numpy as np  # Import numpy for moving average calculation


def moving_average(data, window_size):
    """Calculate the moving average of a 1D array."""
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def customize_axes(
    axes,
    keep_indices,
    xlabel=None,
    ylabel=None,  # Now can be a list of ylabels or a single string
    font_size=14,
    title=None,
    fig_title=None,
    legends=None,
    output_path="figures/customized_steps_plot_repeats.png",
    hide_axes=True,
    show_legend=True,
    figsize=None,
    add_letters=False,
):
    """Customize only selected axes: set font, labels, and hide others.
    Rearranges the layout to avoid blank spaces and shares axis labels/ticks.
    Shows a single legend at the top of the figure.
    """
    kept_axes = [axes[idx] for idx in keep_indices if idx < len(axes)]
    n = len(kept_axes)
    if n == 0:
        return None

    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (8 * ncols, 4 * nrows)
    fig, new_axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=True,
    )
    new_axes = new_axes.flatten()

    # Collect legend handles/labels from all axes
    legend_handles = {}
    y_label_counter = -1  # Counter for y-labels if a list is provided
    for i, ax in enumerate(kept_axes):
        # Handle twin axes (two y-scales)
        twin_ax = None
        if hasattr(ax, "get_shared_y_axes") and hasattr(ax, "figure"):
            # Try to find a twin y-axis (right y-axis)
            for other_ax in ax.figure.axes:
                if (
                    other_ax is not ax
                    and hasattr(other_ax, "get_shared_x_axes")
                    and other_ax.get_shared_x_axes().joined(ax, other_ax)
                ):
                    if (
                        other_ax.get_position().bounds
                        == ax.get_position().bounds
                    ):
                        twin_ax = other_ax
                        break
        # Plot lines from main axis
        for j, line in enumerate(ax.get_lines()):
            label = legends[j] if legends else line.get_label()
            # Calculate moving average for smoothness
            smoothed_x = moving_average(line.get_xdata(), window_size=3)
            smoothed_y = moving_average(line.get_ydata(), window_size=3)
            new_axes[i].plot(
                smoothed_x,
                smoothed_y,
                label=label,
                color=line.get_color(),
                linewidth=3,  # Increased line thickness
            )
            # Set axis color to match line color
            # new_axes[i].spines["left"].set_color(line.get_color())
            # new_axes[i].yaxis.label.set_color(line.get_color())
            # new_axes[i].tick_params(axis="y", colors=line.get_color())
            if label not in legend_handles:
                legend_handles[label] = line
        # Plot lines from twin axis if present
        if twin_ax is not None:
            ax2 = new_axes[i].twinx()
            for j, line in enumerate(twin_ax.get_lines()):
                label = (
                    legends[j + len(ax.get_lines())]
                    if legends and (j + len(ax.get_lines())) < len(legends)
                    else line.get_label()
                )
                ax2.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    label=label,
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=2.5,  # Increased line thickness
                )
                # Set right axis color to match line color
                # ax2.spines["right"].set_color(line.get_color())
                # ax2.yaxis.label.set_color(line.get_color())
                # ax2.tick_params(axis="y", colors=line.get_color())
                if label not in legend_handles:
                    legend_handles[label] = line
            ax2.yaxis.set_tick_params(labelsize=font_size)
            # Distinguish between left and right y-labels
            if isinstance(ylabel, list):
                # Use i for left, i+1 for right if available
                if i < len(ylabel) and ylabel[i] is not None:
                    print(y_label_counter)
                    y_label_counter += 1
                    ax2.set_ylabel(
                        ylabel[y_label_counter]
                        if y_label_counter < len(ylabel)
                        and ylabel[y_label_counter] is not None
                        else ax2.get_ylabel(),
                        fontsize=font_size,
                    )
                else:
                    ax2.set_ylabel(ax2.get_ylabel(), fontsize=font_size)
            elif ylabel:
                ax2.set_ylabel(ylabel, fontsize=font_size)
            else:
                ax2.set_ylabel(ax2.get_ylabel(), fontsize=font_size)
        new_axes[i].grid(False)
        # Remove per-axis legend
        # Add subplot label (a, b, c, d, etc.) in top left corner
        subplot_label = chr(
            ord("a") + i
        )  # Convert index to letter (a, b, c, d, ...)
        if add_letters:
            new_axes[i].text(
                0.02,
                0.98,
                f"{subplot_label})",
                transform=new_axes[i].transAxes,
                fontsize=font_size + 2,
                fontweight="bold",
                verticalalignment="top",
                horizontalalignment="left",
            )
        # Set subplot title
        if title and i < len(title):
            new_axes[i].set_title(title[i], fontsize=font_size, y=0.9)
        else:
            new_axes[i].set_title(ax.get_title(), fontsize=font_size, y=1.0)
        new_axes[i].set_xlim(ax.get_xlim())
        new_axes[i].set_ylim(ax.get_ylim())
        new_axes[i].xaxis.set_tick_params(labelsize=font_size)
        new_axes[i].yaxis.set_tick_params(labelsize=font_size)
        # set the axis labels
        if xlabel:
            new_axes[i].set_xlabel(xlabel, fontsize=font_size)
        else:
            new_axes[i].set_xlabel(ax.get_xlabel(), fontsize=font_size)
        # Support ylabel as a list or a single string
        if isinstance(ylabel, list):
            if i < len(ylabel) and ylabel[i] is not None:
                y_label_counter += 1
                new_axes[i].set_ylabel(
                    ylabel[y_label_counter], fontsize=font_size
                )
            else:
                new_axes[i].set_ylabel(ax.get_ylabel(), fontsize=font_size)
        elif ylabel:
            new_axes[i].set_ylabel(ylabel, fontsize=font_size)
        else:
            new_axes[i].set_ylabel(ax.get_ylabel(), fontsize=font_size)
    if hide_axes:
        # Hide any unused axes in the new grid
        for j in range(n, len(new_axes)):
            new_axes[j].set_visible(False)
        # ax.tick_params(axis="both", which="minor", length=4, width=1, color='black')
        # Only show x labels/ticks on bottom row, y labels/ticks on first column
        for idx, ax in enumerate(new_axes[:n]):
            row = idx // ncols
            col = idx % ncols

            # X axis: show only on bottom row
            if row != nrows - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False, which="major")
            else:
                ax.tick_params(axis="x", labelbottom=True, which="major")
            # Y axis: show only on first column
            if col != 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False, which="major")
            else:
                ax.tick_params(axis="y", labelleft=True, which="major")
            ax.tick_params(
                axis="both", which="major", length=7, width=1.5, color="black"
            )

            ax.yaxis.set_ticks([0, 20, 40, 60, 80, 100])

    # Add a single legend at the top of the figure
    if show_legend and legend_handles:
        for handle in legend_handles.values():
            handle.set_linewidth(4)  # Set desired line width here

        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=len(legend_handles) // 2,
            fontsize=font_size,
            frameon=False,
        )

    # Set overall figure title if provided
    if fig_title:
        fig.suptitle(fig_title, fontsize=font_size + 2)

    fig.tight_layout(
        rect=[0, 0, 1, 0.85] if fig_title or legend_handles else None
    )

    # Save the customized figure
    return fig


def set_axes_limits(axes, limits):
    """Set axis limits for a list of axes. Limits should be a list of (ymin, ymax) tuples or None for each axis."""
    for i, ax in enumerate(axes):
        if limits and i < len(limits) and limits[i] is not None:
            ymin, ymax = limits[i]
            ax.set_ylim(ymin, ymax)


if __name__ == "__main__":
    import pickle

    output_path = "figures/modified/steps_plot_multi_experiment_second.png"
    os.makedirs("figures/modified", exist_ok=True)
    # Load a figure and axes from a pickle file
    with open(
        "/media/mohammed/Work/Navi_diversity/example_other_cases/figures/raw/steps_plot_multi_experiment.pkl",
        "rb",
    ) as f:
        fig = pickle.load(f)
    axes = fig.get_axes()

    fig_custom = customize_axes(
        axes,
        keep_indices=[0, 1,  12, 13, 4, 5,10, 11,],  # , 10, 11], # 0, 1
        xlabel="Reinforcement Learning Steps",
        ylabel=[
            # "Unique Circles (Morgan Fingerprint)",
            "Score",
            "Prior Negative \n Log-Likelihood",
            "Fragment Occurrences in \n > 10% Molecules ",
            "Distinct Fragment \n Ratio (%)",
            "Mean Internal Similarity ",
            "Distinct Circles \n Ratio (%)",
            # "Distinct Scaffolds Ratio (%)",
            "10-gram Occurrences in \n > 10% Molecules",
            "Distinct 10-gram \n Ratio (%)",
            "Occurrences in \n > 10% Molecules ",
            "Distinct 10-gram \n Ratio (%)",
            # "Ring Diversity",
            # "Wireframe Scaffold Diversity",
        ],
        font_size=18,
        title=["" for _ in range(10)],
        # legends=["Strong Guidance", "Weak Guidance", "With Penalty"],
        legends=[
            "Combined High Constraints",
            "Combined Low Constraints",
            "Baseline (No Constraints)",
            "Fragment-Based",
            "10-gram-Based",
            # "Scaffold Only",
            "Similarity-Based",
        ],
        output_path=output_path,
        hide_axes=False,
        show_legend=True,
        figsize=(12, 15),
        add_letters=False,
    )
    ylim_list = [
        (0.5, 1.0),
        (20, 40),
        (0, 5),
        (70, 110),
        (0.10, 0.30),
        (80, 110),
        (0, 50),
        (50, 110),
        (0, 110),
    ]
    for i, ax in enumerate(fig_custom.get_axes()):
        if i < len(ylim_list) and ylim_list[i] is not None:
            ax.set_ylim(ylim_list[i])
        else:
            ax.set_ylim(0, 100)
        ax.set_xlim([0, 1020])
    fig_custom.savefig(output_path, bbox_inches="tight", dpi=300)
