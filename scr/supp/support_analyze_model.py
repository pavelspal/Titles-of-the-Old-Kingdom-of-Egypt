import matplotlib.pyplot as plt


class GraphPlotter:
    def __init__(self, plot_function):
        self.curves = []  # Store curves to be plotted later
        self.plot_function = plot_function  # Function to use for plot creation
        self.figure_size = {plot_cdf: (10, 6),
                            plot_roc: (8, 8),
                            reliability_plot: (8, 8)
                            }[plot_function]

    def plot(self, x_list, y_list, label_name):
        """
        Plots a single curve. Stores the curve for later plotting.
        """
        # Store the curve for later plotting
        self.curves.append((x_list, y_list, label_name))

    def show(self):
        """
        Plots all accumulated curves in a single figure.
        """

        # Create a single figure for all plots
        plt.figure(figsize=self.figure_size)

        # Plot each curve
        for id_plot in range(len(self.curves)):
            x_list, y_list, label_name = self.curves[id_plot]
            params = {'id_plot': id_plot
                      }
            self.plot_function(x_list, y_list, label_name, show=False, **params)

        # Show plot
        plt.show()


def plot_cdf(y_true, y_hat, column_name, show=True, **kwargs):
    """
    Plot a Cumulative Density Function (CDF) with jittered points below the x-axis

    Parameters:
    - y_true (pandas series): pandas series containing ground true values.
    - y_hat (pandas series): pandas series containing predicted probabilities.
    - column_name (str): list of names of the column for plot title
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create figure
    if show:
        plt.figure(figsize=(10, 6))
    custom_legend_handles = []
    marker_styles = ['x', '+', '1', '2', '3', '4', 'o', 's', '^', 'P', 'D', 'p', 'X', '*']
    line_styles = ['-', '--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5))]
    id_plot = kwargs.get("id_plot", 0)

    # Covert lists to ppd series
    if isinstance(y_hat, list):
        y_hat = pd.Series(y_hat)
    if isinstance(y_true, list):
        y_true = pd.Series(y_true)

    levels = y_true.unique()  # Get unique vizier levels
    for level in levels:
        mask = (y_true == level)
        subset = y_hat[mask]

        # For vizier == 0, select the top 20% of points
        if level == 0:
            subset = subset[subset >= subset.quantile(0.98)]

        # Plot the CDF
        sns.ecdfplot(
            subset,
            marker=marker_styles[id_plot],
            linestyle=line_styles[id_plot],
            label=f'Model {column_name}, vizier = {level}'
        )

        # Define vertical jitter range based on vizier level, all below y=0
        if level == 0:
            jitter_base = -0.11  # Base for vizier = 0
        else:
            jitter_base = -0.03  # Base for vizier = 1

        # Apply jitter around the base for each level
        jitter = np.random.uniform(-0.03, 0.03, len(subset))  # Narrow vertical jitter range
        scatter = plt.scatter(
            subset,  # Fixed x-coordinate for probabilities
            [jitter_base + j for j in jitter],  # Apply jitter to the y-axis points
            alpha=0.6,
            s=40,  # Larger point size for better visibility
            marker=marker_styles[id_plot]
        )


    plt.title(f'CDF curve')
    plt.xlabel('Predicted probability')
    plt.ylabel('CDF')
    plt.xlim(0, 1)  # Set x-axis limits
    plt.grid(True)
    # Add legend with custom handles
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # Adjust layout to prevent clipping
    plt.tight_layout()

    if show:
        plt.title(f'CDF for {column_name}')
        plt.savefig(rf'img/cdf_plot_{column_name}.png', dpi=600,
                    bbox_inches='tight')  # Adjust dpi and bounding box if needed
        plt.show()


def plot_roc(y_true, y_hat, column_name, show=True, **kwargs):
    """
    Plot a ROC curve

    Parameters:
    - y_true (pandas series): pandas series containing ground true values.
    - y_hat (pandas series): pandas series containing predicted probabilities.
    - column_name (str): list of names of the column for plot title
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    # Create figure
    if show:
        plt.figure(figsize=(8, 8))
    # Set list of markers and line styles
    marker_styles = ['o', 's', '^', 'P', 'D', 'p', 'X', '*']
    line_styles = ['-', '--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5))]
    id_plot = kwargs.get("id_plot", 0)

    # Calculate FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_hat)

    # Calculate AUC
    auc = roc_auc_score(y_true, y_hat)

    # Plot diagonal reference line
    if id_plot == 0:
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")  # Dashed diagonal line

    # Plot the ROC Curve
    plt.plot(fpr, tpr,
             marker=marker_styles[id_plot],
             linestyle=line_styles[id_plot],
             label=f"Model {column_name}, (AUC = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    if show:
        plt.title(f"ROC Curve for {column_name} model")
        plt.savefig(rf'img/roc_plot_{column_name}.png', dpi=600,
                    bbox_inches='tight')  # Adjust dpi and bounding box if needed
        plt.show()


# Function for getting calibration_curve data
def get_calibration_curve(y_true_list, y_pred_list, bin_edges):
    """
    Calculate Y, X data for calibration curve

    Parameters:
    - y_true (list of floats): list containing ground true values.
    - y_hat (list of floats): list containing predicted probabilities.
    - bin_edges (list of float): edges for calibration bins.
    """
    import numpy as np

    # Convert input list to numpy array
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    # Manually bin the data
    bins = np.digitize(y_pred, bins=bin_edges) - 1

    # Compute mean predicted probabilities and observed fraction
    bin_means = []
    observed_fraction = []
    for i in range(len(bin_edges) - 1):
        mask = (bins == i)
        if np.sum(mask) > 0:
            bin_means.append(y_pred[mask].mean())
            observed_fraction.append(y_true[mask].mean())
        else:
            bin_means.append(np.nan)  # Handle empty bins
            observed_fraction.append(np.nan)

    # Remove empty bins
    bin_means = np.array(bin_means)
    observed_fraction = np.array(observed_fraction)
    valid_bins = ~np.isnan(bin_means)
    bin_means = bin_means[valid_bins]
    observed_fraction = observed_fraction[valid_bins]

    return observed_fraction, bin_means

def reliability_plot(y_true, y_hat, column_name, bin_edges=None, show=True, **kwargs):
    """
    Plot a reliability plot

    Parameters:
    - y_true (pandas series): pandas series containing ground true values.
    - y_hat_list (pandas series): pandas series containing predicted probabilities.
    - column_name (str): name of the column for plot title.
    - bin_edges (list of float): edges for calibration bins.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    # Create figure
    if show:
        plt.figure(figsize=(8, 8))
    # Set bin_edges if not defined
    if bin_edges is None:
        bin_edges = [0 - 1e-10, 0.1, 0.6, 0.9, 1 + 1e-10]
    # Set list of markers and line styles
    marker_styles = ['o', 's', '^', 'P', 'D', 'p', 'X', '*']
    line_styles = ['--', '-.', ':', (0, (5, 10)), (0, (3, 5, 1, 5))]
    id_plot = kwargs.get("id_plot", 0)

    # Plot diagonal reference line
    if id_plot == 0:
        plt.plot([0, 1], [0, 1], "k-", linewidth=1, label="Perfectly calibrated")

    # Compute calibration curve
    # prob_true, prob_pred = calibration_curve(y_true, y_hat, n_bins=5, strategy='uniform')
    prob_true, prob_pred = get_calibration_curve(y_true, y_hat, bin_edges)
    # Plot reliability curve
    plt.plot(prob_pred, prob_true,
             marker=marker_styles[id_plot],
             linestyle=line_styles[id_plot],
             label=f'Model: {column_name}')

    # Highlight bins with alternating gray areas
    for i in range(len(bin_edges) - 1):
        if i % 2 == 1:  # Fill every second bin
            plt.axvspan(bin_edges[i], bin_edges[i + 1], color="gray", alpha=0.2)

    # Customize plot
    plt.title('Reliability Plot')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid()
    # Ensure the same scale on both axes
    plt.axis('equal')
    # Set ticks at 0.1 intervals
    ticks = np.arange(0, 1.1, 0.1)  # Generate ticks from 0 to 1 at 0.1 intervals
    plt.xticks(ticks)
    plt.yticks(ticks)
    # Set x and y-axis limits to exactly [0, 1]
    margin = 0.012
    plt.xlim(0 - margin, 1 + margin)
    plt.ylim(0 - margin, 1 + margin)

    if show:
        plt.title(f'Reliability plot for model')
        plt.savefig(rf'img/reliability_plot.png', dpi=600, bbox_inches='tight')  # Adjust dpi and bounding box if needed
        plt.show()
