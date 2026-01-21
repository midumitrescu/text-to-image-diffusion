import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def show_plot_non_blocking():
    plt.show(block=False)
    plt.close("all")