import matplotlib
import matplotlib.pyplot as plt
import  pytest

def show_plot_non_blocking():
    matplotlib.use("Agg", force=True)
    plt.show(block=False)
    plt.close("all")
    print("CLOSECLOSECLOSECLOSECLOSECLOSECLOSECLOSECLOSECLOSE????")

@pytest.fixture(autouse=True)
def _mpl_non_interactive_backend():
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    matplotlib.use("Agg", force=True)
    yield
    plt.close("all")