import matplotlib.pyplot as plt


def generate_boxplot(np_array, plot_title=None, save_path=None):
    fig = plt.figure(figsize=(10, 7))
    plt.boxplot(np_array)
    plt.title(plot_title)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def generate_line_plot(np_array, plot_title=None, save_path=None, x_label=None, y_label=None):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(np_array)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(plot_title)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
