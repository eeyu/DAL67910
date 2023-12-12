import os.path

import numpy as np
import matplotlib.pyplot as plt

class PlotHandler:
    def __init__(self, c, r):
        self.fig, self.axs = plt.subplots(ncols=c, nrows=r)
        self.fig.set_figwidth(15)
        self.fig.set_figheight(10)


    def plot(self, v, h, data, data_name, title):
        self.axs[v, h].plot(data, label=data_name)
        self.axs[v, h].title.set_text(title)

        lines = self.axs[v, h].get_lines()
        handles, labels = self.axs[v, h].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc='outside right upper')


    def save(self, name, folder):
        path = "./" + folder + "/"
        if not os.path.exists(path):
            os.mkdir(path)
        path += name
        self.fig.savefig(path)


def get_evaluations_multiple(param_overlay_options,
                             param_horiz_name, param_horiz_options,
                             param_vert_name, param_vert_options):
    plt.ion()
    c = len(param_horiz_options)
    r = len(param_vert_options)
    plot1 = PlotHandler(c, r)
    plot2 = PlotHandler(c, r)

    # fig.suptitle('Vertically stacked subplots')

    for i_po, param_overlay in enumerate(param_overlay_options):
        for i_pv, param_vert in enumerate(param_vert_options):
            for i_ph, param_horiz in enumerate(param_horiz_options):
                title = str(param_horiz_name) + ": " + str(param_horiz) + ", " + str(param_vert_name) + ": " + str(param_vert)
                plot1.plot(i_pv, i_ph, np.random.rand(100), data_name=str(param_overlay), title=title)
                plot2.plot(i_pv, i_ph, np.random.rand(100), data_name=str(param_overlay), title=title)


                plt.draw()
                plt.pause(0.01)

    plt.show()
    plot1.save("f1.png", "test_plot")

if __name__=="__main__":
    get_evaluations_multiple(param_overlay_options=[1,2,3],
                             param_horiz_name="h", param_horiz_options=[4,5,6],
                             param_vert_name="v", param_vert_options=[7,8,9])