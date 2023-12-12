import matplotlib.pyplot as plt
import os

class PlotHandler:
    def __init__(self, c, r, title):
        self.fig, self.axs = plt.subplots(ncols=c, nrows=r)
        self.fig.suptitle(title, fontsize=16)
        self.fig.set_figwidth(15)
        self.fig.set_figheight(10)



    def plot(self, v, h, data, x_data, data_name, title, x_axis_name, y_axis_name):
        self.axs[v, h].plot(x_data, data, label=data_name)
        self.axs[v, h].title.set_text(title)
        self.axs[v, h].set_xlabel(x_axis_name)
        self.axs[v, h].set_ylabel(y_axis_name)

        lines = self.axs[v, h].get_lines()
        handles, labels = self.axs[v, h].get_legend_handles_labels()
        self.fig.legend(lines, labels, loc='outside right upper')


    def save(self, name, folder):
        path = "./" + folder + "/"
        if not os.path.exists(path):
            os.mkdir(path)
        path += name
        self.fig.savefig(path)

class DictList:
    def __init__(self):
        self.lists = {}
        self.names = []

    def keys(self):
        return self.names

    def start_list(self, name):
        self.lists[name] = []
        self.names.append(name)

    def add_to_list(self, name, datapoint):
        self.lists[name].append(datapoint)

    def get_list(self, name):
        return self.lists[name]

class MultiPlotHandler:
    def __init__(self, r, c, list_names):
        self.plots = {}
        self.list_names = list_names
        for name in list_names:
            self.plots[name] = PlotHandler(c=c, r=r, title=name)

    def fill_with_list(self, v, h, label, lists, title, x_axis_name, y_axis_name):
        for name in self.list_names:
            self.plots[name].plot(v=v, h=h, data=lists.get_list(name), x_data = lists.get_list("dataset_size"),
                                  data_name=label, title=title,
                                  x_axis_name=x_axis_name, y_axis_name=y_axis_name)

    def save(self, folder):
        for name in self.list_names:
            self.plots[name].save(name=name, folder=folder)