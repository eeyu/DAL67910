import numpy as np
import matplotlib.pyplot as plt
def get_evaluations_multiple(param_overlay_options,
                             param_horiz_name, param_horiz_options,
                             param_vert_name, param_vert_options):
    plt.ion()
    fig1, axs1 = plt.subplots(ncols=len(param_horiz_options), nrows=len(param_vert_options))
    fig2, axs2 = plt.subplots(ncols=len(param_horiz_options), nrows=len(param_vert_options))
    # fig.suptitle('Vertically stacked subplots')

    for i_po, param_overlay in enumerate(param_overlay_options):
        for i_pv, param_vert in enumerate(param_vert_options):
            for i_ph, param_horiz in enumerate(param_horiz_options):
                axs2[i_pv, i_ph].plot(np.random.rand(100), label=str(param_overlay))
                axs2[i_pv, i_ph].title.set_text(str(param_horiz_name) + ": " + str(param_horiz) + ", \n" +
                                       str(param_vert_name) + ": " + str(param_vert))
                axs1[i_pv, i_ph].plot(np.random.rand(100), label=str(param_overlay))
                axs1[i_pv, i_ph].title.set_text(str(param_horiz_name) + ": " + str(param_horiz) + ", \n" +
                                       str(param_vert_name) + ": " + str(param_vert))

                plt.draw()
                plt.pause(0.01)

    plt.show()
    fig1.savefig("test.png")
    fig2.savefig("f2.png")

if __name__=="__main__":
    get_evaluations_multiple(param_overlay_options=[1,2,3],
                             param_horiz_name="h", param_horiz_options=[4,5,6],
                             param_vert_name="v", param_vert_options=[7,8,9])