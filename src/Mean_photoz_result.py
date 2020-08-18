from tensorflow import keras
import  tensorflow as tf
import mdn
import numpy as np
import time as time
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import scipy.stats as ss
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy as sp
from matplotlib import colors
from scipy.stats import norm




def plot_meanredshift(data_pre, data,  plotname, path):
    try:

        print('data["photoz"]', data["PhotoZ"])
        print('data["zs"]', data["zs"])
        delta = (data["PhotoZ"] - data["zs"]) / (1 + data["zs"])#(data["PhotoZ"] - data["zs"]) / (1 + data["zs"])

        print("residuals inside the plot def", np.array(delta))

        print("zs inside the plot def", data["zs"])
        print("photoz inside the plot def", data["PhotoZ"])
        data_eval = np.mean(np.array(delta))
        print("data_eval", data_eval)
        data_eval_2 = np.sqrt(np.mean(np.power(delta, 2)))

        print("mean delta on mean redshift for data_zs_lim with model is:", data_eval)
        print("rms delta on mean redshift for data_zs_lim with model is:", data_eval_2)

        out_file =open(str(path)+ "/MeanPhotoz"+str(plotname)+".txt", "w")
        out_file.write("mean delta:%s\n" % data_eval)
        out_file.write("rms delta:%s\n" % data_eval_2)

        out_file.close()


        ####eval 3 different range



        def range_eval(data_range, name_range):
            delta = (data_range["PhotoZ"] - data_range["zs"]) / (1 + data_range["zs"])

            data_eval = np.mean(delta)
            data_eval_2 = np.sqrt(np.mean(np.power(delta, 2)))


            out_file = open(str(path)+ "/MeanPhotoz" + str(plotname) + str(name_range)+".txt", "w")
            out_file.write("mean delta:%s\n" % data_eval)
            out_file.write("rms delta:%s\n" % data_eval_2)

            out_file.close()

            #####histogram on deltaz
            plt.figure()
            # Fit a normal distribution to the data:
            mu_hist, std_hist = norm.fit(delta)

            # Plot the histogram.
            plt.hist(delta, bins=200, density=True)#, alpha=0.6)

            # Plot the PDF.
            xmin, xmax = plt.xlim()
            x_hist = np.linspace(xmin, xmax, 200)
            p_hist = norm.pdf(x_hist, mu_hist, std_hist)
            plt.plot(x_hist, p_hist, 'k', linewidth=2)
            plt.title("Fit results: mu = {:0.02f},  std = {:0.02f}".format(mu_hist, std_hist))
            plt.xlabel(r"$\Delta Z$")
            plt.savefig(str(path)+ "/Meanredshift_deltaHist_" + str(plotname) + str(name_range)+".png")


        data_range1 = data[data["zs"] > 0.3]
        data_range1 = data_range1[data_range1["zs"] < 0.6]
        data_range2 = data[data["zs"] < 1.001]
        data_range2 = data_range2[data_range2["zs"] > 0.999]
        data_range3 = data[data["zs"] > 2]

        range_eval(data_range1, "Range1")
        range_eval(data_range2, "Range2")
        range_eval(data_range3, "Range3")
        data_range4 = data[data["zs"] < 0.4]
        data_range4 = data_range4[data_range4["class"] == 'GALAXY']
        data_range5 = data[data["zs"] < 0.5]
        data_range5 = data_range5[data_range5["class"] == 'GALAXY']

        range_eval(data_range1, "Range1")
        range_eval(data_range2, "Range2")
        range_eval(data_range3, "Range3")
        range_eval(data_range4, "Range4_GALAXY")
        range_eval(data_range5, "Range5_GALAXY")









        ####eval classes
        def range_eval_cl(data_range, name_range, cl_):
            delta = (data_range["PhotoZ"] - data_range["zs"]) / (1 + data_range["zs"])

            data_eval = np.mean(delta)
            data_eval_2 = np.sqrt(np.mean(np.power(delta, 2)))

            out_file = open(str(path)+ "/MeanPhotoz" + str(plotname) + str(cl_) + str(name_range) + ".txt", "w")
            out_file.write("mean delta:%s\n" % data_eval)
            out_file.write("rms delta:%s\n" % data_eval_2)


            out_file.close()



        print("DATAs columns", data.columns)
        x = data["zs"]
        photoz = data["PhotoZ"]


        zph_lim = 0.01*(1 + data["PhotoZ"])


        deltaZ_A_ = data["deltaZ_A"]

        res= (data["PhotoZ"] - data["zs"]) / (1 + data["zs"])

        plt.figure(figsize=(10, 12))
        fig, (ax1,  ax4) = plt.subplots(nrows=2, sharex=True, gridspec_kw={
            # 'width_ratios': [1, 1],
            'height_ratios': [5, 1]})
        plt.subplots_adjust(hspace=.1)

        # ax2.grid()
        ax1.grid()

        my_cmap = plt.cm.viridis  # plt.cm.jet
        my_cmap.set_under('w', 1)

        ###1st plot

        ax1.plot((0, 7), (0, 7), linestyle='-.', color='grey', linewidth=0.5)
        h1 = ax1.hist2d(x, photoz, bins=700, norm=colors.LogNorm(),
                        cmap=my_cmap)#, cmin=1, vmin=0)
        ax1.set_ylabel(r"$z_{photo}$")
        ax1.set_ylim(-0.2, 7)
        ax1.tick_params(axis="y", labelsize=10)

        ###2nd plot
        # h2 = ax2.hist2d(x, probability, bins=700, norm=colors.LogNorm(),
        #                 cmap=my_cmap)#, cmin=1, vmin=0)
        # ax2.set_ylabel("P")

        ###3rd plot

        # h3 = ax3.hist2d(x, sigmas, bins=700, norm=colors.LogNorm(),
        #                 cmap=my_cmap)#, cmin=1, vmin=0)
        # ax3.set_ylabel(r"$\sigma$")

        ###4th plot
        h4 = ax4.hist2d(x, res, bins=700, norm=colors.LogNorm(),
                        cmap=my_cmap)#, cmin=1, vmin=0)
        ax4.set_ylabel(r"$\Delta z$")#, rotation=90)
        ax4.set_xlabel(r"$z_{spec}$")
        ax4.set_xlim(-0.2, 7)

        cbbox = inset_axes(ax1, '2%', '80%', loc=7)
        plt.colorbar(h1[3], cax=cbbox)

        # cbbox = inset_axes(ax2, '2%', '80%', loc=7)
        # plt.colorbar(h2[3], cax=cbbox)

        # cbbox = inset_axes(ax3, '2%', '80%', loc=7)
        # plt.colorbar(h3[3], cax=cbbox)

        cbbox = inset_axes(ax4, '2%', '80%', loc=7)
        plt.colorbar(h4[3], cax=cbbox)

        # ax1.title.set_text("Objects:" + str(len(data)) )#+ "frac:{:0.2f}".format(frac))

        plt.savefig(str(path)+ "/Meanredshift_hist_"+str(plotname)+".png")


        plt.figure()
        plt.scatter(x, photoz)

        plt.savefig(str(path)+ "/Meanredshift_scatter_" + str(plotname) + ".png")


        #
        # #################max redshift plot for each class
        #
        class_list = ["STAR", "GALAXY", "QSO"]
        print(data.columns)
        for cl_ in class_list:
            data_lim_class_plot = data[data["class"] == cl_]

            ####zoom hist
            if cl_ == "GALAXY":
                data_lim_class_plot = data[data["class"] == cl_]
                data_lim_class_plot_zoom = data_lim_class_plot[data_lim_class_plot["zs"] < 0.3]
                data_lim_class_plot_zoom = data_lim_class_plot_zoom[data_lim_class_plot_zoom["PhotoZ"] < 0.3]
                x = data_lim_class_plot_zoom["zs"]
                photoz = data_lim_class_plot_zoom["PhotoZ"]

                deltaZ_A_ = data_lim_class_plot_zoom["deltaZ_A"]
                res=  (data_lim_class_plot_zoom["PhotoZ"] - data_lim_class_plot_zoom["zs"])/(1+data_lim_class_plot_zoom["zs"])
                errorz = np.abs(x - photoz) / (x + 1)
                # probability = data_lim_class_plot_zoom["max_pis"]
                # sigmas = data_lim_class_plot_zoom["sigs"]
                zph_lim = 0.01 * (1 + data_lim_class_plot_zoom["PhotoZ"])
                data_merged_class = data_pre[data_pre["class"] == cl_]
                frac_class = len(data_lim_class_plot) / len(data_merged_class)

                plt.figure(figsize=(10, 12))
                fig, (ax1, ax4) = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                    # 'width_ratios': [1, 1],
                    'height_ratios': [5,  1]})
                plt.subplots_adjust(hspace=.1)

                # ax2.grid()
                ax1.grid()

                my_cmap = plt.cm.viridis  # plt.cm.jet
                my_cmap.set_under('w', 1)

                ###1st plot

                ax1.plot((0, 1), (0, 1), linestyle='--', color='grey', linewidth=0.5)
                h1 = ax1.hist2d(x, photoz, bins=700, #norm=colors.LogNorm(),
                                cmap=my_cmap , cmin=1, vmin=0)
                ax1.set_ylabel(r"$z_{photo}$")
                ax1.set_ylim(-0.001, 0.3)
                ax1.tick_params(axis="y", labelsize=10)

                ###2nd plot
                # h2 = ax2.hist2d(x, probability, bins=700, #norm=colors.LogNorm(),
                #                 cmap=my_cmap , cmin=1, vmin=0)
                # ax2.set_ylabel("P")

                ###3rd plot
                #
                # h3 = ax3.hist2d(x, sigmas, bins=700, #norm=colors.LogNorm(),
                #                 cmap=my_cmap , cmin=1, vmin=0)
                # ax3.set_ylabel(r"$\sigma$")

                ###4th plot
                h4 = ax4.hist2d(x, res, bins=700, #norm=colors.LogNorm(),
                                cmap=my_cmap , cmin=1, vmin=0)
                ax4.set_ylabel(r"$\Delta z$")
                ax4.set_xlabel(r"$z_{spec}$")
                ax4.set_xlim(-0.001, 0.3)

                cbbox = inset_axes(ax1, '2%', '80%', loc=7)
                plt.colorbar(h1[3], cax=cbbox)

                # cbbox = inset_axes(ax2, '2%', '80%', loc=7)
                # plt.colorbar(h2[3], cax=cbbox)
                #
                # cbbox = inset_axes(ax3, '2%', '80%', loc=7)
                # plt.colorbar(h3[3], cax=cbbox)

                cbbox = inset_axes(ax4, '2%', '80%', loc=7)
                plt.colorbar(h4[3], cax=cbbox)
                frac_class_zoom = len(data_lim_class_plot_zoom) / len(data_lim_class_plot)

                # ax1.title.set_text(
                #     str(cl_) + ":" + str(len(data_lim_class_plot)) )#+ "frac:{:0.02f}".format(frac_class_zoom))

                plt.savefig(str(path)+ "/Meanredshift_hist_Zoom_" + str(plotname) + str(cl_) + ".png")

                range_eval_cl(data_lim_class_plot_zoom, "Zoom", cl_)
                print("eval zoom")


    except Exception as e:

        print("no profile", e)



def outliers_meanredshift(data_pre, data,  plotname, path):
    try:
        delta = (data["PhotoZ"] - data["zs"]) / (1 + data["zs"])
        #
        # data_eval = np.mean(delta)
        data_eval_2 = np.sqrt(np.mean(np.power(delta, 2)))
        print("data type", type(data))
        print(len(data))
        print(data.columns)
        print(data.tail())
        # print(data.loc[0,"PhotoZ"])
        # print("index", data.index)

        out_3sig=0
        in_3sig=0
        df_misszs={}
        df_misszs["index"]=[]
        for i_dat in data.index:
            # print("index",i_dat)
            delta_i= (data.loc[i_dat,"PhotoZ"] - data.loc[i_dat,"zs"]) / (1 + data.loc[i_dat,"zs"])
            # print("delta",delta_i)


            if delta_i <= data_eval_2 :
                in_3sig= in_3sig +1

            else:
                out_3sig=out_3sig+1
                df_misszs["index"].append(i_dat)

        frac_out= (out_3sig /len(data) )* 100
        print("frac out", frac_out)

        df_misszs=pd.DataFrame(df_misszs)
        df_misszs_=pd.merge(df_misszs, data, on="index")
        print("df_misszs_", df_misszs_.tail())
        df_misszs_.to_pickle(str(path)+ "/Mean_df_misszs_" + str(plotname) + ".pkl")



        out_file = open(str(path)+ "/MeanPhotoz_outliers_frac" + str(plotname) + ".txt", "w")

        out_file.write("frac percentage:%s\n" % frac_out)

        out_file.close()









    except Exception as e:

        print("no profile", e)
        frac_out=e

    # return frac_out


def ZoomGal_outliers_meanredshift(data_pre, data,  plotname, path):
    print("data type", type(data))
    print(len(data))
    print(data.columns)
    print(data.tail())
    data_=data[data["class"] == 'GALAXY']
    data__=data_[data_["zs"] <= 0.3]


    try:
        delta = (data__["PhotoZ"] - data__["zs"]) / (1 + data__["zs"])
        #
        # data_eval = np.mean(delta)
        data_eval_2 = np.sqrt(np.mean(np.power(delta, 2)))

        # print(data.loc[0,"PhotoZ"])
        # print("index", data.index)

        out_3sig=0
        in_3sig=0
        df_misszs = {}
        df_misszs["index"] = []
        for i_dat in data__.index:
            # print("index",i_dat)
            delta_i= (data__.loc[i_dat,"PhotoZ"] - data__.loc[i_dat,"zs"]) / (1 + data__.loc[i_dat,"zs"])
            # print("delta",delta_i)


            if delta_i <= data_eval_2 :
                in_3sig= in_3sig +1
            else:
                out_3sig=out_3sig+1
                df_misszs["index"].append(i_dat)

        frac_out= (out_3sig /len(data__) )* 100
        print("frac out", frac_out)

        df_misszs = pd.DataFrame(df_misszs)
        df_misszs_ = pd.merge(df_misszs, data, on="index")
        print("df_misszs_", df_misszs_.tail())
        df_misszs_.to_pickle(str(path)+ "/Mean_df_misszs_ZOOM_GALAXY_" + str(plotname) + ".pkl")



        out_file = open(str(path)+ "/MeanPhotoz_outliers_frac_ZOOM_GALAXY_" + str(plotname) + ".txt", "w")

        out_file.write("frac percentage:%s\n" % frac_out)

        out_file.close()









    except Exception as e:

        print("no profile", e)




