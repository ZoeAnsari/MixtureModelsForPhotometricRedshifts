from tensorflow import keras
import tensorflow as tf
import mdn
import numpy as np
import time as time
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import scipy.stats as ss
import seaborn as sns
from src.Peak_photoz_result import plot_peakredshift, outliers_peakredshift, ZoomGal_outliers_peakredshift
from src.Mean_photoz_result import plot_meanredshift, outliers_meanredshift, ZoomGal_outliers_meanredshift


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def MDN_result(mus, sigs, pis,MDN_path, path, data_z_tot,weight_con_prior):
    start=time.time()
    try:
        sigs = sigs.drop(columns={"index"})
        pis = pis.drop(columns={"index"})
        y_mem_=pd.read_pickle(str(MDN_path) + "/y_mem.pkl")
        mus = mus.drop(columns={"index"})



        df_max = {}
        df_max["branch"] = []
        df_max["max_pis"] = []
        df_max["mus"] = []
        df_max["sigs"] = []
        df_max["y_mem"] = []
        df_max["index"]=[]


        for i_mus in range(len(mus)):
            maxValueIndex = (pis.iloc[i_mus, :]).idxmax(axis=0)
            df_max["branch"].append(maxValueIndex)
            df_max["max_pis"].append(pis.iloc[i_mus, maxValueIndex])
            df_max["mus"].append(mus.iloc[i_mus, maxValueIndex])
            df_max["sigs"].append(sigs.iloc[i_mus, maxValueIndex])
            df_max["y_mem"].append(float(y_mem_.iloc[i_mus]))
            df_max["index"].append(i_mus)



        df_max = pd.DataFrame.from_dict(df_max)
        df_max.to_pickle(str(path)+"/df_max.pkl")

        df_max = pd.read_pickle(str(path)+"/df_max.pkl")



        data_train = data_z_tot[['g', 'Err_g', 'r', 'Err_r', 'i',
                                 'Err_i', 'z', 'Err_z', 'extinction_i', 'w1', 'w1_sig',
                                 'w2', 'w2_sig', 'psfMag_g', 'psfMag_r',
                                 'psfMag_i', 'psfMag_z', "index", "zs", "zErr", "class"]]
        # print(data_train.columns)

        data_train["g-r"] = (data_train['g'] - data_train['r'])
        data_train["r-i"] = (data_train['r'] - data_train['i'])
        data_train["i-z"] = (data_train['i'] - data_train['z'])
        data_train["z-w1"] = (data_train['z'] - data_train['w1'])
        data_train["w1-w2"] = (data_train['w1'] - data_train['w2'])


        df_max_=pd.merge(data_train, df_max, on="index")


        df_max_["mus"]=  10 **df_max_["mus"]
        df_max_["y_mem"] = 10 ** df_max_["y_mem"]

        df_max_=df_max_[df_max_["mus"] <7]



        print("--------------check point", np.max(df_max_["mus"]))
        print("--------------check point", np.max(df_max_["y_mem"]))


        df_max_["r-w1"] = df_max_["r"] - df_max_["w1"]


        # ############ the peak redshift

        df_uncertanity = {}
        df_uncertanity["deltaZ_A"] = []
        df_uncertanity["index"] = []
        df_uncertanity["PhotoZ"] = []
        df_uncertanity["zs"] = []
        df_uncertanity["max_pis"] = []
        df_uncertanity["sigs"] = []


        for branch in range(len(mus.iloc[0, :])):
            df_max_branch = df_max_[df_max_["branch"] == branch]
            sig_branch = np.mean(df_max_branch["sigs"])
            p_branch = np.mean(df_max_branch["max_pis"])

            for ind in df_max_branch["index"]:  # range(len(df_max_branch)):
                sig_max = df_max_branch.loc[ind, "sigs"]
                p_max = df_max_branch.loc[ind, "max_pis"]
                deltaZ_A = sig_branch * p_max / sig_max
                df_uncertanity["zs"].append(df_max_branch.loc[ind, "zs"])
                df_uncertanity["deltaZ_A"].append(deltaZ_A)
                df_uncertanity["index"].append(ind)
                df_uncertanity["PhotoZ"].append(df_max_branch.loc[ind, "mus"])
                df_uncertanity["max_pis"].append(df_max_branch.loc[ind, "max_pis"])
                df_uncertanity["sigs"].append(df_max_branch.loc[ind, "sigs"])

        df_uncertanity = pd.DataFrame.from_dict(df_uncertanity)
        df_uncertanity.to_pickle(str(path)+"/uncertanity_max.pkl")

        df_uncertanity = pd.read_pickle(str(path)+"/uncertanity_max.pkl")



        #########drop zs_error< 0.01(1+zs)

        df_uncertanity = df_uncertanity.set_index('index')
        data_merged = pd.merge(df_uncertanity, df_max_, on="index")
        data_merged["zs"] = data_merged["zs_x"]
        data_merged = data_merged.set_index('index')

        zs_lim = 0.01 * (1 + data_merged["zs"])
        data_zslim = data_merged[data_merged["zErr"] < zs_lim]


        frac_zslim = len(data_zslim) / len(data_merged)

        n_=10


#data_pre, data,  plotname, path
        plot_peakredshift(data_merged,data_zslim, "data_zslim", path)
        outliers_peakredshift(data_merged, data_zslim,  "data_zslim", path)
        ZoomGal_outliers_peakredshift(data_merged, data_zslim,  "data_zslim", path)


#####################mean redshift

        df_uncertanity_mean = {}
        df_uncertanity_mean["deltaZ_A"] = []
        df_uncertanity_mean["index"] = []
        df_uncertanity_mean["PhotoZ_A"] = []
        df_uncertanity_mean["zs"] = []


        # df__=df__[:10]
        df__=data_train
        df_mus = 10 ** mus
        df_pis=pis
        df_sigs=sigs
        print(df_mus.shape)
        for i_data in range(len(df__)):  #

            mu_i = df_mus.iloc[i_data, :]
            p_i = df_pis.iloc[i_data, :]
            sig_i = df_sigs.iloc[i_data, :]


            photoz_sum_nom = []
            photoz_sum_denom = []
            deltaz_sum_nom = []
            deltaz_sum_denom = []
            for k in range(n_):
                photoz_sum_nom.append((mu_i[k] * p_i[k]) / sig_i[k])
                photoz_sum_denom.append((p_i[k]) / sig_i[k])
                deltaz_sum_nom.append((sig_i[k]) * (p_i[k]))
                deltaz_sum_denom.append((p_i[k]) / (sig_i[k]))



            photo_z_test = (np.sum(photoz_sum_nom)) / (np.sum(photoz_sum_denom))
            delta_z_test = np.sqrt((np.sum(deltaz_sum_nom)) / (np.sum(deltaz_sum_denom)))

            # photoz_a = (np.sum(mu_i * p_i / sig_i)) / (np.sum(p_i / sig_i))
            # deltaz_a = (np.sqrt(np.sum(sig_i * p_i)) / (np.sum(p_i / sig_i)))

            df_uncertanity_mean["PhotoZ_A"].append(photo_z_test)
            df_uncertanity_mean["deltaZ_A"].append(delta_z_test)

            df_uncertanity_mean["zs"].append(y_mem_.loc[i_data])
            df_uncertanity_mean["index"].append(i_data)

        df_uncertanity_mean = pd.DataFrame.from_dict(df_uncertanity_mean)
        df_uncertanity_mean.to_pickle(str(path)+"/uncertanity_mean.pkl")



        df_uncertanity_mean = pd.read_pickle(str(path)+"/uncertanity_mean.pkl")

        # df_uncertanity_mean=df_uncertanity_mean[:1000]

        df_uncertanity_mean["PhotoZ"] = df_uncertanity_mean["PhotoZ_A"]
        df_uncertanity_mean["deltaZ"] = df_uncertanity_mean["deltaZ_A"]

        df_uncertanity_mean = df_uncertanity_mean[df_uncertanity_mean["PhotoZ"] < 7]


        df_uncertanity_mean = pd.merge(data_train, df_uncertanity_mean, on="index")
        df_uncertanity_mean["zs"] = df_uncertanity_mean["zs_x"]
        print("photoz and zs", df_uncertanity_mean["PhotoZ"], df_uncertanity_mean["zs"])

        #########drop zs_error< 0.01(1+zs)


        df_uncertanity = df_uncertanity_mean.set_index('index')
        data_merged = df_uncertanity


        zs_lim = 0.01 * (1 + data_merged["zs"])

        data_zslim = data_merged[data_merged["zErr"] < zs_lim]



        frac_zslim = len(data_zslim) / len(data_merged)



        plot_meanredshift(data_merged,data_zslim, "data_zslim", path)
        outliers_meanredshift(data_merged, data_zslim,  "data_zslim", path)
        ZoomGal_outliers_meanredshift(data_merged, data_zslim,  "data_zslim", path)





    except Exception as e:

        print("no profile", e)

    end = time.time()

    tt = end - start
    print("time for plotting:", tt, " seconds")




