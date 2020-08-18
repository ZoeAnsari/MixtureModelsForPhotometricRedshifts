from tensorflow import keras
import mdn
import numpy as np
import time as time
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from src.Predict_from_trainedGMM_forNew import predict_from_tranied_GMM

opt = "adam"
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# #Just disables the warning, doesn't enable AVX/FMA
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def mdn_log(GMM_path, df_training, test_path,
        n_components, epoch, hn, n_ , lr, model_name, GMM_target, MDN_path, weight_con_prior, data_predict_read_path, data_predict_predict_path, X_train_GMM_newdataset):

    start = time.time()

    X_mem = df_training.drop(columns={"index", "zs", "label", "class", "zErr", "ra", "dec", "objid"})
    print("input features:", X_mem.columns)

    #############make zs as log values

    y_mem = np.log10(df_training["zs"])

    X_train, X_test, y_train, y_test = train_test_split(X_mem, y_mem, test_size=0.2,
                                                        random_state=30)

    print("y_mem before training--------------", y_mem)
    y_mem = pd.DataFrame(y_mem)  #

    #####--------------------------------MDN----------------------------------------
    # dynamic_teddy1 = hn#500#(len(X_mem.columns))#*3)#int(n_components / 2)
    print("number of hidden units in the Dense layer", hn)

    # dynamic_teddy2 = epoch#150#int(n_components)# * 3)  # *2/3)
    print("epochs", epoch)

    N_HIDDEN = hn  # len(X_mem.columns)  # number of hidden units in the Dense layer
    N_MIXES = n_#30#3*(n_components)  # number of mixture components #=n_final
    OUTPUT_DIMS = 1  # number of real-values predicted by each mixture component

    # print(X_train.shape)
    Xtrainshape = X_train.shape[1]


    # learningrate = lr#1e-3#int(n_components / 3) * 1e-4
    print("learning_rate", lr)



    #########--------------------------model just MDN----------------------------------------------------------
    model = keras.Sequential()
    model.add(keras.layers.Dense(N_HIDDEN, batch_input_shape=(None, Xtrainshape)))
    model.add(keras.layers.PReLU())
    model.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))



    print(model.summary())

    model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES),
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=['mse'])

    print("after compile")

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        batch_size=64,
                        epochs=epoch)  # , validation_split=0.15)#epochs=epoch_size_, batch_size=batch_size,
    print("after fitting")
    print(history.history)
    f = open(str(MDN_path)+"/history.pckl", 'wb')
    pickle.dump(history.history, f)
    f.close()

    model.save(str(MDN_path)+"/model.h5")

    plt.figure()

    sns.set_style('darkgrid', {'legend.frameon':True})

    print(history.history.keys())
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    # plt.title('loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig(str(MDN_path)+"/loss.png")




    ###########predict from IGMM trained model
    predict_from_tranied_GMM(GMM_path, X_train_GMM_newdataset, GMM_target, n_components,
                             weight_con_prior, data_predict_read_path, data_predict_predict_path)






    # ######predict a new dataset with MDN-----------------------------------------------------------------
    df_new=pd.read_pickle(str(data_predict_read_path))

    recorddata_new = pd.read_pickle(str(data_predict_predict_path)+"/" + str(GMM_target)+"/recorddata.pkl")
    df_label_new = pd.read_pickle(str(data_predict_predict_path)+"/" + str(GMM_target)+"/whole_IGMM_label.pkl")


    weights_new = recorddata_new["weights"]

    memprob_new = recorddata_new["prob"][0]
    df_memprob_new = pd.DataFrame(memprob_new)
    df_memprob_new["index"] = df_memprob_new.index


    df_mean_cov_new = pd.DataFrame(recorddata_new["means"])
    df_mean_cov_new["covariances"] = recorddata_new["covariances"]


    data_z_tot_new = df_new
    data_z_tot_new["index"] = data_z_tot_new.index

    data_train_new = data_z_tot_new[['g', 'Err_g', 'r', 'Err_r', 'i',
                             'Err_i', 'z', 'Err_z', 'extinction_i', 'w1', 'w1_sig',
                             'w2', 'w2_sig', 'psfMag_g', 'psfMag_r',
                             'psfMag_i', 'psfMag_z', "index", "ra", "dec", "objid"]]


    data_train_new["g-r"] = (data_train_new['g'] - data_train_new['r'])
    data_train_new["r-i"] = (data_train_new['r'] - data_train_new['i'])
    data_train_new["i-z"] = (data_train_new['i'] - data_train_new['z'])
    data_train_new["z-w1"] = (data_train_new['z'] - data_train_new['w1'])
    data_train_new["w1-w2"] = (data_train_new['w1'] - data_train_new['w2'])

    df_pl_new = data_train_new[['g', 'Err_g', 'r', 'Err_r', 'i',
                        'Err_i', 'z', 'Err_z', 'extinction_i', 'w1', 'w1_sig',
                        'w2', 'w2_sig', 'psfMag_g', 'psfMag_r',
                        'psfMag_i', 'psfMag_z', 'g-r', 'r-i',
                        'i-z', 'z-w1', 'w1-w2',"ra", "dec", "objid"]]


    df_l_new = pd.merge(df_pl_new, df_label_new, on="index")
    df_mem_new = pd.merge(df_l_new, df_memprob_new, on="index")
    df_training_new = df_mem_new


    X_new = df_training_new.drop(columns={"index", "label", "ra", "dec", "objid"})


    y_pred = model.predict(X_new)

    # Sample from the predicted distributions
    y_samples = np.apply_along_axis(mdn.sample_from_output, 1, y_pred, 1, N_MIXES, temp=1.0)
    # print("after sampling")

    # Split up the mixture parameters (for future fun)
    mus_new = np.apply_along_axis((lambda a: a[:N_MIXES]), 1, y_pred)  # the means
    sigs_new = np.apply_along_axis((lambda a: a[N_MIXES:2 * N_MIXES]), 1,
                               y_pred)  # the standard deviations
    pis_new = np.apply_along_axis((lambda a: mdn.softmax(a[2 * N_MIXES:])), 1,
                              y_pred)  # the mixture components

    mus_new = pd.DataFrame(mus_new)
    mus_new["index"] = mus_new.index
    print("mus_new------------", mus_new.head())
    print(mus_new.columns)

    sigs_new = pd.DataFrame(sigs_new)
    sigs_new["index"] = sigs_new.index
    print("sigs_new------------", sigs_new.head())
    print(sigs_new.columns)

    pis_new = pd.DataFrame(pis_new)
    pis_new["index"] = pis_new.index
    print("pis_new------------", pis_new.head())
    print(pis_new.columns)

    mus_new.to_pickle(str(data_predict_predict_path)+"/" + str(GMM_target)+"/mus_new.pkl")

    sigs_new.to_pickle(str(data_predict_predict_path)+"/" + str(GMM_target)+"/sigs_new.pkl")

    pis_new.to_pickle(str(data_predict_predict_path)+"/" + str(GMM_target)+"/pis_new.pkl")




    #####--------------------test

    # Make predictions from the model
    y_pred = model.predict(X_mem)

    # Sample from the predicted distributions
    y_samples = np.apply_along_axis(mdn.sample_from_output, 1, y_pred, 1, N_MIXES, temp=1.0)
    # print("after sampling")

    # Split up the mixture parameters (for future fun)
    mus = np.apply_along_axis((lambda a: a[:N_MIXES]), 1, y_pred)  # the means
    sigs = np.apply_along_axis((lambda a: a[N_MIXES:2 * N_MIXES]), 1,
                               y_pred)  # the standard deviations
    pis = np.apply_along_axis((lambda a: mdn.softmax(a[2 * N_MIXES:])), 1,
                              y_pred)  # the mixture components

    mus = pd.DataFrame(mus)
    mus["index"] = mus.index
    print("mus------------", mus.head())
    print(mus.columns)

    sigs = pd.DataFrame(sigs)
    sigs["index"] = sigs.index
    print("sigs------------", sigs.head())
    print(sigs.columns)

    pis = pd.DataFrame(pis)
    pis["index"] = pis.index
    print("pis------------", pis.head())
    print(pis.columns)

    mus.to_pickle(str(MDN_path) + "/mus.pkl")

    sigs.to_pickle(str(MDN_path) + "/sigs.pkl")

    pis.to_pickle(str(MDN_path) + "/pis.pkl")

    y_mem.to_pickle(str(MDN_path) + "/y_mem.pkl")

    # print("after sigs")
    print("X_test:", X_test.shape)
    print("mus", mus.shape)
    print("sigs", sigs.shape)
    print("pis", pis.shape)
    print(pis[0][0])
    print(mus[0][0])
    print(sigs[0][0])

    print("y_pred", y_pred.shape)
    print("y_mem", y_mem.shape)




    end = time.time()

    tt = end - start
    print("time for MDN:", tt, " seconds")




