
import pickle



import time as time
from sklearn.mixture import BayesianGaussianMixture

import pandas as pd
import pickle

from keras.models import load_model
import pickle
from scipy import io
from sklearn.externals import joblib as joblib
import os
import numpy as np




def predict_from_tranied_GMM(GMM_path, X_train, GMM_target, n_components, weight_con_prior, data_predict_read_path, data_predict_predict_path):
    start=time.time()

    try:
        if not os.path.exists(str(data_predict_predict_path)+"/" + str(GMM_target)):
            os.makedirs(str(data_predict_predict_path)+"/" + str(GMM_target))

        loaded_model = pickle.load(open(str(GMM_path)+"/model.sav"
                    , 'rb'))


        y_pred_15t=loaded_model.predict(X_train)
        score_sample_15t = loaded_model.score_samples(X_train)
        prob_15t = loaded_model.predict_proba(X_train)


        recordlist = []
        record = {}
        record["components"] = []
        record["weights"] = []
        record["covariances"] = []
        record["means"] = []
        record["n_compo"] = []
        record["weight_con_prior"] = []
        record["y_trainpred"] = []
        record["index"] = []
        record["score_sample"] = []
        record["prob"] = []
        record["likelihood"] = []




        if loaded_model.converged_:  # and (score_train < 0.5):# and score_test < 0.5):
            print("Converged")  # , "iter", str(it))
            print("likelihood is:", loaded_model.lower_bound_, "iter", loaded_model.n_iter_)
            # it could have several convergenced till step 100th, but just save the last
            # convergenced plot
            fit_file = str(data_predict_predict_path)+"/" + str(GMM_target)+"/model.sav"
            pickle.dump(loaded_model, open(fit_file, 'wb'))
            for k in range(n_components):
                record["n_compo"].append(n_components)
                record["weight_con_prior"].append(weight_con_prior)
                record["components"].append(k)
                record["weights"].append(loaded_model.weights_[k])
                record["covariances"].append(loaded_model.covariances_[k])
                record["means"].append(loaded_model.means_[k])
                record["y_trainpred"].append(y_pred_15t)
                record["index"].append(X_train.index)
                record["score_sample"].append(score_sample_15t)
                record["prob"].append(prob_15t)
                record["likelihood"].append(loaded_model.lower_bound_)

            recorddata = pd.DataFrame.from_dict(record)
            # recorddata["index"]=recorddata.index

            # the following lines have to be considered in run, for mean and covariances for countor plots
            # hopefully
            if not os.path.exists(str(data_predict_predict_path)+"/" +str(GMM_target)):
                os.makedirs(str(data_predict_predict_path)+"/" +str(GMM_target))


            recorddata.to_pickle(str(data_predict_predict_path)+"/" + str(GMM_target) +"/recorddata.pkl")

            #######----all rows of "index_train" and "y_trainpred" are the same, I just append it over the loop
            #######----cause a dict needs to have a same length lists
            df_label_index_train = pd.DataFrame({'index': recorddata["index"][0],
                                                 'label': recorddata["y_trainpred"][0],
                                                 }
                                                )

            df_label_index_train.to_pickle(str(data_predict_predict_path)+"/" + str(GMM_target)+"/whole_IGMM_label.pkl")

        else:

            print("likelihood is:", loaded_model.lower_bound_, "iter", loaded_model.n_iter_)
            # it could have several convergenced till step 100th, but just save the last
            # convergenced plot


            fit_file = str(data_predict_predict_path)+"/"+str(GMM_target)+"/model.sav"
            pickle.dump(loaded_model, open(fit_file, 'wb'))
            for k in range(n_components):
                record["n_compo"].append(n_components)
                record["weight_con_prior"].append(weight_con_prior)
                record["components"].append(k)
                record["weights"].append(loaded_model.weights_[k])
                record["covariances"].append(loaded_model.covariances_[k])
                record["means"].append(loaded_model.means_[k])
                record["y_trainpred"].append(y_pred_15t)
                record["index"].append(X_train.index)
                record["score_sample"].append(score_sample_15t)
                record["prob"].append(prob_15t)
                record["likelihood"].append(loaded_model.lower_bound_)

            recorddata = pd.DataFrame.from_dict(record)
            # recorddata["index"] = recorddata.index
            print("length of recorddata", len(recorddata))

            # the following lines have to be considered in run, for mean and covariances for countor plots
            # hopefully
            if not os.path.exists(str(data_predict_predict_path)+"/"+str(GMM_target)):
                os.makedirs(str(data_predict_predict_path)+"/"+str(GMM_target))

            recorddata.to_pickle(str(data_predict_predict_path)+"/"+str(GMM_target)+"/recorddata.pkl")

            #######----all rows of "index_train" and "y_trainpred" are the same, I just append it over the loop
            #######----cause a dict needs to have a same length lists
            df_label_index_train = pd.DataFrame({'index': recorddata["index"][0],
                                                 'label': recorddata["y_trainpred"][0],
                                                 }
                                                )

            df_label_index_train.to_pickle(str(data_predict_predict_path)+"/"+str(GMM_target)+ "/whole_IGMM_label.pkl")


    except Exception as e:
        print("no profile", e)




    end=time.time()
    tt= end - start
    print("timing:", tt)