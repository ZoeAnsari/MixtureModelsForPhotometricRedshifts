
import time as time
import pandas as pd
import pickle
import os



def predict_memprob(data_train, path, test_path,n_components):

    X_train=data_train
    ####BGMM clustering

    print("Features that are used in IGMM" , data_train.columns)


    mean_prec_prior = 1
    start= time.time()


    try:

        loaded_model = pickle.load(open(str(path)+"/model.sav"
                    , 'rb'))
        print("IGMM trained model: ",loaded_model)


        y_pred_15t=loaded_model.predict(X_train)
        score_sample_15t = loaded_model.score_samples(X_train)
        prob_15t = loaded_model.predict_proba(X_train)
        print(y_pred_15t)
        print(y_pred_15t.shape)
        print(score_sample_15t)
        print(score_sample_15t.shape)
        print(prob_15t)
        print(prob_15t.shape)

        recordlist = []
        record = {}
        record["components"] = []
        record["weights"] = []
        record["covariances"] = []
        record["means"] = []
        record["n_compo"] = []
        record["mean_prec_prior"] = []
        record["y_trainpred"] = []
        record["index"] = []
        record["score_sample"] = []
        record["prob"] = []
        record["likelihood"] = []




        if loaded_model.converged_:

            fit_file = str(test_path)+"/model_Converged.sav"

            pickle.dump(loaded_model, open(fit_file, 'wb'))
            for k in range(n_components):
                record["n_compo"].append(n_components)
                record["mean_prec_prior"].append(mean_prec_prior)
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

            recorddata.to_pickle(str(test_path)+"/recorddata.pkl")

            df_label_index_train = pd.DataFrame({'index': recorddata["index"][0],
                                                 'label': recorddata["y_trainpred"][0],
                                                 }
                                                )

            df_label_index_train.to_pickle(str(test_path)+"/whole_IGMM_label.pkl")

        else:





            fit_file = str(test_path)+"model.sav"

            pickle.dump(loaded_model, open(fit_file, 'wb'))
            for k in range(n_components):
                record["n_compo"].append(n_components)
                record["mean_prec_prior"].append(mean_prec_prior)
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
            print("length of recorddata", len(recorddata))


            recorddata.to_pickle(str(test_path)+"/recorddata.pkl")


            df_label_index_train = pd.DataFrame({'index': recorddata["index"][0],
                                                 'label': recorddata["y_trainpred"][0],
                                                 }
                                                )

            df_label_index_train.to_pickle(str(test_path)+"/whole_IGMM_label.pkl")


    except Exception as e:
        print("no profile", e)




    end=time.time()
    tt= end - start
    print("timing:", tt)