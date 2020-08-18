import pandas as pd
import numpy as np
from src.preprocessing import features_preprocessing_GMM, features_preprocessing_MDN
from src.Predict_from_trainedGMM import predict_memprob
from src.MDN_log import mdn_log
from src.Result_of_MDN import MDN_result
from src.Table_prediction_forNew import MDN_result_forNew



# # # # ###GMM parameters
target="Supervised_cl"
# # y_train=data_input["class"]
path="Data/TrainedGMM/"+ str(target) +"/No_u"
path_plot="Data/TrainedGMM/"+ str(target) +"/No_u/plot"
n_components= 52
weight_con_prior = 509796.0


#####################----------------------------------------------------------------------------

# # #####predict test datase
df_test=pd.read_pickle("Data/Test_data/No_u/test_data.pkl")
df_test_GMM= features_preprocessing_GMM(df_test) ###add color features
test_path="Data/Test_data/No_u/"+str(target)
test_path_plot="Data/Test_data/No_u/"+str(target)+"/plot"




#######predict the IGMM membership probabilites for the spectroscopic sample---------------------
predict_memprob(df_test_GMM,path, test_path,n_components)




# ########preprocessing for MDN-------------------------------------------------------------------
data_MDN=features_preprocessing_MDN(df_test, test_path)





# # ##################MDN for log_redshift-------------------------------------------------------
#####MDN hyperparameters
epoch = 300
hn = 518
n_ = 10
lr = 1e-4

model_name = "64_PReLU_"
GMM_target = "cl"
GMM_path_="GMM_Supervised_cl"
MDN_path="Data/TrainedMDN/"+str(GMM_path_)+"/No_u"
GMM_path=path


# TODO: define the path for reading and predicting the newdataset
data_predict_read_path="Data/SDSS/SDSSALL.pkl"
data_predict_predict_path="Data/SDSS/Predicted"
data_new=pd.read_pickle(data_predict_read_path)
X_train_GMM_newdataset=features_preprocessing_GMM(data_new)
# mdn_log(GMM_path, data_MDN, test_path,
#         n_components, epoch, hn, n_ , lr, model_name, GMM_target, MDN_path, weight_con_prior,data_predict_read_path, data_predict_predict_path, X_train_GMM_newdataset)

mus=pd.read_pickle(str(MDN_path) + "/mus.pkl")
sigs=pd.read_pickle(str(MDN_path) + "/sigs.pkl")
pis=pd.read_pickle(str(MDN_path) + "/pis.pkl")
data=df_test
path_=MDN_path


# MDN_result(mus, sigs, pis,MDN_path, path_, data, weight_con_prior)



########photoz estimation table
mus_new=pd.read_pickle(str(data_predict_predict_path)+"/" + str(GMM_target)+"/mus_new.pkl")
sigs_new=pd.read_pickle(str(data_predict_predict_path)+"/" + str(GMM_target)+"/sigs_new.pkl")
pis_new=pd.read_pickle(str(data_predict_predict_path)+"/" + str(GMM_target)+"/pis_new.pkl")

path_new=str(data_predict_predict_path)+"/" + str(GMM_target)


MDN_result_forNew(mus_new, sigs_new, pis_new,MDN_path, path_new, data_new,weight_con_prior, data_predict_predict_path)

