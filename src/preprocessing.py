import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def features_preprocessing_GMM(data_input):



    data_train = data_input[['g', 'Err_g', 'r', 'Err_r', 'i',
                             'Err_i', 'z', 'Err_z', 'extinction_i',
                             'w1', 'w1_sig','w2', 'w2_sig', 'psfMag_g', 'psfMag_r',
                               'psfMag_i', 'psfMag_z']]

    data_train["g-r"] = (data_train['g'] - data_train['r'])
    data_train["r-i"] = (data_train['r'] - data_train['i'])
    data_train["i-z"] = (data_train['i'] - data_train['z'])
    data_train["z-w1"] = (data_train['z'] - data_train['w1'])
    data_train["w1-w2"] = (data_train['w1'] - data_train['w2'])

    return data_train


def features_preprocessing_GMM_u(data_input):
    # print(data_input.columns)


    data_train = data_input[['u', 'Err_u', 'g', 'Err_g', 'r', 'Err_r', 'i',
                               'Err_i', 'z', 'Err_z', 'extinction_i', 'w1', 'w1_sig',
                               'w2', 'w2_sig', 'psfMag_u','psfMag_g', 'psfMag_r',
                               'psfMag_i', 'psfMag_z']]

    data_train["u-g"] = (data_train['u'] - data_train['g'])
    data_train["g-r"] = (data_train['g'] - data_train['r'])
    data_train["r-i"] = (data_train['r'] - data_train['i'])
    data_train["i-z"] = (data_train['i'] - data_train['z'])
    data_train["z-w1"] = (data_train['z'] - data_train['w1'])
    data_train["w1-w2"] = (data_train['w1'] - data_train['w2'])

    return data_train


def features_preprocessing_MDN(data_input, test_path):
    # print(data_input.columns)


    data_train = data_input[['g', 'Err_g', 'r', 'Err_r', 'i',
                               'Err_i', 'z', 'Err_z', 'extinction_i', 'w1', 'w1_sig',
                               'w2', 'w2_sig', 'psfMag_g', 'psfMag_r',
                               'psfMag_i', 'psfMag_z',"index", "zs", "zErr", "class", "ra", "dec", "objid"]]

    data_train["g-r"] = (data_train['g'] - data_train['r'])
    data_train["r-i"] = (data_train['r'] - data_train['i'])
    data_train["i-z"] = (data_train['i'] - data_train['z'])
    data_train["z-w1"] = (data_train['z'] - data_train['w1'])
    data_train["w1-w2"] = (data_train['w1'] - data_train['w2'])

    df_pl = data_train[['g', 'Err_g', 'r', 'Err_r', 'i',
                        'Err_i', 'z', 'Err_z', 'extinction_i', 'w1', 'w1_sig',
                        'w2', 'w2_sig', 'psfMag_g', 'psfMag_r',
                        'psfMag_i', 'psfMag_z', 'g-r', 'r-i',
                        'i-z', 'z-w1', 'w1-w2', "index", "zs", "zErr", "class", "ra", "dec", "objid"]]




    df_label = pd.read_pickle(str(test_path)+"/whole_IGMM_label.pkl")
    recorddata = pd.read_pickle(str(test_path)+"/recorddata.pkl")

    memprob = recorddata["prob"][0]
    df_memprob = pd.DataFrame(memprob)
    df_memprob["index"] = df_memprob.index


    print(df_label.head())
    df_l = pd.merge(df_pl, df_label, on="index")
    df_mem = pd.merge(df_l, df_memprob, on="index")
    print("length of df_mem", len(df_mem))

    # df_memprob = df_memprob.drop(columns={"index"})
    df_training = df_mem.dropna()

    zs = df_training["zs"]
    print("df columns:", df_training.columns)


    return df_training




def features_preprocessing_MDN_u(data_input, test_path):
    # print(data_input.columns)


    data_train = data_input[['u', 'Err_u', 'g', 'Err_g', 'r', 'Err_r', 'i',
                               'Err_i', 'z', 'Err_z', 'extinction_i', 'w1', 'w1_sig',
                               'w2', 'w2_sig', 'psfMag_u','psfMag_g', 'psfMag_r',
                               'psfMag_i', 'psfMag_z',"index", "zs", "zErr", "class", "ra", "dec", "objid"]]
    data_train["u-g"] = (data_train['u'] - data_train['g'])
    data_train["g-r"] = (data_train['g'] - data_train['r'])
    data_train["r-i"] = (data_train['r'] - data_train['i'])
    data_train["i-z"] = (data_train['i'] - data_train['z'])
    data_train["z-w1"] = (data_train['z'] - data_train['w1'])
    data_train["w1-w2"] = (data_train['w1'] - data_train['w2'])

    df_pl = data_train[['u', 'Err_u','g', 'Err_g', 'r', 'Err_r', 'i',
                        'Err_i', 'z', 'Err_z', 'extinction_i', 'w1', 'w1_sig',
                        'w2', 'w2_sig', 'psfMag_u','psfMag_g', 'psfMag_r',
                        'psfMag_i', 'psfMag_z', 'u-g','g-r', 'r-i',
                        'i-z', 'z-w1', 'w1-w2', "index", "zs", "zErr", "class", "ra", "dec", "objid"]]




    df_label = pd.read_pickle(str(test_path)+"/whole_IGMM_label.pkl")
    recorddata = pd.read_pickle(str(test_path)+"/recorddata.pkl")

    memprob = recorddata["prob"][0]
    df_memprob = pd.DataFrame(memprob)
    df_memprob["index"] = df_memprob.index


    print(df_label.head())
    df_l = pd.merge(df_pl, df_label, on="index")
    df_mem = pd.merge(df_l, df_memprob, on="index")
    print("length of df_mem", len(df_mem))

    # df_memprob = df_memprob.drop(columns={"index"})
    df_training = df_mem.dropna()

    zs = df_training["zs"]
    print("df columns:", df_training.columns)


    return df_training