from datetime import date
from Machine_Learning_Processes import *
from Create_Flood_Hazard_Maps import *

def training(Muggia, Trieste, Monfalcone):
    #------------------------------------------------------------------------------------------- Step 1
    # Retrieve data from the analysis of Satellite Images from the folder "SatImAn_Data"

    # get working directory
    print("====== Start Retrieve Data....")
    get_wkd = os.getcwd()
    print(get_wkd)

    # Create a directory to store Results
    today = date.today()
    today.strftime("%b-%d-%Y")
    results_path_dir = get_wkd + "/" + "Results_New"+"/"
    os.makedirs(results_path_dir, exist_ok=True)

    # Constant in order to transform Degrees to Km
    # 1o = 111.139 meters
    deg_km = 111139

    # Get the images path
    img_data_mugia = Muggia[0]['path']
    img_data_trieste = Trieste[0]['path']
    img_data_monf = Monfalcone[0]['path']

    # List the content of directory with the images
    dir_files_list_muggia = os.listdir(img_data_mugia)
    dir_files_list_trieste = os.listdir(img_data_trieste)
    dir_files_list_monf = os.listdir(img_data_monf)

    # Create an array with tif images
    tif_names_muggia = []
    for d in range(len(dir_files_list_muggia)):
        # Search only for tif files
        if len(dir_files_list_muggia[d].split(".tif")) == 2 and dir_files_list_muggia[d].split(".tif")[1] == '':
            tif_names_muggia.append( dir_files_list_muggia[d])


    tif_names_trieste = []
    for d in range(len(dir_files_list_trieste)):
        if len(dir_files_list_trieste[d].split(".tif")) == 2 and dir_files_list_trieste[d].split(".tif")[1] == '':
            tif_names_trieste.append( dir_files_list_trieste[d])


    tif_names_monf = []
    for d in range(len(dir_files_list_monf)):
        if len(dir_files_list_monf[d].split(".tif")) == 2 and dir_files_list_monf[d].split(".tif")[1] == '':
            tif_names_monf.append(dir_files_list_monf[d])

    #------------------------------------------------------------------------------------------------
    # Step 1: extract features from tif files to create a dataset for each interesting date and place
    # MUGGIA #
    print("MUGGIA EXTRACTION DATA")
    DF_muggia = pd.DataFrame()
    for int_dates in range(len(Muggia)):
        df_img = pd.DataFrame()
        df_img = extract_features(tif_names_muggia, Muggia, int_dates, img_data_mugia, df_img, deg_km, "train")
        print("ALL DATAFRAME: ", df_img.shape)
        print("End of extract")

        idx_slope = df_img[df_img['Slope'] == float(-10000.0)].index.tolist()
        idx_water_depth = df_img[df_img['Water_Depth'] == float(0)].index.tolist()
        idx_asp = df_img[df_img['Aspect'] == float(-10000.0)].index.tolist()
        print("end idx")
        idx_union = sorted(list(set().union( idx_asp, idx_slope, idx_water_depth)))

        df_img_cleaned = pd.DataFrame(df_img.drop(index=idx_union, axis=0, inplace=False)).reset_index(drop=True)
        print("CLEAN DATAFRAME: ",df_img_cleaned.shape)
        del df_img
        print("End of cleaning the dataframe")
        # Calculate the Water Velocity per pixel based on Water_Depth, Slope, Resolution and Roughness
        print("FIND RESOLUTION")
        if int_dates == 0:
            for i in range(len(tif_names_muggia)):
                if "DEM" in tif_names_muggia[i]:
                    im = gdal.Open(img_data_mugia + "/" + tif_names_muggia[i])
                    ulx, pixelwidthx, xskew, uly, yskew, pixelheighty = im.GetGeoTransform()
                    Pixel_Width = abs(pixelwidthx * deg_km)
                    Pixel_Height = abs(pixelheighty * deg_km)
                    Pixel_Resolution = (Pixel_Width + Pixel_Height) / 2.0
                    del im, ulx, pixelwidthx, xskew, uly, yskew, pixelheighty

        print("FIND Velocity")
        wv_values_muggia = calc_water_velocity(df_img_cleaned, Pixel_Resolution)
        df_img_cleaned['Water_Velocity'] = wv_values_muggia
        del wv_values_muggia
        # Calculate the Flood Hazard and hazard cardinality
        sampling_df = cardinality(df_img_cleaned)
        del df_img_cleaned
        sampling_df = calc_flood_hazard(sampling_df, "train")

        print("\n---------------->>>>>>>>>>>>>>>>>>>>>>>> ")
        print(" sampling_df shape = ", sampling_df.shape)
        print(" Class distribution in sampling_df: ")
        print(" No hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == 'No Hazard'].shape)
        print(" Low hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == 'Low Hazard'].shape)
        print(" Medium hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == "Medium Hazard"].shape)
        print(" High hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == "High Hazard"].shape)
        print("<<<<<<<<<<<<<<<<<<<<<<<<------------------------------\n ")
        # DROP NAN
        nan = sampling_df.isnull().values.any()
        print("NAN? ", nan)
        if nan:
            print("DROP NAN")
            sampling_df = sampling_df.dropna().reset_index(drop=True)

        DF_muggia = pd.concat([DF_muggia, sampling_df], axis=0, ignore_index=False)
        print(" MUGGIA DF append the df_img_cleaned:", sampling_df.shape, "\n")
        del sampling_df


    # Trieste
    print("TRIESTE EXTRACTION DATA")
    DF_trieste = pd.DataFrame()
    for int_dates in range(len(Trieste)):
        df_img = pd.DataFrame()
        df_img = extract_features(tif_names_trieste, Trieste, int_dates, img_data_trieste, df_img, deg_km, "train")
        print("ALL DATAFRAME: ", df_img.shape)

        idx_slope = df_img[df_img['Slope'] == float(-10000.0)].index.tolist()
        idx_water_depth = df_img[df_img['Water_Depth'] == float(0)].index.tolist()
        idx_asp = df_img[df_img['Aspect'] == float(-10000.0)].index.tolist()

        idx_union = sorted(list(set().union( idx_slope, idx_asp, idx_water_depth)))

        df_img_cleaned = pd.DataFrame(df_img.drop(index=idx_union, axis=0, inplace=False)).reset_index(drop=True)
        del df_img
        print("End of cleaning the dataframe",df_img_cleaned.shape)
        # # Calculate the Water Velocity per pixel based on Water_Depth, Slope, Resolution and Roughness
        print("FIND RESOLUTION")
        if int_dates == 0:
            for i in range(len(tif_names_trieste)):
                if "DEM" in tif_names_trieste[i]:
                    im = gdal.Open(img_data_trieste + "/" + tif_names_trieste[i])
                    ulx, pixelwidthx, xskew, uly, yskew, pixelheighty = im.GetGeoTransform()
                    Pixel_Width = abs(pixelwidthx * deg_km)
                    Pixel_Height = abs(pixelheighty * deg_km)
                    Pixel_Resolution = (Pixel_Width + Pixel_Height) / 2.0
                    del im, ulx, pixelwidthx, xskew, uly, yskew, pixelheighty

        print("FIND Velocity")
        wv_values_trieste = calc_water_velocity(df_img_cleaned, Pixel_Resolution)
        df_img_cleaned['Water_Velocity'] = wv_values_trieste
        del wv_values_trieste
        # Calculate the Flood Hazard and hazard cardinality
        print("Calculate Hazard cardinality")
        sampling_df = cardinality(df_img_cleaned)
        del df_img_cleaned
        print("Calculate Hazard")
        sampling_df = calc_flood_hazard(sampling_df, "train")

        print("\n---------------->>>>>>>>>>>>>>>>>>>>>>>> ")
        print(" sampling_df shape = ", sampling_df.shape)
        print(" Class distribution in sampling_df: ")
        print(" No hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == 'No Hazard'].shape)
        print(" Low hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == 'Low Hazard'].shape)
        print(" Medium hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == "Medium Hazard"].shape)
        print(" High hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == "High Hazard"].shape)
        print("<<<<<<<<<<<<<<<<<<<<<<<<------------------------------\n ")
        # DROP NAN
        nan = sampling_df.isnull().values.any()
        print("NAN? ", nan)
        if nan:
            print("DROP NAN")
            sampling_df = sampling_df.dropna().reset_index(drop=True)

        DF_trieste = pd.concat([DF_trieste, sampling_df], axis=0, ignore_index=False)
        print(" Trieste DF append the df_img_cleaned:", DF_trieste.shape, "\n")
        del sampling_df

    # Monfalcone
    print("MONFALCONE EXTRACTION DATA")
    DF_monfalcone = pd.DataFrame()
    for int_dates in range(len(Monfalcone)):
        df_img = pd.DataFrame()
        df_img = extract_features(tif_names_monf, Monfalcone, int_dates, img_data_monf, df_img, deg_km, "train")
        print("ALL DATAFRAME: ", df_img.shape)
        print("End of extract")

        idx_dem = df_img[df_img['DEM'] == float("-inf")].index.tolist()
        idx_slope = df_img[df_img['Slope'] == float("-inf")].index.tolist()
        idx_water_depth = df_img[df_img['Water_Depth'] == float(0)].index.tolist()
        idx_asp = df_img[(df_img['Aspect'] == float(-10000.0)) | (df_img['Aspect'] == float("nan"))].index.tolist()
        idx_tpi = df_img[(df_img['TPI'] == float(-10000.0)) | (df_img['TPI'] == float("-inf")) | (df_img['TPI'] == float("inf"))].index.tolist()

        print("end idx")
        idx_union = sorted(list(set().union( idx_dem, idx_slope, idx_tpi, idx_asp, idx_water_depth)))
        # print("end union")

        df_img_cleaned = pd.DataFrame(df_img.drop(index=idx_union, axis=0, inplace=False)).reset_index(drop=True)
        del df_img
        print("End of cleaning the dataframe",df_img_cleaned.shape )
        # # Calculate the Water Velocity per pixel based on Water_Depth, Slope, Resolution and Roughness
        print("FIND RESOLUTION")
        if int_dates == 0:
            for i in range(len(tif_names_monf)):
                if "DEM" in tif_names_monf[i]:
                    im = gdal.Open(img_data_monf + "/" + tif_names_monf[i])
                    ulx, pixelwidthx, xskew, uly, yskew, pixelheighty = im.GetGeoTransform()
                    Pixel_Width = abs(pixelwidthx * deg_km)
                    Pixel_Height = abs(pixelheighty * deg_km)
                    Pixel_Resolution = (Pixel_Width + Pixel_Height) / 2.0
                    del im, ulx, pixelwidthx, xskew, uly, yskew, pixelheighty

        print("FIND Velocity")
        wv_values_monf = calc_water_velocity(df_img_cleaned, Pixel_Resolution)
        df_img_cleaned['Water_Velocity'] = wv_values_monf
        del wv_values_monf
        # Calculate the Flood Hazard and hazard cardinality
        print("Calculate Cardinality Hazard")
        sampling_df = cardinality(df_img_cleaned)
        del df_img_cleaned
        print("Calculate Hazard")
        sampling_df = calc_flood_hazard(sampling_df, "train")

        print("\n---------------->>>>>>>>>>>>>>>>>>>>>>>> ")
        print(" sampling_df shape = ", sampling_df.shape)
        print(" Class distribution in sampling_df: ")
        print(" No hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == 'No Hazard'].shape)
        print(" Low hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == 'Low Hazard'].shape)
        print(" Medium hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == "Medium Hazard"].shape)
        print(" High hazard = ", sampling_df.loc[sampling_df['Hazard_Category'] == "High Hazard"].shape)
        print("<<<<<<<<<<<<<<<<<<<<<<<<------------------------------\n ")
        # DROP NAN
        nan = sampling_df.isnull().values.any()
        print("NAN? ", nan)
        if nan:
            print("DROP NAN")
            sampling_df = sampling_df.dropna().reset_index(drop=True)

        DF_monfalcone = pd.concat([DF_monfalcone, sampling_df], axis=0, ignore_index=False)
        print(" Monfalcone DF append the df_img_cleaned:", DF_monfalcone.shape, "\n")
        del sampling_df

    # Concat all Dataframes
    # print("CONCAT OF DATAFRAMES")
    frames = [DF_muggia,DF_trieste,DF_monfalcone]
    result = pd.concat(frames)
    print("FINAL result DATAFRAME",result.shape)

    print("SPLITTING THE DATASET")
    # Select attributes for analysis
    attrbs = ["Slope", "Aspect", "TPI", "TRI", "Water_Depth", "DEM", "Roughness", "Water_Mask", "Water_Velocity"]

    # Targets: "Hazard", "Hazard_Category"
    # targs: denotes the class attribute
    targs = 'Hazard_Category'

    # perc: is the percentage for testing (randomly chosen)
    perc = 0.30
    P_train, P_test, T_train, T_test, minmax_scaler = preprocess( result, attrbs, targs, perc )

    print("\n Training set:")
    print( P_train.shape)
    print( P_train.head(20))
    print( T_train.shape )
    print( T_train.head(20))

    print("\n Testing set:")
    print( P_test.shape )
    print( P_test.head(20))
    print( T_test.shape )
    print( T_test.head(20))

    #------------------------------------------------------------------------------------------------
    # Step 3: Create and Train Machine Learning models

    #Run classifiers SVM with several parameters
    pipeline_SVM = Pipeline((('clf', SVC()), ), verbose=True)

    # parameters_SVM = {
    #     'clf__C': [20.0],
    #     'clf__kernel': ['rbf', 'poly', 'sigmoid'],
    #     'verbose': [True]
    # }
    parameters_SVM = {
        'clf__C': [20.0],
        'clf__kernel': ['sigmoid'],
        'verbose': [True]
    }

    # pipeline_NB = Pipeline((('clf', MultinomialNB()), ), verbose=True)
    # parameters_NB = {'clf__alpha': [0.01, 0.1, 1.0]}
    pipeline_NB = Pipeline((('clf', MultinomialNB()),), verbose=True)
    parameters_NB = {'clf__alpha': [1.0]}


    # pipeline_RF = Pipeline((('clf', RandomForestClassifier()), ), verbose=True)
    # parameters_RF = {
    #     'clf__n_estimators': [50, 100, 200, 500],
    #     'clf__oob_score': [True],
    #     'clf__random_state': [1],
    #     'clf__criterion': ['gini', 'entropy'],
    #     'clf__max_features': ['auto', 'log2', 'sqrt', None]
    # }

    pipeline_RF = Pipeline((('clf', RandomForestClassifier()),), verbose=True)
    parameters_RF = {
        'clf__n_estimators': [50],
        'clf__oob_score': [True],
        'clf__random_state': [1],
        'clf__criterion': ['gini'],
        'clf__max_features': ['log2']
    }

    ml_names = [ 'SVM','Random_Forest', 'Naive_Bayes']
    pars = [parameters_SVM, parameters_RF,  parameters_NB]
    pips = [pipeline_SVM, pipeline_RF,  pipeline_NB]

    model_list = []
    for i in range(len(pars)):
        best_param = {}
        print("!!!!!!!   ",ml_names[i],"   !!!!!!!")
        outpt_fname = results_path_dir + 'Experiment_Results' + '_' + ml_names[i] + '.csv'
        model_name = results_path_dir + 'Trained_' + ml_names[i] + '_model'

        # Cross Validation k-folds
        cv_folds = 10

        targ_categ = pd.unique(result[targs])
        print("\n Target categories: ", targ_categ)

        ML_results,best_param = create_ML_models(pars[i], cv_folds, pips[i], P_train, P_test, T_train, T_test, targ_categ, outpt_fname, model_name, best_param, minmax_scaler)

        T_pred = ML_results['Targets_predicted_best']
        # classifier = ML_results['Model']

        conf_mat, conf_mat_DF = evaluate_confusion_matrix(T_test, T_pred, targ_categ, outpt_fname, ml_names[i], results_path_dir )
        print('\n Confusion Matrix to evaluate best model for ', ml_names[i])
        print(conf_mat_DF)
        print('\n ---------------------------------------------\n')
        model_list.append(best_param)

    scores = [x['score'] for x in model_list]
    paths = [x['model'] for x in model_list]
    max_value = max(scores)
    max_index = scores.index(max_value)
    best_score = scores.pop(max_index)
    best_path = paths.pop(max_index)
    print("Best model is in: ", best_path, "with score: ", best_score)
    for q in range(len(paths)):
        os.remove(paths[q])

    print("\n TRAINING IS TERMINATED NORMALLY!!! \n")
    return