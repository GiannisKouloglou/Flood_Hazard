import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Machine_Learning_Processes import *
from old.Create_Flood_Hazard_Maps import *
from sklearn.preprocessing import LabelEncoder
import os

get_wkd = os.getcwd()
print(get_wkd)

mug_new = get_wkd + '/../' + 'SatImAn_Data' + '/' + "Muggia-City"
tri_new = get_wkd + '/../' + 'SatImAn_Data' + '/' + "Trieste-City"
monf = get_wkd + '/../' + 'SatImAn_Data' + '/' + "Monfalcone-City"

Muggia_new= [{'date': '20190923', 'dt_flag_used': False, 'place': 'Muggia-City', "path": mug_new},
                    {'date': '20191115', 'dt_flag_used': False, 'place': 'Muggia-City', "path": mug_new},
                    {'date': '20191117', 'dt_flag_used': False, 'place': 'Muggia-City', "path": mug_new}
                    ]

Trieste_new = [ {'date': '20190924', 'dt_flag_used': False, 'place': 'Trieste-City', "path": tri_new},
    {'date': '20191115', 'dt_flag_used': False, 'place': 'Trieste-City', "path": tri_new},
    {'date': '20191223', 'dt_flag_used': False, 'place': 'Trieste-City', "path": tri_new}
]


Monfalcone = [{'date': '20181029', 'dt_flag_used': False, 'place': 'Monfalcone-City', "path": monf},
                    {'date': '20190923', 'dt_flag_used': False, 'place': 'Monfalcone-City', "path": monf},
                    {'date': '20191117', 'dt_flag_used': False, 'place': 'Monfalcone-City', "path": monf},
                    {'date': '20191223', 'dt_flag_used': False, 'place': 'Monfalcone-City', "path": monf}
                    ]
#------------------------------------------------------------------------------------------- Step 1
# Retrieve data from the analysis of Satellite Images from the folder "SatImAn_Data"

# get working directory
print("====== Start Retrieve Data....")

# Create a directory to store Results
results_path_dir = get_wkd + "/" + "Results_New_Keras/"
os.makedirs(results_path_dir, exist_ok=True)

# variable indicators to the names
var_ids_names = [ {'attr': 'Water_Mask', 'attr_flag_used' : False},
                  {'attr': 'Water_Depth', 'attr_flag_used' : False},
                  {'attr': 'Roughness', 'attr_flag_used' : False},
                  {'attr': 'Slope', 'attr_flag_used' : False},
                  {'attr': 'DEM', 'attr_flag_used' : False},
                  {'attr': 'Aspect', 'attr_flag_used' : False},
                  {'attr': 'TRI', 'attr_flag_used' : False},
                  {'attr': 'TPI', 'attr_flag_used' : False}
                ]

# Constant in order to transform Degrees to Km
# 1o = 111.139 meters
deg_km = 111139

# Get the images path
img_data_mug_new = Muggia_new[0]['path']
img_data_tri_new = Trieste_new[0]['path']
img_data_monf = Monfalcone[0]['path']

# List the content of directory with the images
dir_files_list_mug_new = os.listdir(img_data_mug_new)
dir_files_list_tri_new = os.listdir(img_data_tri_new)
dir_files_list_monf = os.listdir(img_data_monf)

# Create an array with tif images
tif_names_mug_new = []
for d in range(len(dir_files_list_mug_new)):
    # Search only for tif files
    if len(dir_files_list_mug_new[d].split(".tif")) == 2 and dir_files_list_mug_new[d].split(".tif")[1] == '':
        tif_names_mug_new.append( dir_files_list_mug_new[d])


tif_names_tri_new = []
for d in range(len(dir_files_list_tri_new)):
    if len(dir_files_list_tri_new[d].split(".tif")) == 2 and dir_files_list_tri_new[d].split(".tif")[1] == '':
        tif_names_tri_new.append( dir_files_list_tri_new[d])


tif_names_monf = []
for d in range(len(dir_files_list_monf)):
    if len(dir_files_list_monf[d].split(".tif")) == 2 and dir_files_list_monf[d].split(".tif")[1] == '':
        tif_names_monf.append(dir_files_list_monf[d])

#------------------------------------------------------------------------------------------------
# Step 1: extract features from tif files to create a dataset for each interesting date and place
# MUGGIA NEW #
print("MUGGIA NEW EXTRACTION DATA")
DF_muggia_new = pd.DataFrame()
for int_dates in range(len(Muggia_new)):
    df_img = pd.DataFrame()
    df_img = extract_features(tif_names_mug_new, Muggia_new, int_dates, img_data_mug_new, df_img, deg_km,"train")
    print("ALL DATAFRAME: ", df_img.shape)
    print("End of extract")

    # idx_dem = df_img[df_img['DEM'] == float(-10000.0)].index.tolist()
    # idx_slope_inf = df_img[df_img['Slope'] == float('inf')].index.tolist()
    idx_slope = df_img[df_img['Slope'] == float(-10000.0)].index.tolist()
    idx_water_depth = df_img[df_img['Water_Depth'] == float(0)].index.tolist()
    # idx_water_rough = df_img[df_img['Roughness'] <= float(-32770.0)].index.tolist()
    idx_asp = df_img[df_img['Aspect'] == float(-10000.0)].index.tolist()
    # idx_tpi = df_img[df_img['TPI'] == float(-10000.0)].index.tolist()
    # idx_tri = df_img[df_img['TRI'] == float(-10000.0)].index.tolist()
    print("end idx")
    # idx_union = sorted(list(set().union(idx_water_rough, idx_dem, idx_slope, idx_slope_inf, idx_asp, idx_tpi, idx_tri)))
    idx_union = sorted(list(set().union( idx_asp, idx_slope, idx_water_depth)))

    df_img_cleaned = pd.DataFrame(df_img.drop(index=idx_union, axis=0, inplace=False)).reset_index(drop=True)
    print("CLEAN DATAFRAME: ",df_img_cleaned.shape)
    del df_img
    print("End of cleaning the dataframe")
    # # Calculate the Water Velocity per pixel based on Water_Depth, Slope, Resolution and Roughness
    print("FIND RESOLUTION")
    if int_dates == 0:
        for i in range(len(tif_names_mug_new)):
            if "DEM" in tif_names_mug_new[i]:
                im = gdal.Open(img_data_mug_new + "/" + tif_names_mug_new[i])
                ulx, pixelwidthx, xskew, uly, yskew, pixelheighty = im.GetGeoTransform()
                Pixel_Width = abs(pixelwidthx * deg_km)
                Pixel_Height = abs(pixelheighty * deg_km)
                Pixel_Resolution = (Pixel_Width + Pixel_Height) / 2.0
                del im, ulx, pixelwidthx, xskew, uly, yskew, pixelheighty

    print("FIND Velocity")
    wv_values_mug_new = calc_water_velocity(df_img_cleaned, Pixel_Resolution)
    df_img_cleaned['Water_Velocity'] = wv_values_mug_new
    del wv_values_mug_new
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

    DF_muggia_new = pd.concat([DF_muggia_new, sampling_df], axis=0, ignore_index=False)
    print(" MUGGIA NEW DF append the df_img_cleaned:", sampling_df.shape, "\n")
    del sampling_df


# Trieste NEW #
print("TRIESTE NEW EXTRACTION DATA")
DF_trieste_new = pd.DataFrame()
for int_dates in range(len(Trieste_new)):
    df_img = pd.DataFrame()
    df_img = extract_features(tif_names_tri_new, Trieste_new, int_dates, img_data_tri_new, df_img, deg_km,"train")
    print("ALL DATAFRAME: ", df_img.shape)

    # idx_dem = df_img[df_img['DEM'] == float(-10000)].index.tolist()
    idx_slope = df_img[df_img['Slope'] == float(-10000.0)].index.tolist()
    idx_water_depth = df_img[df_img['Water_Depth'] == float(0)].index.tolist()
    # idx_water_rough = df_img[df_img['Roughness'] <= float(-32770.0)].index.tolist()
    idx_asp = df_img[df_img['Aspect'] == float(-10000.0)].index.tolist()
    # idx_tpi = df_img[df_img['TPI'] == float(-10000.0)].index.tolist()
    # idx_tri = df_img[df_img['TRI'] == float(-10000.0)].index.tolist()

    # idx_union = sorted(list(set().union(idx_water_rough, idx_dem, idx_slope, idx_asp, idx_tpi, idx_tri)))
    idx_union = sorted(list(set().union( idx_slope, idx_asp, idx_water_depth)))

    df_img_cleaned = pd.DataFrame(df_img.drop(index=idx_union, axis=0, inplace=False)).reset_index(drop=True)
    del df_img
    print("End of cleaning the dataframe",df_img_cleaned.shape)
    # # Calculate the Water Velocity per pixel based on Water_Depth, Slope, Resolution and Roughness
    print("FIND RESOLUTION")
    if int_dates == 0:
        for i in range(len(tif_names_tri_new)):
            if "DEM" in tif_names_tri_new[i]:
                im = gdal.Open(img_data_tri_new + "/" + tif_names_tri_new[i])
                ulx, pixelwidthx, xskew, uly, yskew, pixelheighty = im.GetGeoTransform()
                Pixel_Width = abs(pixelwidthx * deg_km)
                Pixel_Height = abs(pixelheighty * deg_km)
                Pixel_Resolution = (Pixel_Width + Pixel_Height) / 2.0
                del im, ulx, pixelwidthx, xskew, uly, yskew, pixelheighty

    print("FIND Velocity")
    wv_values_tri_new = calc_water_velocity(df_img_cleaned, Pixel_Resolution)
    df_img_cleaned['Water_Velocity'] = wv_values_tri_new
    del wv_values_tri_new
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

    DF_trieste_new = pd.concat([DF_trieste_new, sampling_df], axis=0, ignore_index=False)
    print(" Trieste NEW DF append the df_img_cleaned:", DF_trieste_new.shape, "\n")
    del sampling_df


# Monfalcone #
print("MONFALCONE NEW EXTRACTION DATA")
DF_monfalcone = pd.DataFrame()
for int_dates in range(len(Monfalcone)):
    df_img = pd.DataFrame()
    df_img = extract_features(tif_names_monf, Monfalcone, int_dates, img_data_monf, df_img, deg_km, "train")
    print("ALL DATAFRAME: ", df_img.shape)
    print("End of extract")

    idx_dem = df_img[df_img['DEM'] == float("-inf")].index.tolist()
    idx_slope = df_img[df_img['Slope'] == float("-inf")].index.tolist()
    idx_water_depth = df_img[df_img['Water_Depth'] == float(0)].index.tolist()
    # idx_water_rough = df_img[df_img['Roughness'] <= float(-32770.0)].index.tolist()
    idx_asp = df_img[(df_img['Aspect'] == float(-10000.0)) | (df_img['Aspect'] == float("nan"))].index.tolist()
    idx_tpi = df_img[(df_img['TPI'] == float(-10000.0)) | (df_img['TPI'] == float("-inf")) | (df_img['TPI'] == float("inf"))].index.tolist()
    # idx_tri = df_img[(df_img['TRI'] == float(-10000.0)) | (df_img['TRI'] == float("inf"))].index.tolist()

    print("end idx")
    # idx_union = sorted(list(set().union(idx_water_rough, idx_dem, idx_slope, idx_asp, idx_tpi, idx_tri)))
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

# #Concat all Dataframes
# print("CONCARNATION OF DATAFRAMES")
frames = [DF_muggia_new,DF_trieste_new,DF_monfalcone]
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
X_train, X_test, Y_train, Y_test, minmax_scaler = preprocess( result, attrbs, targs, perc )

print("\n Training set:")
print( X_train.shape)
print( X_train.head(20))
print( X_test.shape )
print( X_test.head(20))

print("\n Testing set:")
print( Y_test.shape )
print( Y_test.head(20))
print( Y_train.shape )
print( Y_train.head(20))

encoder_train = LabelEncoder()
encoder_train.fit(Y_train)
encoded_Y_train = encoder_train.transform(Y_train)
myset_train = set(encoded_Y_train)
print("my set train")
print(myset_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = tf.keras.utils.to_categorical(encoded_Y_train)

encoder_test = LabelEncoder()
encoder_test.fit(Y_test)
encoded_Y_test = encoder_test.transform(Y_test)
myset_test = set(encoded_Y_test)
print("my set test")
print(myset_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_test = tf.keras.utils.to_categorical(encoded_Y_test)

units = [1,2,4,6,8]
epoch = [100,300,500]
func = ['relu','sigmoid']

for i in range(len(units)):
    for j in range(len(epoch)):
        for q in range(len(func)):

            #create model
            model = tf.keras.models.Sequential()
            #     tf.keras.layers.Flatten(input_shape=(1, nBands))
            model.add(tf.keras.layers.Dense(units[i], input_dim=9, activation=func[q]))
            model.add(tf.keras.layers.Dense(3, activation='softmax'))
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # Fit the model
            history = model.fit(X_train, dummy_y_train, epochs=epoch[j], batch_size=100, verbose=1)
            # evaluate the model
            scores = model.evaluate(X_train, dummy_y_train, verbose=1)
            accuracy = scores[1]*100
            loss = history.history['loss']
            loss = loss[-1]
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

            # # serialize model to JSON
            # model_json = model.to_json()
            # with open("model.json", "w") as json_file:
            #     json_file.write(model_json)
            # # serialize weights to HDF5
            # model.save_weights("model.h5")
            # print("Saved model to disk")

            # load json and create model
            # json_file = open('model.json', 'r')
            # loaded_model_json = json_file.read()
            # json_file.close()
            # loaded_model = tf.keras.models.model_from_json(loaded_model_json)
            # # load weights into new model
            # loaded_model.load_weights("model.h5")
            # print("Loaded model from disk")
            # summary = loaded_model.summary()

            # evaluate loaded model on test data
            # loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # score = loaded_model.evaluate(X_test, dummy_y_test, verbose=1)
            # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


            # preds_classes_1 = loaded_model.predict_classes(X_test)

            preds_classes_2 = np.argmax((model.predict(X_test) > 0.5).astype("int32"), axis=-1)
            test = pd.DataFrame(encoder_test.inverse_transform(preds_classes_2), columns = ['Hazard Category'])

            confusion = confusion_matrix(Y_test, test)
            print('Confusion Matrix\n')
            print(confusion)

            #importing accuracy_score, precision_score, recall_score, f1_score
            print('\nAccuracy: {:.2f}\n'.format(accuracy_score(Y_test, test)))

            print('Micro Precision: {:.2f}'.format(precision_score(Y_test, test, average='micro')))
            print('Micro Recall: {:.2f}'.format(recall_score(Y_test, test, average='micro')))
            print('Micro F1-score: {:.2f}\n'.format(f1_score(Y_test, test, average='micro')))

            print('Macro Precision: {:.2f}'.format(precision_score(Y_test, test, average='macro')))
            print('Macro Recall: {:.2f}'.format(recall_score(Y_test, test, average='macro')))
            print('Macro F1-score: {:.2f}\n'.format(f1_score(Y_test, test, average='macro')))

            print('Weighted Precision: {:.2f}'.format(precision_score(Y_test, test, average='weighted')))
            print('Weighted Recall: {:.2f}'.format(recall_score(Y_test, test, average='weighted')))
            print('Weighted F1-score: {:.2f}'.format(f1_score(Y_test, test, average='weighted')))

            print('\nClassification Report\n')
            outpt_fname = results_path_dir + 'Experiment_Results' + '_' + str(units[i]) + "_" + str(epoch[j]) + "_" + str(func[q]) + '.csv'

            file_experiment_results = open(outpt_fname, 'a+')
            print("Loss: ",loss,file=file_experiment_results)
            print("Accuracy: ",accuracy, file=file_experiment_results)
            print(classification_report(Y_test, test, target_names=['High Hazard', 'Low Hazard', 'Medium Hazard']),
                  file=file_experiment_results)

            # Save confusion matrix fig
            targ_categ = pd.unique(result[targs])

            file_experiment_results.close()

            conf_mat, conf_mat_DF = evaluate_confusion_matrix(Y_test, test, targ_categ, outpt_fname, str(units[i]) + "_" + str(epoch[j]) + "_" + str(func[q]) , results_path_dir)

print("asd")
