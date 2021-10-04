# Auxiliary functions to extract features and data from Satellite Images and GIS files (tifs)
#
from osgeo import gdal
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import math


# Extract attribute values (Wate_Mask, Water_Depth, DEM, Slope, Roughness)from the corresponding image
def extract_Attribute_Values(tif_name, attr_name, path_sat_img_data):
    #  Open file with gdal and extract the values of each pixel
    im_name = path_sat_img_data + '/' + tif_name
    df_im = gdal.Open(im_name)
    im_array = np.array(df_im.GetRasterBand(1).ReadAsArray()).astype(np.float16)
    arr_size = im_array.shape

    # Find unique values and count the frequency of the minimum value
    #    uniq_vals, counts = np.unique(im_array, return_counts=True)
    #    print("\n ***** Unique Values: ", dict(zip(uniq_vals, counts)))

    min_val = np.amin(im_array)
    freq_min = np.count_nonzero(im_array == min_val)

    max_val = np.amax(im_array)
    freq_max = np.count_nonzero(im_array == max_val)

    # print("Minimum = ", min_val, " count min= ", freq_min, " and Maximum =", max_val, " count max = ", freq_max)

    # Change extreme minimum values with zero (min values indicates that no water exist in the pixel)
    if attr_name == 'Water_Depth':
        im_array = np.where(im_array == min_val, 0, im_array)

    # Create a temporary data frame to keep the values of the specific attribute
    attr_df = pd.DataFrame(np.reshape(im_array, arr_size[0] * arr_size[1]))
    attr_df.columns = [attr_name]

    return attr_df


# -------------------------------------------------------------------------------------------
def calc_img_resolution(tif_name, deg_km, path_sat_img_data):
    #  Open file with gdal and extract the values of each pixel
    im_name = path_sat_img_data + '/' + tif_name
    df_im = gdal.Open(im_name)
    im_array = np.array(df_im.GetRasterBand(1).ReadAsArray()).astype(np.float32)
    arr_size = im_array.shape
    # print("\n --->>> INSIDE calc_img_resolution -----")
    # print(" im_arr size = ", arr_size)

    ncol = df_im.RasterXSize  # number of columns (aka number of cells along x axis)
    nrow = df_im.RasterYSize  # number of rows (aka number of cells along y axis)
    ulx, pixelwidthx, xskew, uly, yskew, pixelheighty = df_im.GetGeoTransform()
    # print("\n ncol=", ncol, " nrow=", nrow)

    Pixel_Width = abs(pixelwidthx * deg_km)
    Pixel_Height = abs(pixelheighty * deg_km)
    Pixel_Resolution = (Pixel_Width + Pixel_Height) / 2.0

    # print('Pixel_Width=', Pixel_Width, 'Pixel_Height=', Pixel_Height, 'Pixel_Resolution=', Pixel_Resolution)

    #  Create and add 3 columns in the dataframe corresponding to the resolution of each pixel,
    #  namely Pixel_Width, Pixel_Height, Pixel_Resolution
    num_pixels = arr_size[0] * arr_size[1]
    Pixel_Width = pd.DataFrame([Pixel_Width] * num_pixels, columns=['Pixel_Width']).astype(np.float16)
    Pixel_Height = pd.DataFrame([Pixel_Height] * num_pixels, columns=['Pixel_Height']).astype(np.float16)
    Pixel_Resolution = pd.DataFrame([Pixel_Resolution] * num_pixels, columns=['Pixel_Resolution']).astype(np.float16)

    Img_Width = pd.DataFrame([ncol]*num_pixels, columns=['Img_Width']).astype(np.int)
    Img_Heigth = pd.DataFrame([nrow] * num_pixels, columns=['Img_Heigth']).astype(np.int)

    # Add 5 columns to a new temporal dataframe
    Pix_Res_df = pd.concat([Pixel_Width, Pixel_Height, Pixel_Resolution, Img_Width, Img_Heigth], axis=1)

    # print("\n ******* Pix_Res_df shape adding only resolution: ")
    # print(Pix_Res_df.shape)

    return Pix_Res_df


# -------------------------------------------------------------------------------------------
def calc_water_velocity(df, Resolution):
    # sqrt_slope = df['Roughness'] ** (1 / 2)
    # sqrt_slope = df['Roughness'].apply(np.sqrt)

    # numerator = df['Water_Depth'] * df['Pixel_Resolution']
    #
    # denominator = df['Water_Depth'].astype(float) * 2.00 + df['Pixel_Resolution']
    #
    # frac = numerator / denominator

    # water_velocity = pd.DataFrame(sqrt_slope * frac / df['Roughness'])
    # water_velocity.columns = ['Water_Velocity']
    # water_velocity = water_velocity.astype(np.float16)

    v = (1/df['Roughness'])*np.sqrt(df['Slope'])*(((df['Water_Depth'].astype(float) * Resolution)/((2*df['Water_Depth'].astype(float))+ Resolution))**(2/3))
    water_velocity = pd.DataFrame()
    water_velocity['Water_Velocity'] = v.astype(np.float16)


    return water_velocity

# -------------------------------------------------------------------------------------------
def calc_flood_hazard( df, type="eval" ):

    conditions = [
       ( df['Water_Velocity'] < 1.0 ) & ( df['Water_Depth'] < 1.0 ) ,
       ( df['Water_Velocity'] < 1.0 ) & ( df['Water_Depth'] >= 1.0 ) ,
       ( df['Water_Velocity'] >= 1.0 )  #& ( df['Water_Depth'] < 1 )
    ]
    choices_cat = ['Low Hazard', 'Medium Hazard', 'High Hazard' ]

    df['Hazard_Category'] = np.select(conditions, choices_cat, default='No Hazard')

    if type=="eval":
       choices_val = [0.4, 0.8, 1.0]
       df['Hazard'] = np.select(conditions, choices_val, default=0.0).astype(np.float16)

    return df

# -------------------------------------------------------------------------------------------
# Version 2 - more complicated rule
#
def calc_flood_hazard_v2( df ):

    df['multi'] = df['Water_Velocity']*df['Water_Depth']

    conditions = [((df['Water_Velocity'] <= 0.5) & (df['Water_Depth'] <= 1)) | ((df['Water_Velocity'] >= 0.5 ) & (df['multi'] <= 0.5)),
              ((df['Water_Velocity'] <= 0.5) & ((df['Water_Depth'] > 1) & (df['Water_Depth'] <= 2))) | ((df['Water_Velocity'] >= 0.5 ) & ((df['multi'] > 0.5) & (df['multi'] <= 1))),
              ((df['Water_Velocity'] <= 0.5) & (df['Water_Depth'] >2)) | ((df['Water_Velocity'] >= 0.5 ) & (df['multi'] > 1))
             ]

    choices_val = [0.4, 0.8, 1.0]
    choices_cat = ['Low Hazard', 'Medium Hazard', 'High Hazard' ]

    df['Hazard'] = np.select(conditions, choices_val, default=0.0).astype(np.float16)
    df['Hazard_Category'] = np.select(conditions, choices_cat, default='No Hazard')

    # print("\n ====== \n")
    # print(" df.shape =", df.shape)
    # print(" columns = ", df.columns)

    return df


# -------------------------------------------------------------------------------------------
def extract_features( tif_names, interest_dates, int_dates, path_sat_img_data, df, deg_km, mode="eval" ):
    flag_resol = False

    for im_name in range(len(tif_names)):

        print("\n ===>>>> NEW IMAGE: ", tif_names[im_name])

        # Check if in the image file name exists the specific interesting date
        if interest_dates[int_dates]['date'] in tif_names[im_name]:
            # print(" found the date ", interest_dates[int_dates]['date'])

            # Find the attribute that relates with the image
            if 'Water_Mask' in tif_names[im_name]:
                # print(" Extract values for the attribute Water_Mask ")
                attr_name = 'Water_Mask'
                wm_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df_img
                df = pd.concat([df, wm_values], axis=1)
                # print(" New size df", df.shape)

            elif 'Water_Depth' in tif_names[im_name]:
                # print(" Extract values for the attribute Water_Depth ")
                attr_name = 'Water_Depth'
                wd_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, wd_values], axis=1)
                # print(" New size df", df.shape)
        else:
            # Find the attribute that relates with the image
            if 'DEM' in tif_names[im_name]:

                # print(" Extract values for the attribute DEM")
                attr_name = 'DEM'
                dem_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, dem_values], axis=1)
                # print(" New size df", df.shape)

            elif 'Slope' in tif_names[im_name]:
                # print(" Extract values for the attribute Slope")
                attr_name = 'Slope'
                slp_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df_img
                df = pd.concat([df, slp_values], axis=1)
                # print(" New size df", df.shape)

            elif 'Roughness' in tif_names[im_name]:
                # print(" Extract values for the attribute Roughness")
                attr_name = 'Roughness'
                rough_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, rough_values], axis=1)
                # print(" New size df", df.shape)

            elif 'Aspect' in tif_names[im_name]:
                # print(" Extract values for the attribute Aspect")
                attr_name = 'Aspect'
                aspect_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, aspect_values], axis=1)
                # print(" New size df", df.shape)

            elif 'TPI' in tif_names[im_name]:
                # print(" Extract values for the attribute TPI")
                attr_name = 'TPI'
                tpi_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, tpi_values], axis=1)
                # print(" New size df", df.shape)

            elif 'TRI' in tif_names[im_name]:
                # print(" Extract values for the attribute TRI")
                attr_name = 'TRI'
                tri_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, tri_values], axis=1)
                # print(" New size df", df.shape)
            elif 'clc' in tif_names[im_name] and mode=="eval":
                print(" Extract values for Corine land cover Tif")
                attr_name = 'CorineLC'
                clc_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, clc_values], axis=1)
                # print(" New size df", df.shape)
            elif 'inhabitants' in tif_names[im_name] and mode=="eval":
                print(" Extract values for inhabitants Tif")
                attr_name = 'inh'
                inh_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, inh_values], axis=1)
                # print(" New size df", df.shape)
            elif 'fluvial' in tif_names[im_name] and mode=="eval":
                print(" Extract values for fluvial Tif")
                attr_name = 'fluvial'
                flv_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, flv_values], axis=1)
                # print(" New size df", df.shape)
            elif 'p1' in tif_names[im_name] and mode=="eval":
                print(" Extract values for p1 Tif")
                attr_name = 'p1'
                p1_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, p1_values], axis=1)
                # print(" New size df", df.shape)
            elif 'p3' in tif_names[im_name] and mode=="eval":
                print(" Extract values for p3 Tif")
                attr_name = 'p3'
                p3_values = extract_Attribute_Values(tif_names[im_name], attr_name, path_sat_img_data)

                # Concatenate the new feature to the dataframe df
                df = pd.concat([df, p3_values], axis=1)
                # print(" New size df", df.shape)

        # if flag_resol == False:
            # resol_values = calc_img_resolution(tif_names[im_name], deg_km, path_sat_img_data)
            # flag_resol = True
            # # Concatenate the new features Pixel_Width, Pixel_Height and Pixel_Resolution, Img_Width, Img_Heigth to the dataframe df
            # df = pd.concat([df, resol_values], axis=1)
            # # print(" New size df", df.shape)


    # print(" New size df_img", df.shape)
    # print("Columns names: ", df.columns)

    return df



