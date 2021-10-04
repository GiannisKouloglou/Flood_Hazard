# FLOOD_SAT PROJECT
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from os.path import basename

from Create_Flood_Hazard_Maps import *
pd.options.mode.chained_assignment = None  # default='warn'


def evaluation(model_path,interest_dates, results_path_dir):
    #------------------------------------------------------------------------------------------- Step 1
    # Retrieve data from the analysis of Satellite Images from the folder "SatImAnal_Data"

    # get working directory
    print("====== Start Retrieve Data....")
    get_wkd = os.getcwd()
    print(get_wkd)
    model_dict = pickle.load(open(model_path, "rb"))
    classifier = model_dict['Model']
    scaler = model_dict['Scaler']

    path_shapefiles = get_wkd + "/" + "Shapefiles"
    if not os.path.exists(path_shapefiles):
        os.makedirs(path_shapefiles, exist_ok=True)


    # Get the images path
    # Constant in order to transform Degrees to Km
    # 1o = 111.139 meters
    deg_km = 111139

    # Create the subfolder path-name to find the images
    print(interest_dates)
    path_sat_img_data = interest_dates[0]['path'] +"/"
    print(path_sat_img_data)

    # List the content of directory with the images
    dir_files_list = os.listdir(path_sat_img_data)
    # print(dir_files_list)

    # Create an array with tif images
    tif_names = []
    for d in range(len(dir_files_list)):
        # Search only for tif files
        if len(dir_files_list[d].split(".tif")) == 2 and dir_files_list[d].split(".tif")[1] == '':
            tif_names.append( dir_files_list[d] )
            # print(tif_names[d])

    #-------------------------------------------------------------------------------------------------------------------------------
    #        ****** E V A L U A T I O N  P H A S E  *******
    #-------------------------------------------------------------------------------------------------------------------------------
    # Create flood hazard maps by all the images (water masks or depth) in the interest dates
    print("#  ****** E V A L U A T I O N  P H A S E  ******* ")
    print("DEM AND RESOLUTION")
    # Find DEM tiff for guidance to the trip for the map
    for im_name in range(len(tif_names)):
        if "DEM" in tif_names[im_name]:
            print("\n ===>>>> Image name as guidance: ")
            dem_name = path_sat_img_data + '/' + tif_names[im_name]


    # Open image with gdal
    gdal_guidance_image = gdal.Open(dem_name)
    ulx, pixelwidthx, xskew, uly, yskew, pixelheighty = gdal_guidance_image.GetGeoTransform()
    Pixel_Width = abs(pixelwidthx * deg_km)
    Pixel_Height = abs(pixelheighty * deg_km)
    Pixel_Resolution = (Pixel_Width + Pixel_Height) / 2.0

    rasterXSize = gdal_guidance_image.RasterXSize
    rasterYSize = gdal_guidance_image.RasterYSize
    format = 'GTiff'

    # For each one of the dates that a flood has occured...create a dynamic flood hazard map
    for int_dates in range(len(interest_dates)):
        print("\n Repeat: ", int_dates, " --- Date: ", interest_dates[int_dates]['date'], " --- Place: ",interest_dates[int_dates]['place'])

        #eval
        # Filenames to store each one of the evaluation results and create the confusion matrix image
        res_fname = results_path_dir + 'Evaluation_Report' + '_' + interest_dates[int_dates]['place'] + '_' + interest_dates[int_dates]['date'] + '.csv'
        # Opening file to write evaluation results
        file_eval_results = open(res_fname, 'a+')
        #eval

        df_img = pd.DataFrame()
        df_img = extract_features(tif_names, interest_dates, int_dates, path_sat_img_data, df_img, deg_km)
        # Filenames to store each one of the evaluation results and create the confusion matrix image
        conf_mat_name = results_path_dir + 'Confusion_Matrix' + '_' + interest_dates[int_dates]['place'] + '_' + interest_dates[int_dates]['date'] + '.png'
        output_file = results_path_dir + 'Flood_Hazard_Map' + "_" + interest_dates[int_dates]['place'] + '_' + interest_dates[int_dates]['date']  #+ '.tiff'

        # Extract the portion of the data frame DF corresponds to the specific interest date
        #
        df_img_WH = {'Width': int(rasterXSize), 'Height': int(rasterYSize)}
        step = df_img_WH['Width'] * df_img_WH['Height']

        #------------------- Testing reasons
        print("df_img_WH - Width=", df_img_WH['Width'])
        print("df_img_WH - Heigth=", df_img_WH['Height'])
        print("step = ", step)
        print("rasterXSize = ", rasterXSize)
        print("rasterYSize = ", rasterYSize)
        #----------------------------------

        # Call function to create colored flood hazard map
        # targs: denotes the class attribute
        targs = 'Hazard_Category'
        # Select attributes for analysis
        attrbs = ["Slope", "Aspect", "TPI", "TRI", "Water_Depth", "DEM", "Roughness", "Water_Mask", "Water_Velocity"]

        results_for_maps, colored_hazard_map_3D = create_flood_hazard_maps(classifier, scaler, df_img, attrbs, targs, df_img_WH, conf_mat_name, Pixel_Resolution)

        # evaluation process
        # Store evaluation results and confusion matrix to output file
        print("\n Evaluation Results for the Location: ", interest_dates[int_dates]['place'], " --- Date: ", interest_dates[int_dates]['date'], file=file_eval_results)
        print(results_for_maps['Evaluation_Measures_Results'], file=file_eval_results)
        print("\n ====== CONFUSION MATRIX ======", file=file_eval_results)
        print( results_for_maps['Confusion_Matrix'], file=file_eval_results)
        print(" <<<<<<<<<================== \n", file=file_eval_results)

        # # Display Confusion Matrix and store it. Save the Evaluation Results in file
        # cm_display = ConfusionMatrixDisplay( results_for_maps['Confusion_Matrix'], display_labels=list(results_for_maps['Confusion_Matrix'].columns) )
        # cm_display = cm_display.plot(include_values=True, cmap='Blues', ax=None, values_format='d', xticks_rotation='horizontal', colorbar=False)
        # print(" \n Location of the Confusion Matrix is: ", conf_mat_name)
        # plt.savefig( conf_mat_name )
        # plt.close()

        # # Store 3D colored hazard map to file. Reshape file to 1D in which row exist a triple of RGB color. The size of the new array to store will be
        # # colored_hazard_map_3D.shape[0] * colored_hazard_map_3D.shape[1] <=> Width * Height

        # save csv ** dont save :) too much space
        # fname_hazard_map_3d = results_path_dir + 'Colored_Hazard_map_3d' + '_' + interest_dates[int_dates]['place'] + '_' + interest_dates[int_dates]['date'] + '_' + '.csv'
        #
        # newdim = colored_hazard_map_3D.shape[0] * colored_hazard_map_3D.shape[1]
        # colored_hazard_map_1D = colored_hazard_map_3D.reshape(newdim, -1)
        #
        # # Saving reshaped array to file.
        # np.savetxt(fname_hazard_map_3d, colored_hazard_map_1D)
        # print("\n The derived colored flood hazard map (1D) has stored to file: ", fname_hazard_map_3d)
        # print(" colored_hazard_map_3D shape = ", colored_hazard_map_3D.shape)
        #
        # fname_hazard_map_3d = results_path_dir + 'Colored_Hazard_map_3d' + '_' + interest_dates[int_dates]['place'] + '_' + interest_dates[int_dates]['date'] + '.csv'
        #
        # newdim = colored_hazard_map_3D.shape[0] * colored_hazard_map_3D.shape[1]
        # colored_hazard_map_1D = colored_hazard_map_3D.reshape(newdim, -1)
        #
        # # Saving reshaped array to file.
        # print(" colored_hazard_map_3D shape = ", colored_hazard_map_3D.shape)
        # # Saving reshaped array to file.
        # np.savetxt(fname_hazard_map_3d, colored_hazard_map_1D)
        # print("\n The derived colored flood hazard map (1D) has stored to file: ", fname_hazard_map_3d)
        # print(" colored_hazard_map_3D shape = ", colored_hazard_map_3D.shape)
        # save csv
        # evaluation process

        # save to Tif
        saveGeoTiffRGB(gdal_guidance_image, output_file, format, rasterXSize, rasterYSize, colored_hazard_map_3D)

        # # tiff to shapefiles HAZARD
        # output_water_bodies_shapefile_haz = 'Flood_Hazard_Map' + "_" + interest_dates[int_dates]['place'] + '_' + interest_dates[int_dates]['date'] + ".shp"
        # cmd = 'gdal_polygonize.py -mask %s %s -f "ESRI Shapefile" %s' % (output_file+".tif", output_file+".tif",
        #                                                                  path_shapefiles + "/" + output_water_bodies_shapefile_haz)
        # os.system(cmd)
        #
        # with ZipFile(path_shapefiles + "/" + output_water_bodies_shapefile_haz.split(".")[0] + '.zip', 'w') as myzip:
        #     filePath_shp = path_shapefiles + "/" + output_water_bodies_shapefile_haz
        #     filePath_dbf = path_shapefiles + "/" + output_water_bodies_shapefile_haz.split(".")[0]+".dbf"
        #     filePath_prj = path_shapefiles + "/" + output_water_bodies_shapefile_haz.split(".")[0]+".prj"
        #     filePath_shx = path_shapefiles + "/" + output_water_bodies_shapefile_haz.split(".")[0]+".shx"
        #     # Add file to zip
        #     myzip.write(filePath_shp, basename(filePath_shp))
        #     myzip.write(filePath_dbf, basename(filePath_dbf))
        #     myzip.write(filePath_prj, basename(filePath_prj))
        #     myzip.write(filePath_shx, basename(filePath_shx))
        #     # Delete them
        #     os.remove(filePath_shp)
        #     os.remove(filePath_dbf)
        #     os.remove(filePath_prj)
        #     os.remove(filePath_shx)


    print("\n VALIDATION IS TERMINATED NORMALLY!!! \n")
    return