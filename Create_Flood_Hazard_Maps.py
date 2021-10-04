# Methods to Create Flood Hazard Maps
#
from osgeo import gdal
import numpy as np
import pandas as pd
from auxiliary_functions import *
import rasterio
from rasterio.features import rasterize
import time
import os
import geopandas as gpd

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

# Function to create Flood Hazard Maps based on the trained Machine Learning model (classifier)
# df_img: annotated dataset extracted from composition of various features from SIA, GIS
# attrbs, targs: lists with attributes names
#
def create_flood_hazard_maps( classifier, minmax_scaler, df_img, attrbs, targs, df_img_WH, Conf_Mat_name,Pixel_Resolution):

    print(" >>>>>>>>>>>>>>>>>>>>>>================== \n")
    print("\n INSIDE create_flood_hazard_maps ")


    # pre-precessing outliers. Get their idx
    print(" Shape of image = ", df_img.shape)
    print("idx")
    idx_dem = df_img[df_img['DEM'] == float('-inf')].index.tolist()
    idx_slope = df_img[(df_img['Slope'] == float('-inf')) | (df_img['Slope'] == float(-10000.0))].index.tolist()
    idx_water_depth = df_img[(df_img['Water_Depth'] == float(0)) & (df_img['p1'] == float(0))].index.tolist()
    idx_asp = df_img[(df_img['Aspect'] == float(-10000.0)) | (df_img['Aspect'] == float("nan"))].index.tolist()
    idx_tpi = df_img[(df_img['TPI'] == float(-10000.0)) | (df_img['TPI'] == float("-inf")) | (df_img['TPI'] == float("inf"))].index.tolist()

    # sort their unique ids and get the clean dataframe
    idx_union = sorted(list(set().union(idx_dem, idx_slope, idx_asp, idx_tpi,idx_water_depth)))
    df_img_clean = df_img[~df_img.index.isin(idx_union)]
    print("To clean: ",df_img_clean.shape)
    nan = df_img_clean.isnull().values.any()

    # check if NaN
    print("NAN? ", nan)
    if nan:
        df_img_clean.isnull().sum().sum()
        # print(df_img_clean)

    # keep the outliers, to reindex later, so we can create the map
    df_img_no_clean = df_img[df_img.index.isin(idx_union)]
    del df_img

    # Calculate the Water Velocity per pixel based on Water_Depth, Slope, Resolution and Roughness
    wv_values = calc_water_velocity(df_img_clean,Pixel_Resolution)
    df_img_clean['Water_Velocity'] = wv_values
    # Calculate the Flood Hazard
    df_img_clean = calc_flood_hazard(df_img_clean)

    idx_no_hazard = df_img_no_clean.index.tolist()
    print("Length No Hazard indices = ", len(idx_no_hazard))

    # Split df_img_to_predict to dataframe contains columns for the Patterns and Targets
    df_pat = pd.DataFrame(df_img_clean.loc[:, attrbs])
    df_targ = pd.DataFrame(df_img_clean.loc[:, targs])

    print(" df_pat = ", df_pat.shape)
    print(" df_targ = ", df_targ.shape)

    targ_categ = pd.unique(df_targ[targs])
    print("\n Target categories in df_img_to_predict targets (df_targ): ", targ_categ)

    # Predict the Target value of each pixel in dataframe for an image (df_img) based on the attributes values
    #
    # Apply min_max_scaler to the patterns and give columns names and indexes as df_pat
    df_pat_scaled = pd.DataFrame( minmax_scaler.transform( df_pat ) )
    df_pat_scaled.columns = df_pat.columns
    df_pat_scaled.index = df_pat.index

    # classifier.fit( df_pat, df_targ.values.ravel())
    pred_targs = pd.DataFrame( classifier.predict( df_pat_scaled ) )
    pred_targs.columns = [targs]
    pred_targs.index = df_pat.index
    print("end of prediction")

    # results to check >>>>>
    print(" shape pred_targs = ",  pred_targs.shape)
    pred_targs_categories = pd.unique(pred_targs[targs])
    print("predicted categories: ", pred_targs_categories)

    # Evaluate the classifier over whole image. Per pixel evaluation...
    if len(targ_categ) < len(pred_targs_categories):
        measures = classification_report(df_targ, pred_targs, target_names=pred_targs_categories, digits=2, output_dict=False)
        cm = confusion_matrix(df_targ, pred_targs, labels=pred_targs_categories)
        conf_mat_img = pd.DataFrame( cm, index=pred_targs_categories, columns=pred_targs_categories )

        # Display Confusion Matrix and store it. Save the Evaluation Results in file
        cm_display = ConfusionMatrixDisplay( cm, display_labels= pred_targs_categories).plot(include_values=True, cmap='Blues', ax=None, values_format='d', xticks_rotation='horizontal', colorbar=False)
        print(" \n Location of the Confusion Matrix is: ", Conf_Mat_name)
        plt.savefig( Conf_Mat_name )
        plt.close()

    elif len(targ_categ) >= len(pred_targs_categories):
        measures = classification_report(df_targ, pred_targs, target_names=targ_categ, digits=2, output_dict=False)
        cm = confusion_matrix(df_targ, pred_targs, labels= targ_categ)
        conf_mat_img = pd.DataFrame( cm, index=targ_categ, columns=targ_categ )

        # Display Confusion Matrix and store it. Save the Evaluation Results in file
        cm_display = ConfusionMatrixDisplay( cm, display_labels= targ_categ).plot(include_values=True, cmap='Blues', ax=None, values_format='d', xticks_rotation='horizontal', colorbar=False)
        print(" \n Location of the Confusion Matrix is: ", Conf_Mat_name)
        plt.savefig( Conf_Mat_name )
        plt.close()

    print(measures)
    print(conf_mat_img)
    print(" <<<<<<<<<================== \n")

    res = {'Confusion_Matrix': conf_mat_img, "Evaluation_Measures_Results": measures}

    #-------------------------------------------------------------------------
    # Start reconstructing the Map
    #
    # Create a new array with the whole pixels
    # new_labeled_NoHaz = pd.DataFrame( ['No Hazard']*len(idx_no_hazard), index=idx_no_hazard, columns=[targs] )
    new_labeled_Haz = pd.DataFrame(pred_targs, index=df_pat.index.tolist(), columns=[targs])

    # drop the annotated columns and join the predicted ones
    df_img_clean.drop(["Hazard_Category", "Hazard"],axis='columns', inplace=True)
    df_img_clean = df_img_clean.join(new_labeled_Haz)

    # calculate the hazard value through label  example: 0.8-->High Haz
    print("calculate hazard")
    df_img_clean = calc_flood_hazard(df_img_clean)

    # concat images
    print("drop columns, and final concat")
    new_df = pd.concat([df_img_clean, df_img_no_clean], axis=0)
    del df_img_clean, df_img_no_clean
    print("sort index")
    new_df = new_df.sort_index()
    new_df = new_df[['Hazard_Category']]

    print("reshaping")

    # Reshape and create 3D hazard map
    # In reshape() the first parameter indicates the rows (height) and the second the columns (width)
    # prepare_3d_array_from_2d_color(the_2d_hazzard_array, df_img_WH) for colored tif
    the_2d_hazard_array_haz = new_df["Hazard_Category"].values.reshape( df_img_WH['Height'], df_img_WH['Width'] )
    colored_hazard_3D_haz = prepare_3d_array_from_2d_color(the_2d_hazard_array_haz, df_img_WH)

    print("end of create_flood_hazard_maps")
    return res, colored_hazard_3D_haz

#====================================================================================================
def prepare_3d_array_from_2d_color(the_2d_hazzard_array, df_img_WH, check = "Hazard"):

    # Give color to water in a new RGB array
    width = df_img_WH['Width']
    height = df_img_WH['Height']

    the_3d_painted_array = np.zeros((height, width, 3))

    # Colors for the Hazard Map in RGB
    if check == "Hazard":
        # Color in OpenCV is handled in BGR arrays, not RGB (thus for the provided color change order of the r,g,b, to b,g,r)#####
        # Give a different color (in RGB) for each hazard level
        # BRG colors
        the_3d_painted_array[ the_2d_hazzard_array == 'No Hazard'] = (193, 205, 205)  # Ivory 3 for 'No Hazard'
        the_3d_painted_array[the_2d_hazzard_array == 'Low Hazard'] = (0, 255, 170)  # Green 2 for 'Low Hazard'
        the_3d_painted_array[the_2d_hazzard_array == 'Medium Hazard'] = (0, 255, 255)  # Orange for 'Medium Hazard'
        the_3d_painted_array[the_2d_hazzard_array == 'High Hazard'] = (0, 170, 255)  # Red for 'High Hazard'

        the_3d_painted_array = the_3d_painted_array[:, :, :].astype(int)

        return the_3d_painted_array


#=================================================================================================================================
# Save flood hazard map as geotiff file
#
# gdal_guidance_image : image opened by gdal and used as guidance in order to extract info for the new GeoTiff file.
# output_file : name of new geotiff file
# format : the format of the output file (default is 'GTiff')
# rasterXSize, rasterYSize : size of the pixel in guidance image
# array_image_rgb : the 3D array with the size of the image and colored info for each pixel in RGB
#
def saveGeoTiffRGB(gdal_guidance_image, output_file, format, rasterXSize, rasterYSize, array_image_rgb):

    tiff_output_file = output_file + '.tiff'

    # test me
    geoTrans_guidance = gdal_guidance_image.GetGeoTransform()  # Retrieve Geo-information of guidance image and save it in geoTrans_guidance
    wkt_guidance = gdal_guidance_image.GetProjection()  # Retrieve projection system of guidance image into well known text (WKT) format and save it in wkt_guidance

    driver = gdal.GetDriverByName(format)  # Generate an object of type Geotiff
    # options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']    , gdal.GDT_Float32, options=options

    # Create a raster of type Geotiff with dimension Guided_Image.RasterXSize x Guided_Image.RasterYSize, with one band and datatype of GDT_Float32
    dst_ds = driver.Create(tiff_output_file, rasterXSize, rasterYSize, 3, gdal.GDT_Byte)
    if dst_ds is None:  # Check if output_file can be saved
        print('Could not save output file %s, path does not exist.' % output_file)
        quit()
    # sys.exit(4)

    # Set the Geo-information of the output file the same as the one of the guidance image
    dst_ds.SetGeoTransform( geoTrans_guidance )
    dst_ds.SetProjection( wkt_guidance )

    dst_ds.GetRasterBand(1).WriteArray(array_image_rgb[:, :, 0])  # write r-band to the raster
    dst_ds.GetRasterBand(2).WriteArray(array_image_rgb[:, :, 1])  # write g-band to the raster
    dst_ds.GetRasterBand(3).WriteArray(array_image_rgb[:, :, 2])  # write b-band to the raster

    dst_ds.FlushCache()  # Write to disk.

def water_bodies(shapefile_m, dem, burned, water_depth, water_mask):
    # saves greyscale tif from 2d array, will be used for watermask creation
    def save_geo_tiff(gdal_guidance_image, output_file, out_format, rasterXSize, rasterYSize, array_image, dtype,
                      noDataValue=""):
        # test me
        geoTrans_guidance = gdal_guidance_image.GetGeoTransform()  # Retrieve Geo-information of guidance image and save it in geoTrans_guidance
        wkt_guidance = gdal_guidance_image.GetProjection()  # Retrieve projection system of guidance image into well known text (WKT) format and save it in wkt_guidance

        # format = 'GTiff'
        driver = gdal.GetDriverByName(out_format)  # Generate an object of type GeoTIFF / KEA
        dst_ds = driver.Create(output_file, rasterXSize, rasterYSize, 1, dtype, options=['COMPRESS=DEFLATE'])
        # Create a raster of type Geotiff / KEA with dimension Guided_Image.RasterXSize x Guided_Image.RasterYSize, with one band and datatype of GDT_Float32
        # LZW or DEFLATE compression
        if dst_ds is None:  # Check if output_file can be saved
            print("Could not save output file %s, path does not exist." % output_file)
            quit()
        # sys.exit(4)

        dst_ds.SetGeoTransform(
            geoTrans_guidance)  # Set the Geo-information of the output file the same as the one of the guidance image
        dst_ds.SetProjection(wkt_guidance)
        # this line sets zero to "NaN"
        if noDataValue != "":
            dst_ds.GetRasterBand(1).SetNoDataValue(noDataValue)
            print("Value to be replaced with NaN was given: " + str(noDataValue))
        dst_ds.GetRasterBand(1).WriteArray(array_image)  # Save the raster into the output file
        dst_ds.FlushCache()  # Write to disk.

    def create_water_depth_map(watermask_shp, raster_dem, burnt_shapes_raster, water_depth_tif):
        rst = rasterio.open(raster_dem)
        meta = rst.meta.copy()
        meta.update(compress='lzw')

        counties = gpd.read_file(watermask_shp)

        # burn each polygon with an increment value in a raster
        with rasterio.open(burnt_shapes_raster, 'w+', **meta) as out:
            out_arr = out.read(1)

            # this is where we create a generator of geom, value pairs to use in rasterizing
            # shapes = ((geom,value) for geom, value in zip(counties.geometry, counties.DN))
            total_shapes = len(counties)
            shapes_ids = list(range(1, total_shapes + 1))
            shapes = ((geom, value) for geom, value in zip(counties.geometry, shapes_ids))

            burned = rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
            out.write_band(1, burned)

        print("just burnt shapes withs ids")

        ds = gdal.Open(burnt_shapes_raster)
        burnt_shapes_array = ds.GetRasterBand(1).ReadAsArray().astype(
            float)  # We did it to float instead of int due to float calculations with dem

        ds = gdal.Open(raster_dem)
        dem_array = ds.GetRasterBand(1).ReadAsArray()

        no_data_dem_val = np.min(np.unique(dem_array))
        no_data_burnt_shapes_val = np.min(np.unique(burnt_shapes_array))

        print(no_data_dem_val)
        print(no_data_burnt_shapes_val)

        active_mask = np.ones((burnt_shapes_array.shape), dtype=bool)
        active_mask[np.logical_or(dem_array == no_data_dem_val, burnt_shapes_array == no_data_burnt_shapes_val)] = False
        exclusion_mask = np.invert(active_mask)

        # reset the canvas at the no data of the DEM (cause we cannot calculate depth for these areas)
        # There seems to be an overlap between the water/shapes and the no-data of the dem.
        burnt_shapes_array[exclusion_mask] = 0  # shape 0 means no shape
        print(np.unique(burnt_shapes_array))

        # create a copy for this to not interfere with the burnt_Shapes_Array while updating the canvas
        canvas_array = np.copy(burnt_shapes_array)

        start_time = time.time()

        # be-carefull, some ids might have been wiped with no-data exlusion!!!
        available_shapes = np.unique(burnt_shapes_array).tolist()
        available_shapes = [x for x in available_shapes if x != 0]

        for shape_id in available_shapes:
            max_water_height = np.max(dem_array[burnt_shapes_array == shape_id])

            canvas_array[burnt_shapes_array == shape_id] = np.max(dem_array[burnt_shapes_array == shape_id]) - \
                                                           dem_array[burnt_shapes_array == shape_id]

        print("--- %s seconds ---" % (time.time() - start_time))

        # Again reset the exluded values to the no-data value to be passed to the geotiff
        canvas_array[exclusion_mask] = no_data_burnt_shapes_val

        save_geo_tiff(ds, water_depth_tif, 'GTiff', ds.RasterXSize, ds.RasterYSize, canvas_array, gdal.GDT_Float32, noDataValue=no_data_burnt_shapes_val)

        # rasterize the water mask tif

    # WATER MASK
    sph = os.path.basename(shapefile_m).rsplit('.',1)[0]
    gdal_guidance_image = gdal.Open(dem)
    # get the shp path with reverse split
    shp_path = shapefile_m.rsplit("/",1)[0]
    rasterXSize = gdal_guidance_image.RasterXSize
    rasterYSize = gdal_guidance_image.RasterYSize
    cmd = 'gdal_rasterize -co "COMPRESS=DEFLATE" -a DN -ts %s %s -l %s %s %s' % (rasterXSize, rasterYSize, sph, os.path.basename(shapefile_m), water_mask)
    # keep the current working directory in a variable
    wd = os.getcwd()
    # change the cwd so we can execute cmd
    os.chdir(shp_path)
    os.system(cmd)
    # change the cwd back in correct one
    os.chdir(wd)
    print("OK WATER MASK")

    # WATER DEPTH
    create_water_depth_map(shapefile_m,dem,burned,water_depth)
    os.remove(burned)
    print("OK WATER DEPTH")

def transform_waterbodies_DDPMS(city_dict):
    # Create Water_Depth/Water_Mask

    # get the dem, and something we use it only for the waterbodies process and it gets delete
    dem = city_dict['path']+ city_dict['place'] + "_DEM.tif"
    burnt_fn = city_dict['path'] + "nothing_to_see_here"

    # get the waterbodies files in a list
    wb_list = os.listdir(city_dict['path']+"waterbodies/")

    date = city_dict['date']

    for j in range(len(wb_list)):
        if date in wb_list[j]:
            waterbody = wb_list[j]
            out_water_depth = city_dict['path'] + os.path.basename(wb_list[j].split("bodies")[0])+"bodies" + "_Water_Depth.tif"
            out_water_mask = city_dict['path'] + os.path.basename(wb_list[j].split("bodies")[0])+"bodies" + "_Water_Mask.tif"


    water_bodies(city_dict['path']+"waterbodies/"+waterbody.split("bodies")[0]+"bodies.shp", dem, burnt_fn, out_water_depth, out_water_mask)
    return