# Flood_Hazard
Flood Hazard maps using Maching learning technique

all you need to do is run the main.py

in main.py
Choosing the "train" option, retrains the models, choosing the one with the best accuracy. NN arent integraded with the process, must run it individualy. 
Choosing the "evaluation" option, creates the hazard maps per date and per place. If no model is trained, training starts first.

keras_train.py
run this python script to train the NN model.

Create_Flood_Hazard_Maps.py
Th create_waterbodies is the water mask and water depth extraction from the shapefiles

*** SOS ***
in order to run this project you need to download and place the SatImAn_Data fold in current working directory (same as main.py)