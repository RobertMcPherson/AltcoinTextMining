#!/bin/bash
#uncomment line below if compiling a new model for the first time
#javac -cp h2o-genmodel.jar -J-Xmx2g DRF_model_R_1470416938086_15.java 
#java -cp .:h2o-genmodel.jar hex.genmodel.tools.PredictCsv --header --model DRF_model_R_1470416938086_15 --input input.csv --output output.csv
#javac -cp h2o-genmodel.jar -J-Xmx2g NewPredictCsv.java GBM_grid_0_AutoML_20180127_161842_model_33.java
#java -cp .:h2o-genmodel.jar NewPredictCsv --header --model GBM_grid_0_AutoML_20180127_161842_model_33 --input labels_and_vectors.csv --output output.csv
#java -cp .:ICO_GBM_model_33.jar:h2o-genmodel.jar NewPredictCsv --header --model GBM_grid_0_AutoML_20180127_161842_model_33 --input labels_and_vectors.csv --output output.csv
java -cp .:DeepLearning_grid_0_AutoML_20180225_012018_model_9.jar:h2o-genmodel.jar NewPredictCSV --header --model DeepLearning_grid_0_AutoML_20180225_012018_model_9 --input labels_and_vectors.csv --output output.csv







