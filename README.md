There are many python files here which are from varies different test performed throughout. 

The main bulk of analysis is done within the dataAnalysis folder with specific calculations performed within the handlers folder.

Most of the plots on the final report are produced using the plotFiles folder

Many other files are not certain to work as changes may have been made at different stages.

To perform analysis the following lines are used:

from dataAnalysis import configLoader,initDataFiles
config = configLoader.loadConfig()
dataFiles = initDataFiles(config)

This will give a list of dataAnalysis class which can have things called like get_clusters()