# qLFAID

This repository is meant to contain the supplemental code and data for the article "Development of a universal lateral-flow immunoassay reading device with quantitative output for the blind and visually impaired" by Guillermo et al, IEEE Sensors (2023).

Please look at the demo.ipynb file for instructions on how to train a regression model for your dataset of interest.

The PaddleSeg directory is a branch of the official PaddleSeg repository and contains the neural network code necessary to run the qLFAID. Some files have been modified to include additional functionalities.

The PaddleSeg/data directory contains the training/testing data for the neural network. JPEGImages are taken from Mendels et al (https://doi.org/10.1073/pnas.2019893118). Annotations were done manually,using the LabelMe annotation tool (https://github.com/wkentaro/labelme), and are stored in png format.