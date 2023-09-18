# qLFAID

This repository contains the supplemental code and data for the article "Development of a Universal Lateral-Flow Immunoassay Interpretation Device Prototype with Quantitative Output for Blind or Visually Impaired Individuals" by Guillermo *et al.*, IEEE Sensors Journal (https://doi.org/10.1109/JSEN.2023.3306284).

Please look at the **demo.ipynb** file for instructions on how to train a regression model for your dataset of interest.

The **PaddleSeg** directory is a branch of the official PaddleSeg repository and contains the neural network code necessary to run the qLFAID. Some files have been modified to include additional functionalities.

The **PaddleSeg/data** directory contains the training/testing data for the neural network. `JPEGImages` are taken from Mendels *et al.* (https://doi.org/10.1073/pnas.2019893118). `annotations` were done manually, using the LabelMe annotation tool (https://github.com/wkentaro/labelme), and are stored in png format. These folders can also be downloaded in a compressed format using the following link: https://mega.nz/folder/QO53WJoC#AM_lsKFEAWH6r0BC99G71A.

The **PaddleSeg** directory also contains the code necessary to train and run the image segmentation neural network employed in the paper. The configuration file for training is stored in `PaddleSeg/configs/quick_start/pp_liteseg_LFA_512x512_1k.yml`. The inference file `PaddleSeg/deploy/python/infer.py` was also modified to include necessary accessory functions for the qLFAID. Finally, the `run.py` file was created to run the actual qLFAID device, when the neural network is run on a laptop. This can be done without WiFi, via an ethernet connection to the Raspberry Pi. However, ideally, the network would be exported and run on the Raspberry Pi itself. It can be run from the terminal using the command: `python run.py`

Finally, the **RaspberryPiFiles** directory contains python files that were coded in the Raspberry Pi to run the qLFAID device. The `scanOutput.py` file was made to to prompt the user to insert the LFA into the device and take pictures of it using the camera module and stepper motor. The `output.py` was made to output the result of the neural network analysis to the user, using the speaker and vibration module.