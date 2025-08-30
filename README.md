# UofG_Master_Project
This project is the source code of my master's project "Improving image segmentation with a interactive active learning method"

This project implements various image segmentation training and active learning methods, supporting the ISIC 2017 and Oxford IIIT PET datasets. If the project file name contains "ISIC," it indicates that the model was  using the ISIC 2017 dataset; if not, it indicates that the model was tested using the Oxford IIIT PET dataset. `main.py` and `main_ISIC.py` are the entry points for the full-pixel active learning model; running these files allows you to perform full-pixel active learning segmentation. `supervised_main.py` and `supervised_main_ISIC.py` are used for supervised learning with full-pixel annotation; running these files allows you to perform fully supervised segmentation. `scribble_main.py` and `scribble_main_ISIC.py` are the entry points for the Scribble-based active learning model; running these files allows you to perform Scribble-based active learning segmentation. 
`active_learning.py` and `active_learning_ISIC.py` are active learning loops; `active_learning_scribble.py` and `active_learning_scribble_ISIC.py` are Scribble-based active learning loops. Core modules include `model_ISIC.py` (model definition), `train_ISIC.py` (training), `predict_ISIC.py` (prediction), `eval_ISIC.py` and `eval.py` (evaluation functions), `data_loader_ISIC.py` and `data_loader.py` (data loaders), and `utils_ISIC.py`, `utils.py`, and `scribble_utils.py` (helper functions and Scribble tools). 
Before running, organize the ISIC 2017 dataset into:
data/
Ground_Truth/
Training_Data/
and the Oxford IIIT PET dataset into:
data/
annotations/
images/

You can then run the corresponding tasks using different entry files.
