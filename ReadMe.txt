Data and code for "Ensuring Surjectivity in Zero-Shot Learning" - Deep Learning project from Rishabh Singh, Yannick Strümpler, Batuhan Yardim, Nicolas Zobernig
----------------------------------------------------------------------------------------------------------------------------------------------------------------
We provide the Data used here: https://polybox.ethz.ch/index.php/s/wEcRlsPciDGwDJs
--> 4 Datasets (APY, AWA2, CUB, SUN)
	--> /train_set				...	all training data
	--> /test_set				...	all test data
	--> /train_set_i			...	training data for split i
	--> /val_set_i				... 	validation data fro split i

	--> All dataset folders have the same file structure which is then used in the ZSLDataset python class (see below):
		--> class_predicates.txt		...	file containing the attributes for each class
		--> classes.txt				...	file containing all class labels with respective class id
		--> filenames_labels.txt		...	file containing filenames and class id for each sample
		--> glove_embeddings_300.txt		...	file containing GloVe word embeddings for each class label
		--> irevnet_image_embeddings.txt	...	file containing image embeddings for each sample generated with iRevNet
		--> resnet101_image_embeddings.txt	...	file containing image embeddings for each sample generated with ResNet101

	--> we provide the data for generalized zeros hot for APY in gzsl_train and gzsl_test


--> SAE Folder:
	--> data_preparation.m : This matlab file takes the following inputs:
		1. att_splits.mat: This contains the attribute vetors per class and file indices of train, val and test sets used.
		2. res101.mat: This file contains the ResNet features for each image, filenames of each image and file named 'labels' containing the class lables per sample image.
		3. suncemb.txt/ cubcemb.txt/ awacemb.txt/ apycemb.txt: Glove embeddings per class for each dataset.
	All these files are in the respective dataset folders eg. data/SUN/.. The above three files have to be provided as input at the top of the code.
	The code saves a .mat file in the folder "code" which is exactly the format needed to run the next step.
	2. sae.py: The next and last step is to run this python file. The previous step has already stored the input in the current folder (SAE_eval\code\). The name of this .mat file is to be added in the line
		"awa = scipy.io.loadmat('sun.mat')"
		in the main() function.
		Run this file to get either ZSL or GZSL (depending on your input) for both cases of F->S and S->F mappings, where F is the visual space and S is the semantic space.

We provide the python files used for training and testing our model:
--------------------------------------------------------------------

zsl_triplet_crossval.py
=======================
 - file used for training the model and finding the best hyperparameters using the splits on training data
	--> using Triplet Loss

 ==> python zsl_triplet_crossval.py <DATA_PATH> --help

	--> example of DATA_PATH: path/to/Data/AwA2/


zsl_triplet_test.py
===================
 - file used for evaluating the model on test data after fitting on the entire training data
	--> using Triplet Loss

 ==> python zsl_triplet_crossval.py <DATA_PATH> --help

	--> example of DATA_PATH: path/to/Data/AwA2/


zsl_L2_crossval.py
==================
 - file used for training the model and finding the best hyperparameters using the splits on training data
	--> using L2 Loss

 ==> python zsl_triplet_crossval.py <DATA_PATH> --help

	--> example of DATA_PATH: path/to/Data/AwA2/


zsl_L2_crossval.py
==================
 - file used for evaluating the model on test data after fitting on the entire training data
	--> using L2 Loss

 ==> python zsl_triplet_crossval.py <DATA_PATH> --help

	--> example of DATA_PATH: path/to/Data/AwA2/


zsl_linear_test.py
==================
 - file used to evaluate the linear baseline on test data after fitting on the entire training data

==> python zsl_linear_test.py <DATA_PATH> --help

	--> example of DATA_PATH: path/to/Data/AwA2/

models.py
=========
 - file containing the pytorch description of the used networks
	--> ContinuousMap 		...	simple fully-connected neural network (parametrized continuous function)
	--> EncoderAttributes		...	model for jointly encoding visual attributes and word embeddings
	--> DecoderAttributes		...	model for jointly decoding visual attributes and word embeddings
	--> LinearDecoderAttributes	... 	linear network for jointly decoding visual attributes and word embeddings


zsldataset.py
=============
 - file containing the ZSLDataset class, extending the pytorch.utils.data.Dataset class
 	ZSLDataset object containing the data read in from the data folders



We provide the optimal parameter values for the model for each dataset:
-----------------------------------------------------------------------
APY: 	

AWA2:
    Triplet: Words only - attributes only - attributes + words:
    python zsl_triplet_test.py Data/AwA2/ --batch_size 128 --n_epochs 50 --optimizer sgd --learning_rate 5e-3 --alphas 40 1e-3 1e-3 --margin 3 --gamma 0.3 --momentum 0.55 --weight_decay 3e-3

    L2: attributes + words:
    python zsl_L2_test.py Data/AwA2/ --batch_size 128 --n_epochs 50 --optimizer sgd --learning_rate 5e-3 --alphas 40 1e-3 1e-3  --gamma 0.3 --momentum 0.55 --weight_decay 3e-3

    L2 baseline: words only - attributes only - attributes + words:
    python zsl_linear_test.py Data/AwA2/ --batch_size 128 --n_epochs 50 --optimizer sgd --learning_rate 5e-3 --alphas 40 1e-3 1e-3  --gamma 0.3 --momentum 0.55 --weight_decay 3e-3

    Triplet Visual: attributes + words:
    python zsl_triplet_visual_test.py  Data/CUB/ --n_epochs 50 --alphas 100 6e-4 1e-3 --learning_rate 5e-2 --margin 0 --gamma 0.5 --momentum 0.9  --weight_decay 1e-2

CUB:
    Triplet: Words only - attributes only - attributes + words:
    python zsl_triplet_test.py Data/CUB/ --batch_size 128 --n_epochs 50 --optimizer sgd --alphas 100 6e-4 1e-3 --learning_rate 1e-4 --margin 50 --gamma 0.8 --momentum 0.9 -weight_decay 1e-2

    L2: attributes + words:
    python zsl_L2_test.py Data/CUB/ --batch_size 128 --n_epochs 50 --optimizer sgd --alphas 1000 1e-4 1e-5 --learning_rate 1e-2  --gamma 0.8 --momentum 0.9 -weight_decay 3e-2

    L2 baseline: words only - attributes only - attributes + words:
    python zsl_linear_test.py  Data/CUB/  --batch_size 128 --n_epochs 50 --optimizer sgd --alphas 5000 0 1e-4 --learning_rate 1e-2   --momentum 0.9 --weight_decay 1e-3

    Triplet Visual: attributes + words:
    python zsl_triplet_visual_test.py  Data/CUB/ --n_epochs 50 --alphas 100 6e-4 1e-3 --learning_rate 5e-2 --margin 0 --gamma 0.8 --momentum 0.9  --weight_decay 1e-2


SUN:



UTILS
-----

generate_image_embeddings.py
============================
- script that computes iRevnet embeddings, the following values need to be adapted to the respective paths

    dataset_path='Data/CUB/'
    has_bounding_box = True
    image_path ='Data/CUB_200_2011/CUB_200_2011/images/'

- the script expects a file 'filenames_labels.txt' that contains the labels and filenames in the format used for the zsldataset


download_irevnet_pretrained.py
==============================
- downloads a pretrained iRevNet Model


split_data.py
=============
- Function to split a dataset in our zsldataset format into train, test or validation sets:
- It expects the following parameters:
    dataset=<DATASET PATH>: path of the zsldataset
    class_names_file=<filename>: filename that contains list of classnames (one classname per line)
     new_dataset_folder_name=<NEW DATASET PATH>: folder where the new dataset will be created
- the new dataset consists of a subset of classes that are matched with the given list
- the function splits both files for iRevNet and ResNet embeddings and the filenames_labels.txt file


