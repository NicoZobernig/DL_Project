Data and code for "Ensuring Surjectivity in Zero-Shot Learning" - Deep Learning project from Rishabh Singh, Yannick Strümpler, Batuhan Yardim, Nicolas Zobernig
----------------------------------------------------------------------------------------------------------------------------------------------------------------
We provide the Data used here: https://polybox.ethz.ch/index.php/s/wEcRlsPciDGwDJs
--> 4 Datasets (APY, AWA2, CUB, SUN)
	--> /train_set				...	all training data
	--> /test_set				...	all test data
	--> /train_set_i			...	training data for split i
	--> /val_set_i				... 	validation data fro split i

--> All folders have the same file structure which is then used in the ZSLDataset python class (see below):
	--> class_predicates.txt		...	file containing the attributes for each class
	--> classes.txt				...	file containing all class labels with respective class id
	--> filenames_labels.txt		...	file containing filenames and class id for each sample
	--> glove_embeddings_300.txt		...	file containing GloVe word embeddings for each class label
	--> irevnet_image_embeddings.txt	...	file containing image embeddings for each sample generated with iRevNet
	--> resnet101_image_embeddings.txt	...	file containing image embeddings for each sample generated with ResNet101


We provide the python files used for training and testing our model:

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


