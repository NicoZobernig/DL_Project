echo "Downloading apy dataset (1.5 GB)"

wget http://vision.cs.uiuc.edu/attributes/attribute_data.tar.gz
wget http://vision.cs.uiuc.edu/attributes/ayahoo_test_images.tar.gz
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar

tar -xf attribute_data.tar.gz
tar -xf ayahoo_test_images.tar.gz
tar -xf VOCtrainval_14-Jul-2008.tar

mkdir apy

mv VOCdevkit apy/
mv attribute_data apy/
mv ayahoo_test_images apy/

rm attribute_data.tar.gz
rm ayahoo_test_images.tar.gz
rm VOCtrainval_14-Jul-2008.tar