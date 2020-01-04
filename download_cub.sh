wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
wget http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz
wget http://www.vision.caltech.edu/visipedia-data/CUB-200/attributes.tgz
wget http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz

tar -xf annotations.tgz
tar -xf images.tgz
tar -xf lists.tgz
tar -xf attributes.tgz

rm images.tgz
rm lists.tgz
rm attributes.tgz
rm annotations-mat

mkdir cub

mv annotations-mat cub/
mv attributes cub/
mv lists cub/
mv images cub/
