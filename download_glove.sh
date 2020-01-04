wget http://nlp.stanford.edu/data/glove.840B.300d.zip

unzip glove.840B.300d.zip

mkdir glove

mv glove.6B.100d.txt glove/
mv glove.6B.200d.txt glove/
mv glove.6B.300d.txt glove/
mv glove.6B.50d.txt glove/

rm glove.840B.300d.zip

