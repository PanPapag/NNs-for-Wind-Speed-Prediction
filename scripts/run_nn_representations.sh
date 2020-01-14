cd ../app
make
cd build/
./cluster -i ../../datasets/nn_representations.csv -c ../../config/cluster.conf -o ../../results/nn_representations --complete --init random --assign lloyd --update mean
cd ..
make clean
