cd ../app
make
cd build/
./cluster -i ../../datasets/new_representations.csv -c ../../config/cluster.conf -o ../../results/new_representations --complete --init k-means++ --assign lloyd --update pam
cd ..
make clean
