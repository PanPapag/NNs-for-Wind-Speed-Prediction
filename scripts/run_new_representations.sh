cd ../app
make
cd build/
./cluster -i ../../datasets/new_representations.csv -c ../../config/cluster.conf -o ../../results/new_representations_12_111.csv --complete --init random --assign lloyd --update pam
cd ..
make clean
