sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h
sleep 5

for i in $(seq 16 16 128)
do
    python milvus_test_ivf.py "$1" "$2" $i
done