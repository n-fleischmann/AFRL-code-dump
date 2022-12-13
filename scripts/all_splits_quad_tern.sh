
for split in 2 3 4
do
    for loss in ternary quad
    do
        python -u lightning.py $split 5 $loss -m 'traineval' -s 999999
    done
done