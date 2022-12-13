for split in civ hard1 hard2
do
    for loss in vanilla oe energy
    do
        python -u lightning_dict.py $split 5 $loss --dataset 'ships' -s 999999 --eval_output './results_ships.csv'
    done
done