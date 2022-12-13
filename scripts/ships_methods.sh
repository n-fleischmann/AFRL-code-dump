for method in ternary quad;
do
    python -u lightning_dict.py 'military' 5 $method --dataset 'ships' -s 999999 --eval_output './results_ships.csv'
done