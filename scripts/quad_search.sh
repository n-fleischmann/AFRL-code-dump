echo $BASHPID
for quad_delta in 0.1 1 5 10 100;
do
    python -u lightning_dict.py 'military' 3 'quad' --quad_delta $quad_delta --ternary_gamma 0.1 --dataset 'ships' -s 999999 --eval_output './results_ships_quad_search.csv'
done