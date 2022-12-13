echo $BASHPID
for energy_margin in 1000 100 50 25 10 5
do
    python -u lightning_dict.py 'split1' 3 'energy' -s 999999 --eval_output './results_energy_estimation.csv' --energy_margin_in $energy_margin --save_folder 'energy_estimation/energy-margin='$energy_margin
done