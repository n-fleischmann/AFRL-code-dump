#!/usr/bin/zsh

echo 'MASTER PID: '$$
for split in 'split1' 'split2' 'split3' 'split4'
do
    python -u lightning_dict.py $split 5 'energy' -s 999999 --eval_output './results_energy_estimation.csv' --energy_margin_in 25
done