for energy_margin in -1000 -100 -50 -25 -10 -5
do
    CUDA_VISIBLE_DEVICES=1,2,3 python lightning_dict.py 1 1 'energy' -s 999999 -d 'energy' -m 'eval' \
    --eval_output './results_energy_estimation.csv' \
    --energy_margin_in $energy_margin \
    --save_folder 'energy-margin='$energy_margin
done