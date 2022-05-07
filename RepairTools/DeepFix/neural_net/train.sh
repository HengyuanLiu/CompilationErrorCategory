for binid in {0..4}
do
python -O neural_net/train.py data/network_inputs/iitk-typo-1189/bin_"$binid"/ data/checkpoints/iitk-typo-1189/bin_"$binid"/
python -O neural_net/train.py data/network_inputs/iitk-ids-1189/bin_"$binid"/ data/checkpoints/iitk-ids-1189/bin_"$binid"/
done
