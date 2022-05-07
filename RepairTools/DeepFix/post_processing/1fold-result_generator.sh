binid=0
python -O post_processing/generate_fixes.py data/checkpoints/iitk-typo-1189/bin_"$binid"/ -d data/results/iitk-raw-bin_"$binid".db
python -O post_processing/generate_fixes.py data/checkpoints/iitk-ids-1189/bin_"$binid"/ -d data/results/iitk-raw-bin_"$binid".db -t ids
python -O post_processing/apply_fixes.py data/results/iitk-raw-bin_"$binid".db
