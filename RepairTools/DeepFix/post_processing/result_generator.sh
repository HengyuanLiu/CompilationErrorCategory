# Combined
for binid in {0..4}
do
python -O post_processing/generate_fixes.py data/checkpoints/iitk-typo-1189/bin_"$binid"/ -d data/results/iitk-raw.db
python -O post_processing/generate_fixes.py data/checkpoints/iitk-ids-1189/bin_"$binid"/ -d data/results/iitk-raw.db -t ids

python -O post_processing/generate_fixes.py data/checkpoints/iitk-typo-1189/bin_"$binid"/ -d data/results/iitk-seeded-typo.db -w seeded
python -O post_processing/generate_fixes.py data/checkpoints/iitk-ids-1189/bin_"$binid"/ -d data/results/iitk-seeded-ids.db -w seeded -t ids
done
python -O post_processing/apply_fixes.py data/results/iitk-raw.db
python -O post_processing/apply_fixes.py data/results/iitk-seeded-typo.db
python -O post_processing/apply_fixes.py data/results/iitk-seeded-ids.db

# Separate typo and id
for binid in {0..4}
do
python -O post_processing/generate_fixes.py data/checkpoints/iitk-typo-1189/bin_"$binid"/ -d data/results/iitk-raw-typo.db
python -O post_processing/generate_fixes.py data/checkpoints/iitk-ids-1189/bin_"$binid"/ -d data/results/iitk-raw-ids.db -t ids
done
python -O post_processing/apply_fixes.py data/results/iitk-raw-typo.db
python -O post_processing/apply_fixes.py data/results/iitk-raw-ids.db