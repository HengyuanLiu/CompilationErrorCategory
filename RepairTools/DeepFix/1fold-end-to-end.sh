export PYTHONPATH=.
bash data_processing/data_generator.sh
bash neural_net/1fold-train.sh
bash post_processing/1fold-result_generator.sh
