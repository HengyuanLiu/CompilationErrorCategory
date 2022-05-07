# Updates (December 13, 2017)

1. We are glad to announce that the [DeepFix dataset](https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip) has been released under Apache 2.0, courtesy Prof. Amey Karkare and his research group.
It was collected from an introductory programming course at Indian Institute of Technology, Kanpur, India using a programming tutoring system called [Prutor](https://www.cse.iitk.ac.in/users/karkare/prutor/).
If you use this dataset for your research, kindly give due [credits](https://www.cse.iitk.ac.in/users/karkare/prutor/deepfix-bib.html) to both Prutor and DeepFix. 

2. We have updated the training data generation process to remove any bias towards particular error seeding rules. This has resulted in improved performance for DeepFix. We have also upgraded the code to use dynamic RNN implementation of tensorflow and made it easier to set up the environment and to reproduce the results.

3. As discussed in the paper, we use two separate neural networks for fixing the typographic and the missing variable declaration errors.
For the raw test dataset, along with the combined results produced using both the networks, we now report the results for both these networks separately as well.
We also generate two separate seeded datasets, one for each network and report results on them separately only.


| Dataset              | Model         | Erroneous programs | Avg. tokens | Error Msgs. | Completely fixed programs | Partially fixed programs | Msgs. resolved |
|----------------------|---------------|--------------------|-------------|-------------|---------------------------|--------------------------|----------------|
| Raw                  | Typographic   | 6975               | 203         | 16766       | 1625 (23.3%)              | 1129 (16.2%)             | 5156 (30.8%)   |
|                      | Missing Decl. |                    |             |             | 707 (10.1%)               | 851 (12.2%)              | 2164 (12.9%)   |
|                      | Combined      |                    |             |             | 2327 (33.4%)              | 1557 (22.3%)             | 6836 (40.8%)   |
| Seeded-typographic   | Typographic   | 9242               | 206         | 43478       | 6411 (69.4%)              | 1417 (15.3%)             | 34780 (80%)    |
| Seeded-missing Decl. | Missing Decl. | 9241               | 197         | 25754       | 5863 (63.4%)              | 2579 (27.9%)             | 20229 (78.5%)  |

As it is clear from the above table, the number of both resolved error messages as well as fixed programs have increased significantly from what was reported in the paper.
Note that the number of total erroneous programs for the raw and the seeded datasets are slightly more than what was reported in the paper.
This minor difference arises because we now work with programs with length ranging from 75 to 450 tokens, instead of 100 to 400 tokens.

If you compare your own tool with DeepFix in future, we request you to use the above results and cite this repository along with the original paper.

@misc{deepfix2017repository,  
	author = {Gupta, Rahul and Pal, Soham and Kanade, Aditya and Shevade, Shirish},  
	title = "DeepFix: Fixing Common C Language Errors by Deep Learning",  
	year = "2017",  
	url = "http://www.iisc-seal.net/deepfix",  
	note = "[Online; accessed DD-MM-YYYY]"  
}

# License

DeepFix is available under the Apache License, version 2.0. Please see the LICENSE file for details.

# Reference

Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade. "DeepFix: Fixing common C language errors by deep learning", AAAI 2017.

# Running the tool

If you are using Ubuntu 16.04.3 LTS (not tested on other distributions) and have conda installed, you can simply source `init.sh` which creates a new virtual environment called `deepfix` and installs all the dependencies in it.
Furthermore, it downloads, extracts, and preprocesses the student submission data for you into the required directory structure.

To reproduce the results, first you need to generate training, validation, and testing data from the preprocessed dataset.
Next the DeepFix model has to be trained for both typographic and missing variable declaration errors for all 5 folds.
Finally by generating and applying fixes using these 10 trained models, the results have to be generated for both `raw` and `seeded` datasets.
To make it simpler for you, we have provided a script called `end-to-end.sh`, which performs the above three steps and reproduces the results.
As it takes a significant amount of time, you can also run `1fold-end-to-end.sh` to run DeepFix for just one fold.

# Checking the results

The generated results are stored in text files in the `data/results` directory.
If you want to check the programs after fix, check the `.db` files which are sqlite3 databses storing fixes from each iteration and the finally fixed programs.

# TL;DR

    $ source init.sh
    $ bash end-to-end.sh
