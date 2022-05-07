# CompilationErrorCategory

This is the code repository for an empirical study of compilation error repair tools. This work is extended from our previous study(compilation error classfier):

* Li Z, Sun F, Wang H, et al. CLACER: A Deep Learning-based Compilation Error Classification Method for Novice Students’ Programs[C]//2021 IEEE 45th Annual Computers, Software, and Applications Conference (COMPSAC). IEEE, 2021: 74-83.

In this work we futher analysis three compilation error repair tools: DeepFix, RLAssist and MACER, which are all target at the code written by novice programmer in C program language. By compilation error classification, we analysis the advantages and shortcomes of these tools and summary some improvement ways.

## Directory Content

### Dataset

In this study, we construct the student code repositories from two publicly available datasets (i.e., The DeepFix dataset(http://iisc-seal.net/deepfix) and The TEGCER dataset(https://github.com/umairzahmed/tegcer)). These datasets are all curated from Introductory to C Programming (CS1) of college students. These assignments were recorded using a custom web-browser based IDE Prutor. 

Another dataset BUCTOJ is conclude which is collected by our team from the OJ system of Beijing University of Chemical Technology. And the dataset is also avaliable in the Dataset directory.


The DeepFix datasets contains 6,971 programs spanning 93 assignments that fail to compile, each lengths range from 40 to 100 token. 

The TEGCER dataset contains 23,275 buggy programs with corresponding repairs. TEGCER dataset spans 40+ different programming assignments completed by 400+ first-year undergraduate students.

The BUCTOJ dataset contains 12,648 buggy programs. These programs are the assignments submitted by the students. We selected 12,648 programs with compilation errors out of 45,000+ error programs from BUCTOJ.

We  only use programs with single-line buggy statement.Finally, we select 2,911 programs from 6,971 programs in the DeepFix dataset and 14,015 programs from 23,275 programs in the TEGCER dataset and 12648 programs in the BUCTOJ dataset.

### CLACER

CLACER is a compilation error classifier proposed in our previous research. And the details are can be found in our paper and the README.md in the CLACER directory.

### RepairTools

In this directory, we provide the source code of the three repair tools with slight modification which are research in this work.
