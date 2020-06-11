**SQLWL-segmentation**

Tools for segmenting a SQL workload into coherent explorations.
If you wish to use the code, please cite the first reference below.


**References**

[Veronika Peralta, Patrick Marcel, Willeme Verdeaux, Aboubakar Sidikhy Diakhaby: Detecting coherent explorations in SQL workloads. Information systems, Volume 92, September 2020](https://doi.org/10.1016/j.is.2019.101479)

[Veronika Peralta, Patrick Marcel, Willeme Verdeaux, Aboubakar Sidikhy Diakhaby: Detecting coherent explorations in SQL workloads. http://arxiv.org/abs/1907.05618](http://arxiv.org/abs/1907.05618)


[Verónika Peralta, Willeme Verdeaux, Yann Raimont, Patrick Marcel: Qualitative Analysis of the SQLShareWorkload for Session Segmentation. DOLAP 2019](http://ceur-ws.org/Vol-2324/Paper18-PMarcel.pdf)


**How to use the scripts**

- baseline-desC.py computes baseline for Concatenate dataset, prints out results

- correlationToGroundTruth.py computes correlation of the input dataset based on the columns given, it outputs a csv file and pops up a figure

- balance-dopan-smartbi.py learns the svm classifier for concatenate and transfer it on sqlshare. It outputs 2 cuts file (binary vector ordered by position of queries in the input dataset) per sampler, one for concatenate and one for sqlshare

- bestCombination.py finds the best subset of the labelling functions on concatenate. It outputs a csv file detailing scores (accuracy, precision, recall, f1) for each subset tested

- snorkelOverConcatenante.py applies labelling functions over Concatenate. It outputs a csv with the predicted cut by queries. IMPORTANT: cuts are not ordered correctly there, the csv must be sorted by query and then manually written to concatenate-cuts-snorkel.csv.

- snorkelDetectsCuts.py does the same as the previous one for SQLShare. Same IMPORTANT note: reorder the cuts in the output csv and manually write them into sqlshare-cuts-snorkel.csv

- agreement is computed with accord.py, it takes as input the files produced with the other scripts and ground truth csv



**To do**

- clean code

- rewrite to avoid manual file creation

- remove absolute path in source files

- write a meta file that runs all the pipeline

- include java code to create ID files

- add Willeme's code to produce metrics from raw log files

- rationalize the way query parts are computed to avoid double counting (NoP, NoA, NoS)

- generate genuine SQL queries from smartBI queries
 
- rationalize thresholds used for normalization 
