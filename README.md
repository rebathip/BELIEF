#BELIEF - Bayesian Sign Entropy Regularization for LIME Framework (UAI 2025)

For directory setup:
1) `results/` <br>
&nbsp;&nbsp;&nbsp;&nbsp;└── `oxpets/` *(dataset name)* <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── `belief_resnet50/` *(method_model)* <br><br>
2) `fidelity_results/` <br>
&nbsp;&nbsp;&nbsp;&nbsp;├── `aopc_del/` <br>
&nbsp;&nbsp;&nbsp;&nbsp;├── `aopc_ins/` <br>
&nbsp;&nbsp;&nbsp;&nbsp;├── `del/` <br>
&nbsp;&nbsp;&nbsp;&nbsp;└── `ins/`

For Sign Entropy regularization evaluation on Tabular data
1) Run the script regularization_run.py
2) Use the pickle files obtained from step 1 in regularization_analysis.py to get the plots and statistical tests

For Sign Entropy regularization evaluation on Stabilizing LIME
1) First run the belief_20runs.py and this will populate the results dir for each of the xai methods.
2) Use the output pickle files from step 1 in belief_fidelity_20runs.py to get the fidelity results
3) Use output of step 1 with belief_consistency_analysis.py to get consistency plots and statistical test results
4) USe output of step 2 with belief_fidelity_analysis.py to get fidelity plots and statistical tests

#For Runtime Calculation
1) Use the script runtime_calculation.py to get the average runtime for all methods

