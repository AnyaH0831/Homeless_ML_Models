# README

- Need to train models without Indigenous Flag and years homeless since Ottawa doesn't include real Indigenous related data and Lanark doesn't have real years homeless.

- Mental health and substance use should be avoided if possible since they are extracted from the comment section in Lanark data, therefore it can only be possitive or unknown. 

- When training the models under the 4 conditions, always limit the features that we are using to be the same shared common features.
  - ‘age’,  'gender', 'has_dependents',  'outdoor_sleeping', 'chronic_homeless', 'youth', 'no_income', 'income_type' and some of their corresponding availability flags: 'has_dependents_available_in_source', 'outdoor_sleeping_available_in_source'
    - Can potentially add in mental health (and its corresponding flags) and substance use (and its corresponding flags)


To run synth_xgboost.py:
python3 synth_xgboost.py --toronto data/synthetic_toronto_sasm.csv --ottawa data/synthetic_ottawa_sasm.csv --lanark data/real_lanark.csv


## Different Training Conditions: 

Condition A: Lanark Only Data
- Logistical Regression: logreg_lanark.py
- XGBoost: xgboost_lanark.py

Condition B: Train the model using synthetic data only and validate it on Lanark data
- Logistical Regression: synth_log.py
- XGBoost: synth_xgboost.py

Condition C: Pretrain using synthetic data then fine-tune using Lanark data
- Logistical Regression: logreg_finetune.py
- XGBoost: xgboost_finetune.py

Condition D: Partial Pooling 
- partial_pooling.py
- Partial Pooling with Random Slopes: partial_pooling_random_slopes.py