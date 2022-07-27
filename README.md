Implementation of ICLR 2020 paper "Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models", based on Fairseq and PyTorch.

For details, refer to `fairseq/fairseq/tasks/translation_multi_simple_epoch_vaccine.py`


## Run
```
bash setup.py

cd script

bash preprocess.sh
bash train.sh
bash test.sh

# need to reset some paths in the bash scripts
```

## Results

WMT10 EN -> FR(High-R),RO(Low-R) , sp=1  :
|        BLEU           | fr    | ro    | avg    |
|---------------------------------------|-------|-------|--------|
| Baseline Multilingual                 | 34.2  | 31.86 | 33.03  |
| w/ PCGrad                             | 34.31 | 31.55 | 32.93  |
| w/ GradVac fix_obj alpha=0.5 (paper)  | 34.34 | 33    | **33.67**  |
| w/ GradVac all-layers alpha=0.0 ema=0.01 (paper) | 33.65 | 31.83 | 32.74  |
