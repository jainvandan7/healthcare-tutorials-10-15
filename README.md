# Healthcare Analytics — Tutorials 10–15

Python source for tutorials T10–T15 of the healthcare analytics project.

## Files

| File | Tutorials | Dataset |
|------|-----------|---------|
| `healthcare_nlp.py` | **T10** clinical text classification · **T11** clinical named entity recognition · **T12** word sense disambiguation | MTSamples — ~5,000 real transcribed clinical notes across 40 specialties (Kaggle: `tboyle10/medicaltranscriptions`) |
| `healthcare_lifestyle.py` | **T13** dimensionality reduction for exploratory analysis · **T14** behavioural factors and adult weight, part 1 · **T15** behavioural factors and adult weight, part 2 | NHANES 2017–2018 (cycle J) — measured body weight plus behavioural questionnaires |

Each module is standalone: separate datasets, no shared state, and they run
independently of each other and of the T1–T9 image work.

## Expected data layout

Neither dataset is included here — download them and place them as follows,
relative to the directory you run the scripts from:

```
data/mtsamples/mtsamples.csv                  # T10–T12
data/nhanes/{DEMO_J,BMX_J,PAQ_J,SLQ_J,ALQ_J,SMQ_J,DBQ_J,DR1TOT_J}.XPT   # T13–T15
```

NHANES source: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/

## Dependencies

```
numpy pandas matplotlib scikit-learn scipy statsmodels
```

## Running

```bash
python healthcare_nlp.py        # T10–T12
python healthcare_lifestyle.py  # T13–T15
```

Figures and tables are written to `outputs_nlp/` and `outputs_lifestyle/`.
