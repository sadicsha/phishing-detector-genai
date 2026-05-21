# Adversarial Evaluation Results

## Summary
| Metric | Value |
|---|---|
| Total synthetic phishing tested | 798 |
| Caught by DistilBERT detector | 798 (100.0%) |
| Evaded detector | 0 (0.0%) |

## Detection Rate by Lure Type
| Lure Type | Caught | Total | Rate |
|---|---|---|---|
| account_suspended | 133 | 133 | 100.0% |
| it_alert | 134 | 134 | 100.0% |
| hr_notice | 134 | 134 | 100.0% |
| delivery_scam | 133 | 133 | 100.0% |
| invoice_fraud | 134 | 134 | 100.0% |
| password_reset | 130 | 130 | 100.0% |

## Key Findings
- Overall adversarial detection rate: **100.0%**
- False negatives saved to: `results/false_negatives.json`
- These will be used as hard negatives for retraining

## What evasion patterns were found?
Analyze `results/false_negatives.json` to identify common patterns
in emails that evaded the detector (formal tone, fewer urgency words, etc.)
