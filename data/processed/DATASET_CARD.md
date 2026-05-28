# Dataset Card — v2

## Summary
| Split | Total | Phishing | Legitimate |
|---|---|---|---|
| Train | 2,660 | 1,610 | 1,050 |
| Val   | 570 | 345 | 225 |
| Test  | 570 | 345 | 225 |
| **Total** | **3,800** | **2,300** | **1,500** |

## Sources
- Real phishing : SpamAssassin corpus (1,500 emails)
- Real legitimate: SpamAssassin ham corpus (1,500 emails)
- Synthetic      : Groq Llama-3.1 generated (798 emails)

## Synthetic Lure Types
| Lure Type | Count |
|---|---|
| it_alert | 135 |
| account_suspended | 134 |
| hr_notice | 134 |
| invoice_fraud | 134 |
| delivery_scam | 133 |
| password_reset | 130 |

## Files
- `data/processed/dataset_v2.jsonl` — full merged dataset
- `data/processed/splits/train.jsonl`
- `data/processed/splits/val.jsonl`
- `data/processed/splits/test.jsonl`
- `results/dataset_v2_comparison.png` — distribution charts
