### Label correction: original vs corrected

18 questions excluded (982 kept: {'arithmetic': 500, 'factual': 482}); among the kept ones, 24 labels changed from needed to not needed and 1 the other way.

| Scope | Original: n (needed) | Probe AUROC, original | Corrected: n (needed) | Probe AUROC, corrected | Best surface baseline, corrected | Averaged input embeddings, corrected |
|---|---|---|---|---|--:|---|
| all | 1000 (608) | 0.900 [0.882, 0.916] | 982 (568) | 0.910 [0.891, 0.927] | 0.788 | 0.830 [0.803, 0.854] |
| arithmetic | 500 (313) | 0.963 [0.948, 0.976] | 500 (313) | 0.963 [0.948, 0.976] | 0.917 | 0.936 [0.913, 0.955] |
| factual | 500 (295) | 0.814 [0.774, 0.852] | 482 (255) | 0.839 [0.804, 0.872] | 0.605 | 0.657 [0.607, 0.704] |

| Prompt | Over-use n, original -> corrected | Probe says not needed on over-use, original -> corrected | ...on not-needed and not called, original -> corrected |
|---|---|---|---|
| baseline | 370 -> 383 | 81% (n=370) -> 84% (n=383) | 86% (n=22) -> 94% (n=31) |
| intervention | 165 -> 165 | 75% (n=165) -> 74% (n=165) | 87% (n=227) -> 91% (n=249) |
