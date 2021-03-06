## Transformers
This folder includes the different experiments we did with Transformers for OffensEval 2020.

#### English Experiments

| Model                                                                      | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ---------------------------------------------------------------------------|----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| bert-large-cased                                                           | 0.7749    | 0.7626      | 0.7243   | 0.7749         |  0.7703           | 1593 436 160 459 |
| bert-base-cased                                                            | 0.7870    | 0.7856      | 0.7590   | 0.7870         |  0.7847           | 1493 304 260 591 |
| roberta-base                                                               | 0.7874    | 0.7850      | 0.7574   | 0.7874         |  0.7840           | 1508 318 245 577 |
| xlnet-base-cased                                                           | 0.7851    | 0.7841      | 0.7578   | 0.7851         |  0.7834           | 1484 300 269 595 |
| xlnet-large-cased                                                          | 0.7795    | 0.7806      | 0.7562   | 0.7795         |  0.7821           | 1441 272 312 623 |

#### English Experiments with HASOC Transfer Learning

| Model                                                                      | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ---------------------------------------------------------------------------|----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| bert-base-cased                                                            | 0.7802    | 0.7808      | 0.7557   | 0.7802         |  0.7815           | 1452 281 301 614 |


#### Greek Experiments

| Model                                                                      | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ---------------------------------------------------------------------------|----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| bert-multilingual-cased                                                    | 0.8325    | 0.8239      | 0.7714   | 0.8325         |  0.8266           | 1180 210  83 276 |
| distilbert-multilingual-cased                                              | 0.8399    | 0.8315      | 0.7810   | 0.8399         |  0.8351           | 1188 205  75 281 |
| bert-base-greek-uncased-v1                                                 | 0.8553    | 0.8548      | 0.8185   | 0.8553         |  0.8544           | 1142 132 121 354 |


#### Danish Experiments

| Model                                                                      | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ---------------------------------------------------------------------------|----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| bert-multilingual-cased                                                    | 0.9207    | 0.9180      | 0.8131   | 0.9206         |  0.9166           | 497 29 18 48     |
| distilbert-multilingual-cased                                              | 0.9222    | 0.9166      | 0.8038   | 0.9222         |  0.9165           | 503 34 12 43     |
| danish_bert_uncased_v2                                                     | 0.8936    | 0.8939      | 0.7662   | 0.8936         |  0.8942           | 483 31 32 46     |


#### Turkish Experiments

| Model                                                                      | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ---------------------------------------------------------------------------|----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| bert-multilingual-cased                                                    | 0.8377    | 0.8255      | 0.7054   | 0.8377         |  0.8225           | 4717 701 314 524 |
| distilbert-multilingual-cased                                              |           |             |          |                |                   |                  |
| dbmdz/bert-base-turkish-cased                                              | 0.8843    | 0.8769      | 0.7938   | 0.8843         |  0.8778           | 4838 531 193 694 |

#### Arabic Experiments

| Model                                                                      | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ---------------------------------------------------------------------------|----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| distilbert-multilingual-cased                                              |  0.8920   | 0.8841      |  0.7905  |   0.892        |  0.8853           |  794 81 27 98    |
