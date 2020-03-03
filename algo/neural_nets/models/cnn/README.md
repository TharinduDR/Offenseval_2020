## CNN
This folder includes the different experiments we did with CNNs for DeftEval 2020.

#### English Experiments

| Model                                                                    | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ------------------------------------------------------------------------ |----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| CNN - Glove                                                              | 0.7817    | 0.7829      | 0.7589   | 0.7817         |  0.7846           | 1442 267 311 628 |
| CNN - FastText                                                           | 0.7655    | 0.7701      | 0.7496   | 0.7655         |  0.7827           | 1346 214 407 681 |
| CNN - FastText <ul><li>Remove Words</li></ul>                            | 0.7681    | 0.7722      | 0.7510   | 0.7681         |  0.7821           | 1364 225 389 670 |
| CNN - FastText <ul><li>Remove words</li><li>Remove Stop words</li></ul>  | 0.7779    | 0.7800      | 0.7567   | 0.7779         |  0.7834           | 1421 256 332 639 |


#### Arabic Experiments

| Model                                                                    | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ------------------------------------------------------------------------ |----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| CNN - FastText                                                           | 0.9110    | 0.9107      | 0.8476   | 0.9110         |  0.9104           | 778 46 43 133 |


#### Danish Experiments

| Model                                                                    | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ------------------------------------------------------------------------ |----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| CNN - FastText                                                           | 0.8952    | 0.8868      | 0.7321   | 0.8952         |  0.8840           | 496 43 19 34 |


#### GREEK Experiments

| Model                                                                    | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ------------------------------------------------------------------------ |----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| CNN - FastText                                                           | 0.8365    |  0.8331      | 0.7882   | 0.8364         |  0.8319           | 1149 172 114 314 |
