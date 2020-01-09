## CNN
This folder includes the different experiments we did with CNNs for DeftEval 2020.

#### English Experiments

| Model                                                                    | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ------------------------------------------------------------------------ |----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| CNN - Glove                                                              | 0.7817    | 0.7829      | 0.7589   | 0.7817         |  0.7846           | 1442 267 311 628 |
| CNN - FastText                                                           | 0.7655    | 0.7701      | 0.7496   | 0.7655         |  0.7827           | 1346 214 407 681 |
| CNN - FastText <ul><li>Remove Words</li></ul>                            | 0.7681    | 0.7722      | 0.7510   | 0.7681         |  0.7821           | 1364 225 389 670 |
| CNN - FastText <ul><li>Remove words</li><li>Remove Stop words</li></ul>  | 0.7779    | 0.7800      | 0.7567   | 0.7779         |  0.7834           | 1421 256 332 639 |


