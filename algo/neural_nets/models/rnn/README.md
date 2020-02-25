## RNN
This folder includes the different experiments we did with RNNs for DeftEval 2020.

#### English Experiments

| Model                                                                      | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ---------------------------------------------------------------------------|----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| BiGRU- Glove                                                               | 0.7817    | 0.7829      | 0.7589   | 0.7817         |  0.7846           | 1442 267 311 628 |
| BiGRU- Fasttext                                                            | 0.7828    | 0.7854      | 0.7634   | 0.7828         |  0.7901           | 1416 238 337 657 |
| BiGRU - FastText <ul><li>Remove Words</li></ul>                            | 0.7787    | 0.7820      | 0.7609   | 0.7787         |  0.7895           | 1392 225 361 670 |
| BiGRU - FastText <ul><li>Remove words</li><li>Remove Stop words</li></ul>  | 0.7730    | 0.7756      | 0.7526   | 0.7730         |  0.7803           | 1404 252 349 643 |
| BiLSTM - FastText <ul><li>Remove words</li><li>Remove Stop words</li></ul> | 0.7704    | 0.7737      | 0.7516   | 0.7704         |  0.7808           | 1384 239 369 656 |

#### GREEK Experiments

| Model                                                                      | Accuracy  | Weighted F1 | Macro F1 | Weighted Recall| Weighted Precision| (tn, fp, fn, tp) |
| ---------------------------------------------------------------------------|----------:| -----------:| --------:| --------------:| -----------------:| ----------------:|
| BiLSTM- Fasttext                                                           | 0.8536    | 0.8533      | 0.8169   | 0.8536         |  0.8531           | 1138 131 125 355 |

