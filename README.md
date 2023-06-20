# Automated Patch cOrrectness aSsessmenT based on muLtiple pErspectives


# APOSTLE
APOSTLE,  a learning-based unsupervised classification model for predicting the correctness of patches based on multiple perspectives.

## Ⅰ) Requirements

* **Code2Vec representation model.**
  1. Download the trained model and uncompress the file.
  `wget https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz tar -xvzf java14m_model.tar.gz`
  2. Update the variable `MODEL_MODEL_LOAD_PATH` in [./word2vector.py](https://github.com/HaoyeTianCoder/BATS/blob/main/representation/word2vector.py) according to destination folder of trained model 

## Ⅱ) Reproduction
  Follow the [experiment/README.md] to obtain the experimental results in the paper.
