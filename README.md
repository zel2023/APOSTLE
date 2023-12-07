# Automated Patch cOrrectness aSsessmenT based on muLtiple pErspectives


# APOSTLE
APOSTLE, a learning-based unsupervised classification model for predicting the correctness of patches based on multiple perspectives.

## I) Requirements

* **Code2Vec representation model.**
  1. Download the trained model and uncompress the file.
  `wget https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz tar -xvzf java14m_model.tar.gz`
  2. Update the variable `MODEL_MODEL_LOAD_PATH` in [./representation/word2vector.py] according to destination folder of trained model

* **CodeBERT/UniXcoder/GraphCodeBERT representation model.**
  1. When the code runs, these models (except for codebert) can be automatically redirected to the huggingface website for download.
  2. You will need to download codebert's pre-trained model files from https://huggingface.co/microsoft/codebert-base and place them in the codebert folder under the representation folder(you need to create a new folder called codebert).
  3. If the If the code makes an error when downloading the pre-trained model, you can manually download the pre-trained model files on huggingface and update the variable `MODEL_CODEBERT` in [./representation/word2vector.py] according to destination folder of trained model. For an example of how to get a manually downloaded file, see Point 2 for the codebert model.

## II) Reproduction
  Follow the [experiment/README.md] to obtain the experimental results in the paper.

## III) File Orginization
  There are three folders. Every folder make a special part.

  * **data**
  It is used to store some data files, including intermediate result files, final result files, test case data files.

  * **experiment**
  It is used to store the core code files of the experiment.

  * **representation**
  It is used to store code files related to code vectorization modules, and also includes some pre-trained model files.

