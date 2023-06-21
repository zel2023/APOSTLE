
# Experimental Reproduction
Run `main.py` following one of the arguments ('RQ1', 'RQ2')

For instance, execute the following command to obtain the results of research question 2 in the paper.
```
python main.py RQ2
```


## RQ1

OUTPUT: 
* **performance:** The results of the APOSTLE.

## RQ2

Compare the state of the art approaches.

* **Static approach:** ML-based approach.
* **Dynamic approach:** Patch-Sim. Please run `python evaluate_patchsim.py`

OUTPUT:
* **performance:** The results of the state-of-the-art.

## RQ3.1

Customize: Choose one of representation embeddings for the test case under *config.py*.

* **codebert:** a pre-trained model for programming and natural languages.
* **graphcodebert:**  a pre-trained model for programming language that considers the inherent structure of code.
* **unixcoder:** an Unified Cross-Modal Pre-training for Code Representation.
* **code2vec:**  a neural network model that learns a representation of code changes guided by their accompanying log messages, which represent the semantic intent of the code changes.

Customize: Choose one of representation embeddings for the patch under *config.py*.

* **codebert:** 
* **graphcodebert:** 
* **unixcoder:**

OUTPUT:
* **performance:** the results of APOSTLE with different pre-trained models used.

## RQ3.2

Customize: Choose different thresholds and cofficients under *evaluate.py* and *word2vector.py*.

* **threshold1:** a threshold used to assess whether the amount of code changes is excessive.
* **threshold2:** a threshold used to assess whether the code semantics have changed too much.
* **cofficient1:** a cofficient used to indicate that a patch with excessive code changes is the correct expected reduction value.
* **cofficient2:** a cofficient used to indicate that a patch with excessive degree of semantic change is the correct expected reduction value.

OUTPUT:
* **performance:** the results of APOSTLE with different thresholds and cofficients.

## RQ3.3
* **Method1:** For each historical related patch, the Euclidean distance between it and the current patch is calculated and the one with the minimum Euclidean distance is obtained to calculate the similarity.
* **Method2:** The test case similarity is used to balance all historical related patches, resulting in a representative patch, which is then used to calculate the similarity with the current patch.
* **Method3:** The similarity between all historical related patches and the current patch is calculated, and the maximum value is chosen.

OUTPUT:
* **performance:** the results of APOSTLE with different patch similarity calculation methods.
