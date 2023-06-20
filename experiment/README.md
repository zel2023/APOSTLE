
# Experimental Reproduction
Run `main.py` following one of the arguments ('RQ1', 'RQ2')

For instance, execute the following command to obtain the results of research question 2 in the paper.
```
python main.py RQ2
```


## RQ1.1

Customize: 
* **number:** premade the number of clusters. If more than 50, SSE Reduction Rate will be calculated as well.

OUTPUT: 
* **Qualified:** the ratio of clusters that have SC > 0 out of all clusters identified.
* **CSC:** the average value of similarity coefficient (SC) values for all clusters.
* **fig/RQ1/sc_clusters.png:** Similarity coefficient of test cases and patches at each cluster.

## RQ1.2

Customize: pass parameters option1 and option2.
* **option1:** Boolean. whether skip current project-id of search space(all projects or other projects).
* **option2:** Numerical. setup threshold for test cases similarity in scenario H.

OUTPUT: 
* **fig/RQ2/distribution_test_similarity.png:** Distribution on the similarities between each failing test case of each bug and its closest similar test case.
* **fig/RQ2/distribution_pairwise_patches.png:** Distributions on the similarities of pairwise patches.

## RQ2

Compare the state of the art approaches.

* **Static approach:** ML-based approach.
* **Dynamic approach:** Patch-Sim. Please run `python evaluate_patchsim.py`

OUTPUT:
* **performance:** The results of the state of the art.

## RQ3.1

Customize: Choose one of representation embeddings for test case under *config.py*.

* **codebert:** a pre-trained model for programming and natural languages.
* **graphcodebert:**  a pre-trained model for programming language that considers the inherent structure of code.
* **unixcoder:** an Unified Cross-Modal Pre-training for Code Representation.
* **code2vec:** , a neural network model that learns a representation of code changes guided by their accompanying log messages, which represent the semantic intent of the code changes.

Customize: Choose one of representation embeddings for patch under *config.py*.

* **codebert:** 
* **graphcodebert:** 
* **unixcoder:** 

## RQ3.2

Customize: Choose one of representation embeddings for test case under *evaluate.py* and *word2vector.py*.

* **threshold1** 
* **threshold2**
* **cofficient1**
* **cofficient2**


OUTPUT:
* **performance:** classification and ranking of the Baseline and BATS on the APR-generated patches. 
