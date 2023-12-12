# Experimental Reproduction
Run `main.py` following one of the arguments ('apostle', 'RQ2', 'method1', 'method2', 'method3')

Attention: Preferably use CPU to run the code.

## RQ1

Command:  python main.py apostle

OUTPUT: 
* **performance:** The results of the APOSTLE. (You can change the parameters under *config.py* to run the code)

Attention:
* **Some parameter Settings:** You should set threshold1 to 700, threshold2 to 0.2, coefficient1 to 0.55 and coefficient2 to 0.95 under *config.py*.
* **BATS:** Because of the dataset APOSTLE uses is equal to the dataset BATS uses, so you can get the results of the BATS by the code at https://github.com/HaoyeTianCoder/BATS.
* **Qualitative analysis result:** You can obtain the qualitative analysis result of APOSTLE by observing the variables 'y_true' and 'y_pred' in the 'predict_collected_projects' function of *evaluate.py*, which represent the predicted result and the reference result, respectively.

## RQ2

Command:  python main.py RQ2

Compare the state of the art approaches.

* **Static approach:** ML-based approach.
* **Dynamic approach:** Patch-Sim. Please run `python evaluate_patchsim.py`.

OUTPUT:
* **performance:** The results of the state-of-the-art.

Attention:
* **APOSTLE:** The results of the APOSTLE can be got by `python main.py apostle`.

## RQ3.1

Command:  python main.py apostle

Customize: Choose one of representation embeddings for the test case under *config.py*.

* **codebert:** a pre-trained model for programming and natural languages.
* **graphcodebert:**  a pre-trained model for programming language that considers the inherent structure of code.
* **unixcoder:** an Unified Cross-Modal Pre-training for Code Representation.
* **code2vec:**  a neural network model that learns a representation of code changes guided by their accompanying log messages, which represent the semantic intent of the code changes.

Customize: Choose one of representation embeddings for the patch under *config.py*.

* **codebert** 
* **graphcodebert** 
* **unixcoder**

OUTPUT:
* **performance:** the results of APOSTLE with different pre-trained models used.

Attention:
* **Some parameter Settings** We did experiment setting threshold1 to 700, threshold2 to 0.2, coefficient1 to 0.55 and coefficient2 to 0.95 under *config.py.You can try other parameter combinations, and then you may get different result from ours.

## RQ3.2

Command:  python main.py apostle

Customize: Choose different thresholds and cofficients under *config.py*

* **threshold1:** a threshold used to assess whether the amount of code changes is excessive.
* **threshold2:** a threshold used to assess whether the code semantics have changed too much.
* **cofficient1:** a cofficient used to indicate that a patch with excessive code changes is the correct expected reduction value.
* **cofficient2:** a cofficient used to indicate that a patch with excessive degree of semantic change is the correct expected reduction value.

OUTPUT:
* **performance:** the results of APOSTLE with different thresholds and cofficients.

Attention:
Parameter tuning experiments on a small dataset can be done just by modifying line 511 in the evaluatie.py file to 'projects = {'Chart': 26}'.

## RQ3.3

Command:  python main.py method1/method2/method3

Customize: Choose different thresholds and cofficients by different argument

* **Method1:** For each historical related patch, the Euclidean distance between it and the current patch is calculated and the one with the minimum Euclidean distance is obtained to calculate the similarity.
* **Method2:** The test case similarity is used to balance all historical related patches, resulting in a representative patch, which is then used to calculate the similarity with the current patch.
* **Method3:** The similarity between all historical related patches and the current patch is calculated, and the maximum value is chosen.

OUTPUT:
* **performance:** the results of APOSTLE with different patch similarity calculation methods.

## Ablation Study

Command: python main.py apostle

Customize: By setting coefficients under *config.py*, we can get the results.

* **M1:** The complete APOSTLE method.(set coefficient1 to 0.55 and coefficient2 to 0.95)
* **M2:** The APOSTLE without the help of indicator of the degree of code semantic change.(set coefficient1 to 0.55 and coefficient2 to 1)
* **M3:** The APOSTLE without the help of indicator of the amount of code change.(set coefficient1 to 1 and coefficient2 to 0.95)

OUTPUT:
* **performance:** the results of APOSTLE or APOSTLE without someone component.
