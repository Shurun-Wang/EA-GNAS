![image](https://github.com/user-attachments/assets/23828093-09c0-493c-b51c-0c90de4d529d)# EA-GNAS
PyTorch Source code for "Optimizing Graph Neural Network Architectures for Schizophrenia Spectrum Disorder Prediction Using Evolutionary Algorithms".

You can also find other open-sourced biomedical signal analysis projects in my [academic](https://shurun-wang.github.io/) page. :relaxed: :relaxed: :relaxed:

## Data Preparation
 - HC/SCH data needs to be downloaded and placed in `Data/HC_SCH/`
 - The user needs to obtain data usage rights from the webpage [https://bicr-resource.atr.jp/srpbsfc/](https://bicr-resource.atr.jp/srpbsfc/)

## How to run this project
This project contains one main file as follows:

### Searching the proper structure of the graph neural network
`python main_IOAGraph.py `
- Here, you can get the proper model structure （Search flag = True）.
  - Switch different intelligient optimization algorithms.
- Also, you can train, test, and explain the proper model structure （explain flag = True）.


If our work is helpful to you, please **Star** it and kindly **Cite** our paper as:  

    @article{WANG2024108419,
title = {Optimizing graph neural network architectures for schizophrenia spectrum disorder prediction using evolutionary algorithms},
journal = {Computer Methods and Programs in Biomedicine},
volume = {257},
pages = {108419},
year = {2024},
issn = {0169-2607},
doi = {https://doi.org/10.1016/j.cmpb.2024.108419},
url = {https://www.sciencedirect.com/science/article/pii/S0169260724004127},
author = {Shurun Wang and Hao Tang and Ryutaro Himeno and Jordi Solé-Casals and Cesar F. Caiafa and Shuning Han and Shigeki Aoki and Zhe Sun}
}

