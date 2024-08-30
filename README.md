# EA-GNAS
PyTorch Source code for "Optimizing Graph Neural Network Architectures for Schizophrenia Spectrum Disorder Prediction Using Evolutionary Algorithms".

You can also find other open-sourced biomedical signal analysis projects in my [academic](https://shurun-wang.github.io/) page. :relaxed: :relaxed: :relaxed:

## Data Preparation
 - HC/SCH data should download in
    `Data/HC_SCH/`
 - The user needs to obtain data usage rights from the webpage [https://bicr-resource.atr.jp/srpbsfc/](https://bicr-resource.atr.jp/srpbsfc/)

## How to run this project
This project contains one main file as follows:

### Searching the proper structure of the graph neural network
`python main_IOAGraph.py `
- Here, you can get the proper model structure （Search flag = True）.
  - Switch different intelligient optimization algorithms.
- Also, you can train, test, and explain the proper model structure （explain flag = True）.
