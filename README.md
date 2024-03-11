# EA-GNAS
PyTorch Source code for "Optimizing Graph Neural Network Architectures for Schizophrenia Spectrum Disorder Prediction Using Evolutionary Algorithms", Under Review.

We will release the code when the article is accepted. You can also find other open-sourced biomedical signal analysis projects in my [academic](https://shurun-wang.github.io/) page. :relaxed: :relaxed: :relaxed:

## Data Preparation
 - HC/SCH data have been already processed in
    `Data/HC_SCH/`

## How to run this project
This project contains one main file as follows:

### Searching the proper structure of the graph neural network
`python main_IOAGraph.py `
- Here, you can get the proper model structure （Search flag = True）.
  - Switch different intelligient optimization algorithms.
- Also, you can train and test the proper model structure （explain flag = True）.
