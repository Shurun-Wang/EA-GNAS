import warnings
from core.utiles import *
from core.configs import config
from core.ioa import IOA

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    seed_everything(6767)
    search_flag = False
    explain_flag = True
    # -*-*- 🌟 1: configs 🌟 -*-*-
    args = config()
    make_files(args)
    ioa = IOA(args)

    ## Using the following three lines code to test the searched structure
    # from checkpoints.searched_structure import one_structure
    # structure = one_structure('ga')
    # ioa.cross_vali(structure)
    ##

    # -*-*- 🚀 2: start ioa 🚀 -*-*-
    if search_flag:
        best_structure, best_fitness = ioa.start_search()
        print(best_structure)
        print(best_fitness)
    # -*-*- 🐍 3: explain the searched structure 🐍 -*-*-
    # for example: GA is good
    if explain_flag:
        from checkpoints.searched_structure import one_structure
        structure = one_structure('ga')
        ioa.train_test_explain_one_structure(structure)





