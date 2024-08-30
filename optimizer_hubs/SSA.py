# [2017]-"Salp swarm algorithm: A bio-inspired optimizer_hubs for engineering design problems"

import numpy as np
from numpy.random import rand
from optimizer_hubs.function import Fun


def boundary(x, lb, ub):
    if x <= lb:
        x = lb
    if x >= ub:
        x = ub-0.001
    return x


def ssa(trainx, trainy, edge_index, args, se_sp):

    N = args.num_individuals
    max_iter = args.num_iterations

    # Dimension
    codes_batch = []
    for i in range(N):
        codes_batch.append(se_sp.initial_instance())
    codes_batch = np.stack(codes_batch, axis=0)
    dim = np.size(codes_batch, 1)
    # codes modified
    X = se_sp.modify_codes(codes_batch)
    lb, ub = se_sp.bounds()
    lb, ub = np.expand_dims(lb, axis=0), np.expand_dims(ub, axis=0)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xf = np.zeros([1, dim], dtype='float')
    fitF = float('inf')
    Xfood_t = np.zeros([max_iter, dim])
    gbest_t = np.zeros([1, max_iter], dtype='float')
    t = 0
    mfood = []

    while t < max_iter:
        # Binary conversion
        Xbin = se_sp.modify_codes(X)
        tmp_food = []
        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(trainx, trainy, edge_index, se_sp.decode2net(Xbin[i, :]), args)
            if fit[i, 0] < fitF:
                Xf[0, :] = X[i, :]
                fitF = fit[i, 0]
            tmp_food.append(fit[i, 0])
        mfood.append(np.mean(tmp_food))
        if t == max_iter-1:
            last_food = tmp_food
        gbest_t[0, t] = fitF.copy()
        Xfoodt = se_sp.modify_codes(np.expand_dims(Xf[0, :], axis=0))
        Xfood_t[t, :] = Xfoodt.reshape(dim)
        # print("Iteration:", t + 1)
        # print("Best (SSA):", curve[0, t])
        t += 1

        # Compute coefficient, c1 (3.2)
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)

        for i in range(N):
            # First leader update
            if i == 0:
                for d in range(dim):
                    # Coefficient c2 & c3 [0 ~ 1]
                    c2 = rand()
                    c3 = rand()
                    # Leader update (3.1)
                    if c3 >= 0.5:
                        X[i, d] = Xf[0, d] + c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])
                    else:
                        X[i, d] = Xf[0, d] - c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])

                    # Boundary
                    X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                    # Salp update
            elif i >= 1:
                for d in range(dim):
                    # Salp update by following front salp (3.4)
                    X[i, d] = (X[i, d] + X[i - 1, d]) / 2
                    # Boundary
                    X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                    # Best feature subset
    # Best feature subset
    fval = fitF.copy()
    Gbin = se_sp.modify_codes(np.expand_dims(Xf[0, :], axis=0))
    Gbin = Gbin.reshape(dim)
    best_structure = se_sp.decode2net(Gbin)
    m_food = np.array(mfood)
    # Best feature subset
    data = {'best_structure': best_structure, 'best_fitness': fval}
    np.save('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost)+'/'+'fitness.npy', gbest_t)
    np.save('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost)+'/'+'structure.npy', Xfood_t)
    np.save('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost)+'/'+'mfood.npy', m_food)
    np.save('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost)+'/'+'last_food.npy', last_food)
    return data