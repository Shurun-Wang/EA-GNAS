# [2016]-"The whale optimization algorithm"]

import numpy as np
from numpy.random import rand
from optimizer_hubs.function import Fun


def boundary(x, lb, ub):
    if x <= lb:
        x = lb
    if x >= ub:
        x = ub-0.001
    return x


def woa(trainx, trainy, edge_index, args, se_sp):
    # Parameters
    N = args.num_individuals
    max_iter = args.num_iterations

    b = 1  # constant

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

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xfood_t = np.zeros([max_iter, dim])
    gbest_t = np.zeros([1, max_iter], dtype='float')
    mfood = []
    tmp_food = []
    for i in range(N):
        fit[i, 0] = Fun(trainx, trainy, edge_index, se_sp.decode2net(X[i, :]), args)
        if fit[i, 0] < fitG:
            Xgb[0, :] = X[i, :]
            fitG = fit[i, 0]
        tmp_food.append(fit[i, 0])
    mfood.append(np.mean(tmp_food))
    # Pre

    t = 0
    gbest_t[0, t] = fitG.copy()
    Xfoodt = se_sp.modify_codes(np.expand_dims(Xgb[0, :], axis=0))
    Xfood_t[t, :] = Xfoodt.reshape(dim)
    t = t+1

    while t < max_iter:
        # Define a, linearly decreases from 2 to 0
        a = 2 - t * (2 / max_iter)

        for i in range(N):
            # Parameter A (2.3)
            A = 2 * a * rand() - a
            # Paramater C (2.4)
            C = 2 * rand()
            # Parameter p, random number in [0,1]
            p = rand()
            # Parameter l, random number in [-1,1]
            l = -1 + 2 * rand()
            # Whale position update (2.6)
            if p < 0.5:
                # {1} Encircling prey
                if abs(A) < 1:
                    for d in range(dim):
                        # Compute D (2.1)
                        Dx = abs(C * Xgb[0, d] - X[i, d])
                        # Position update (2.2)
                        X[i, d] = Xgb[0, d] - A * Dx
                        # Boundary
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                # {2} Search for prey
                elif abs(A) >= 1:
                    for d in range(dim):
                        # Select a random whale
                        k = np.random.randint(low=0, high=N)
                        # Compute D (2.7)
                        Dx = abs(C * X[k, d] - X[i, d])
                        # Position update (2.8)
                        X[i, d] = X[k, d] - A * Dx
                        # Boundary
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

            # {3} Bubble-net attacking
            elif p >= 0.5:
                for d in range(dim):
                    # Distance of whale to prey
                    dist = abs(Xgb[0, d] - X[i, d])
                    # Position update (2.5)
                    X[i, d] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + Xgb[0, d]
                    # Boundary
                    X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

        # Binary conversion
        Xbin = se_sp.modify_codes(X)
        # Fitness
        tmp_food = []
        for i in range(N):
            fit[i, 0] = Fun(trainx, trainy, edge_index, se_sp.decode2net(Xbin[i, :]), args)
            if fit[i, 0] < fitG:
                Xgb[0, :] = X[i, :]
                fitG = fit[i, 0]
            tmp_food.append(fit[i, 0])
        if t == max_iter-1:
            last_food = tmp_food
        mfood.append(np.mean(tmp_food))
        # Store result
        gbest_t[0, t] = fitG.copy()
        Xfoodt = se_sp.modify_codes(np.expand_dims(Xgb[0, :], axis=0))
        Xfood_t[t, :] = Xfoodt.reshape(dim)
        t += 1

        # Best feature subset
    # Best feature subset
    fval = fitG.copy()
    Gbin = se_sp.modify_codes(np.expand_dims(Xgb[0, :], axis=0))
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