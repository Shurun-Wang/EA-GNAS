# [2014]-"Grey wolf optimizer_hubs"

import numpy as np
from numpy.random import rand
from optimizer_hubs.function import Fun


def boundary(x, lb, ub):
    if x <= lb:
        x = lb
    if x >= ub:
        x = ub-0.001
    return x


def gwo(trainx, trainy, edge_index, args, se_sp):
    # Parameters
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

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')
    Xfood_t = np.zeros([max_iter, dim])
    gbest_t = np.zeros([1, max_iter], dtype='float')

    mfood = []
    tmp_food = []
    for i in range(N):
        fit[i, 0] = Fun(trainx, trainy, edge_index, se_sp.decode2net(X[i, :]), args)
        if fit[i, 0] < Falpha:
            Xalpha[0, :] = X[i, :]
            Falpha = fit[i, 0]

        if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
            Xbeta[0, :] = X[i, :]
            Fbeta = fit[i, 0]

        if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
            Xdelta[0, :] = X[i, :]
            Fdelta = fit[i, 0]
        tmp_food.append(fit[i, 0])
    mfood.append(np.mean(tmp_food))
    # Pre
    t = 0

    gbest_t[0, t] = Falpha.copy()
    Xfoodt = se_sp.modify_codes(np.expand_dims(Xalpha[0, :], axis=0))
    Xfood_t[t, :] = Xfoodt.reshape(dim)
    t += 1

    while t < max_iter:
        # Coefficient decreases linearly from 2 to 0
        a = 2 - t * (2 / max_iter)

        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1 = 2 * rand()
                C2 = 2 * rand()
                C3 = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])
                # Parameter A (3.3)
                A1 = 2 * a * rand() - a
                A2 = 2 * a * rand() - a
                A3 = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.6)
                X1 = Xalpha[0, d] - A1 * Dalpha
                X2 = Xbeta[0, d] - A2 * Dbeta
                X3 = Xdelta[0, d] - A3 * Ddelta
                # Update wolf (3.7)
                X[i, d] = (X1 + X2 + X3) / 3
                # Boundary
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

        # Binary conversion
        Xbin = se_sp.modify_codes(X)
        tmp_food = []
        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(trainx, trainy, edge_index, se_sp.decode2net(Xbin[i, :]), args)
            if fit[i, 0] < Falpha:
                Xalpha[0, :] = X[i, :]
                Falpha = fit[i, 0]

            if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]

            if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]
            tmp_food.append(fit[i, 0])
        mfood.append(np.mean(tmp_food))
        if t == max_iter-1:
            last_food = tmp_food
        gbest_t[0, t] = Falpha.copy()
        Xfoodt = se_sp.modify_codes(np.expand_dims(Xalpha[0, :], axis=0))
        Xfood_t[t, :] = Xfoodt.reshape(dim)
        t += 1

    # Best feature subset
    fval = Falpha.copy()
    Gbin = se_sp.modify_codes(np.expand_dims(Xalpha[0, :], axis=0))
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




