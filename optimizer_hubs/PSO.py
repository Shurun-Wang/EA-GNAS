import numpy as np
from numpy.random import rand
from optimizer_hubs.function import Fun

def init_velocity(lb, ub, N, dim):
    V = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
        Vmin[0, d] = -Vmax[0, d]
    for i in range(N):
        for d in range(dim):
            V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()

    return V, Vmax, Vmin


def boundary(x, lb, ub):
    if x <= lb:
        x = lb
    if x >= ub:
        x = ub-0.001
    return x


def pso(trainx, trainy, edge_index, args, se_sp):
    # Parameters
    w = 0.9  # inertia weight
    c1 = 2  # acceleration factor
    c2 = 2  # acceleration factor

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
    # Initialize position & velocity
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    Xfood_t = np.zeros([max_iter, dim])
    gbest_t = np.zeros([1, max_iter], dtype='float')
    t = 0
    mfood = []
    while t < max_iter:
        # Binary conversion
        Xbin = se_sp.modify_codes(X)
        # Fitness
        tmp_food = []
        for i in range(N):
            fit[i, 0] = Fun(trainx, trainy, edge_index, se_sp.decode2net(Xbin[i, :]), args)
            if fit[i, 0] < fitP[i, 0]:
                Xpb[i, :] = X[i, :]
                fitP[i, 0] = fit[i, 0]
            if fitP[i, 0] < fitG:
                Xgb[0, :] = Xpb[i, :]
                fitG = fitP[i, 0]
            tmp_food.append(fit[i, 0])
        mfood.append(np.mean(tmp_food))
        if t == max_iter-1:
            last_food = tmp_food
        gbest_t[0, t] = fitG.copy()
        Xfoodt = se_sp.modify_codes(np.expand_dims(Xgb[0, :], axis=0))
        Xfood_t[t, :] = Xfoodt.reshape(dim)
        print("Iteration:", t + 1)
        # print("Best (PSO):", gbest_t[0, t])
        t += 1

        for i in range(N):
            for d in range(dim):
                # Update velocity
                r1 = rand()
                r2 = rand()
                V[i, d] = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + c2 * r2 * (Xgb[0, d] - X[i, d])
                # Boundary
                V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
                # Update position
                X[i, d] = X[i, d] + V[i, d]
                # Boundary
                X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

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








