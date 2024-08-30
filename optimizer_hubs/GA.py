import numpy as np
from numpy.random import rand
from optimizer_hubs.function import Fun


def roulette_wheel(prob):
    num = len(prob)
    C = np.cumsum(prob)
    P = rand()
    for i in range(num):
        if C[i] > P:
            index = i
            break
    return index


def ga(trainx, trainy, edge_index, args, se_sp):
    # Parameters
    CR = 0.8  # crossover rate
    MR = 0.01  # mutation rate
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

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='int')
    Xfood_t = np.zeros([max_iter, dim])
    fitG = float('inf')

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
    gbest_t = np.zeros([1, max_iter], dtype='float')
    t = 0

    gbest_t[0, t] = fitG.copy()
    # print("Generation:", t + 1)
    # print("Best (GA):", curve[0, t])
    t += 1

    while t < max_iter:
        print(t)
        # Probability
        inv_fit = 1 / (1 + fit)
        prob = inv_fit / np.sum(inv_fit)

        # Number of crossovers
        Nc = 0
        for i in range(N):
            if rand() < CR:
                Nc += 1

        x1 = np.zeros([Nc, dim], dtype='int')
        x2 = np.zeros([Nc, dim], dtype='int')
        for i in range(Nc):
            # Parent selection
            k1 = roulette_wheel(prob)
            k2 = roulette_wheel(prob)
            P1 = X[k1, :].copy()
            P2 = X[k2, :].copy()
            # Random one dimension from 1 to dim
            index = np.random.randint(low=1, high=dim - 1)
            # Crossover
            x1[i, :] = np.concatenate((P1[0:index], P2[index:]))
            x2[i, :] = np.concatenate((P2[0:index], P1[index:]))
            # Mutation
            for d in range(dim):
                if rand() < MR:
                    x1[i, d] = 1 - x1[i, d]

                if rand() < MR:
                    x2[i, d] = 1 - x2[i, d]

        # Merge two group into one
        Xnew = np.concatenate((x1, x2), axis=0)
        Xnew = se_sp.modify_codes(Xnew)
        # Fitness
        Fnew = np.zeros([2 * Nc, 1], dtype='float')
        tmp_food = []
        for i in range(2 * Nc):
            Fnew[i, 0] = Fun(trainx, trainy, edge_index, se_sp.decode2net(Xnew[i, :]), args)
            tmp_food.append(Fnew[i, 0])
            if Fnew[i, 0] < fitG:
                Xgb[0, :] = Xnew[i, :]
                fitG = Fnew[i, 0]
        mfood.append(np.mean(tmp_food))

        if t == max_iter-1:
            last_food = tmp_food
        # Elitism
        XX = np.concatenate((X, Xnew), axis=0)
        FF = np.concatenate((fit, Fnew), axis=0)
        # Sort in ascending order
        ind = np.argsort(FF, axis=0)
        for i in range(N):
            X[i, :] = XX[ind[i, 0], :]
            fit[i, 0] = FF[ind[i, 0]]

        # Store result
        gbest_t[0, t] = fitG.copy()
        Xfoodt = se_sp.modify_codes(np.expand_dims(Xgb[0, :], axis=0))
        Xfood_t[t, :] = Xfoodt.reshape(dim)
        # print("Generation:", t + 1)
        # print("Best (GA):", curve[0, t])
        t += 1

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
    # np.save('checkpoints/fitness_history.npy', fitness_history)
    return data