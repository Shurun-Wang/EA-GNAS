import numpy as np
from numpy.random import rand
from optimizer_hubs.function import Fun

def boundary(x, lb, ub):
    if x <= lb:
        x = lb
    if x >= ub:
        x = ub-0.001
    return x

def so(trainx, trainy, edge_index, args, se_sp):
    global_fitness = []
    global_structure = []

    vec_flag = np.array([1, -1])
    Threshold, Thresold2 = 0.25, 0.6
    C1 = 0.5
    C2 = 0.05
    C3 = 0.5
    N = args.num_individuals
    T = args.num_iterations
    lb, ub = se_sp.bounds()
    lb, ub = np.expand_dims(lb, axis=0), np.expand_dims(ub, axis=0)

    codes_batch = []
    for i in range(N):
        codes_batch.append(se_sp.initial_instance())
    codes_batch = np.stack(codes_batch, axis=0)
    dim = np.size(codes_batch, 1)
    # codes modified
    X = se_sp.modify_codes(codes_batch)
    fitness_list = []
    mfood = []
    tmp_food = []
    for i in range(N):
        fitness = Fun(trainx, trainy, edge_index, se_sp.decode2net(X[i, :]), args)
        fitness_list.append(fitness)
        tmp_food.append(fitness)
    fitness = np.array(fitness_list)
    mfood.append(np.mean(tmp_food))

    GYbest, gbest = np.min(fitness), np.argmin(fitness)
    Xfood = X[gbest, :]

    # Diving the swarm into two equal groups males and females
    Nm = round(N / 2)
    Nf = N - Nm
    Xm = X[0:Nm, :]
    Xf = X[Nm:N, :]
    Xnewm = np.zeros([Nm, dim])
    Xnewf = np.zeros([Nf, dim])
    fitness_m = fitness[0:Nm]
    fitness_f = fitness[Nm:N]
    fitnessBest_m, gbest1 = np.min(fitness_m), np.argmin(fitness_m)
    Xbest_m = Xm[gbest1, :]
    fitnessBest_f, gbest2 = np.min(fitness_f), np.argmin(fitness_f)
    Xbest_f = Xf[gbest2, :]

    gbest_t = np.zeros((T,))
    Xfood_t = np.zeros([T, dim])

    for t in range(T):
        print(t)
        Temp = np.exp(-((t) / T))  # eq.(4)
        Q = C1 * np.exp(((t - T) / T))  # eq.(5)
        Q = min(Q, 1)

        # Exploration Phase (no Food)
        if Q < Threshold:
            for i in range(Nm):
                for j in range(dim):
                    rand_leader_index = int(np.floor(Nm * np.random.rand()))
                    X_randm = Xm[rand_leader_index, :]
                    flag_index = int(np.floor(2 * np.random.rand()))
                    Flag = vec_flag[flag_index]
                    Am = np.exp(-fitness_m[rand_leader_index] / (fitness_m[i] + np.finfo(float).eps))  # eq.(7)
                    Xnewm[i, j] = X_randm[j] + Flag * C2 * Am * ((ub[0, j] - lb[0, j]) * np.random.rand() + lb[0, j])  # eq.(6)
            for i in range(Nf):
                for j in range(dim):
                    rand_leader_index = int(np.floor(Nf * np.random.rand()))
                    X_randf = Xf[rand_leader_index, :]
                    flag_index = int(np.floor(2 * np.random.rand()))
                    Flag = vec_flag[flag_index]
                    Af = np.exp(-fitness_f[rand_leader_index] / (fitness_f[i] + np.finfo(float).eps))  # eq.(9)
                    Xnewf[i, j] = X_randf[j] + Flag * C2 * Af * ((ub[0, j] - lb[0, j]) * np.random.rand() + lb[0, j])  # eq.(8)
        else:  # Exploitation Phase (Food Exists)
            if Temp > Thresold2:  # hot
                for i in range(Nm):
                    flag_index = int(np.floor(2 * np.random.rand()))
                    Flag = vec_flag[flag_index]
                    for j in range(dim):
                        Xnewm[i, j] = Xfood[j] + C3 * Flag * Temp * np.random.rand() * (Xfood[j] - Xm[i, j])  # eq.(10)
                for i in range(Nf):
                    flag_index = int(np.floor(2 * np.random.rand()))
                    Flag = vec_flag[flag_index]
                    for j in range(dim):
                        Xnewf[i, j] = Xfood[j] + C3 * Flag * Temp * np.random.rand() * (Xfood[j] - Xf[i, j])  # eq.(10)
            else:  # cold
                if np.random.rand() > 0.6:  # fight
                    for i in range(Nm):
                        for j in range(dim):
                            FM = np.exp(-(fitnessBest_f) / (fitness_m[i] + np.finfo(float).eps))  # eq.(13)
                            # Xnewm[i, j] = t1[t] * Xm[i, j] + C3[t] * FM * np.random.rand() * (
                            #         Q * Xbest_f[j] - Xm[i, j])
                            Xnewm[i, j] = Xm[i, j] + C3 * FM * np.random.rand() * (Q * Xbest_f[j] - Xm[i, j])  # eq.(11)
                    for i in range(Nf):
                        for j in range(dim):
                            FF = np.exp(-(fitnessBest_m) / (fitness_f[i] + np.finfo(float).eps))  # eq.(14)
                            # Xnewf[i, j] = t2[t] * Xf[i, j] + C3[t] * FF * np.random.rand() * (
                            #         Q * Xbest_m[j] - Xf[i, j])
                            Xnewf[i, j] = Xf[i, j] + C3 * FF * np.random.rand() * (Q * Xbest_m[j] - Xf[i, j])  # eq.(12)
                else:  # mating
                    for i in range(Nm):
                        for j in range(dim):
                            Mm = np.exp(-fitness_f[i] / (fitness_m[i] + np.finfo(float).eps))  # eq.(17)
                            Xnewm[i, j] = Xm[i, j] + C3 * np.random.rand() * Mm * (Q * Xf[i, j] - Xm[i, j])  # eq.(15)
                    for i in range(Nf):
                        for j in range(dim):
                            Mf = np.exp(-fitness_m[i] / (fitness_f[i] + np.finfo(float).eps))  # eq.(18)
                            Xnewf[i, j] = Xf[i, j] + C3 * np.random.rand() * Mf * (Q * Xm[i, j] - Xf[i, j])  # eq.(16)
                    flag_index = int(np.floor(2 * np.random.rand()))
                    egg = vec_flag[flag_index]
                    if egg == 1:
                        GYworst, gworst = np.max(fitness_m), np.argmax(fitness_m)
                        Xnewm[gworst, :] = lb + np.random.rand() * (ub[0, :] - lb[0, :])  # eq.(19)
                        GYworst, gworst = np.max(fitness_f), np.argmax(fitness_f)
                        Xnewf[gworst, :] = lb + np.random.rand() * (ub[0, :] - lb[0, :])  # eq.(20)
        tmp_food = []
        for j in range(Nm):
            Flag4ub = Xnewm[j, :] > ub[0, :]
            Flag4lb = Xnewm[j, :] < lb[0, :]
            Xnewm[j, :] = Xnewm[j, :] * (~(Flag4ub + Flag4lb)) + ub[0, :] * Flag4ub + lb[0, :] * Flag4lb
            for d in range(dim):
                Xnewm[j, d] = boundary(Xnewm[j, d], lb[0, d], ub[0, d])
            Xnewm_bin = se_sp.modify_codes(Xnewm)
            y = Fun(trainx, trainy, edge_index, se_sp.decode2net(Xnewm_bin[j, :]), args)
            tmp_food.append(y)
            if y < fitness_m[j]:
                fitness_m[j] = y
                Xm[j, :] = Xnewm[j, :]
        Ybest1, gbest1 = np.min(fitness_m), np.argmin(fitness_m)

        for j in range(Nf):
            Flag4ub = Xnewf[j, :] > ub[0, :]
            Flag4lb = Xnewf[j, :] < lb[0, :]
            Xnewf[j, :] = Xnewf[j, :] * (~(Flag4ub + Flag4lb)) + ub[0, :] * Flag4ub + lb[0, :] * Flag4lb
            for d in range(dim):
                Xnewf[j, d] = boundary(Xnewf[j, d], lb[0, d], ub[0, d])
            Xnewf_bin = se_sp.modify_codes(Xnewf)
            y = Fun(trainx, trainy, edge_index, se_sp.decode2net(Xnewf_bin[j, :]), args)
            tmp_food.append(y)
            if y < fitness_f[j]:
                fitness_f[j] = y
                Xf[j, :] = Xnewf[j, :]
        Ybest2, gbest2 = np.min(fitness_f), np.argmin(fitness_f)
        mfood.append(np.mean(tmp_food))
        if t == T-1:
            last_food = tmp_food
        if Ybest1 < fitnessBest_m:
            Xbest_m = Xm[gbest1, :]
            fitnessBest_m = Ybest1
        if Ybest2 < fitnessBest_f:
            Xbest_f = Xf[gbest2, :]
            fitnessBest_f = Ybest2

        gbest_t[t] = min(Ybest1, Ybest2)

        if fitnessBest_m < fitnessBest_f:
            GYbest = fitnessBest_m
            Xfood = Xbest_m
        else:
            GYbest = fitnessBest_f
            Xfood = Xbest_f

        gbest_t[t] = GYbest
        Xfoodt = se_sp.modify_codes(np.expand_dims(Xfood, axis=0))
        Xfood_t[t, :] = Xfoodt.reshape(dim)

    fval = GYbest
    Gbin = se_sp.modify_codes(np.expand_dims(Xfood, axis=0))
    Gbin = Gbin.reshape(dim)

    best_structure = se_sp.decode2net(Gbin)
    m_food = np.array(mfood)
    # Create dictionary
    data = {'best_structure': best_structure, 'best_fitness': fval}
    np.save('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost)+'/'+'fitness.npy', gbest_t)
    np.save('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost)+'/'+'structure.npy', Xfood_t)
    np.save('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost)+'/'+'mfood.npy', m_food)
    np.save('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost)+'/'+'last_food.npy', last_food)
    # np.save('checkpoints/fitness_history.npy', fitness_history)
    return data
