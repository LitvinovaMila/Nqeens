import numpy as np
import random


class Solver_8_queens:
    def __init__(self, pop_size=150, cross_prob=0.6, mut_prob=0.25):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.len_vector = 48
        self.population = np.zeros((self.pop_size, self.len_vector))
        for i in range(self.pop_size):
            dot_array = np.random.randint(0, 8, 16)
            bol_str = np.array([])
            for j in dot_array:
                for l in range(3):
                    result_div = divmod(j, 2)
                    bol_str = np.hstack((bol_str, result_div[1]))
                    j = result_div[0]
            self.population[i] = bol_str
        self.pop_fit = np.zeros(self.pop_size)

    def __fit_func(self):
        self.pop_fit = np.zeros(self.population.shape[0])
        for l in range(self.population.shape[0]):
            Y = np.zeros((8, 2))
            for i in range(8):
                for j in [0, 1]:
                    for k in [0, 1, 2]:
                        Y[i, j] += self.population[l, i * 6 + j * 3 + k] * 2 ** k
            row = 8 - len(set(Y[:, 0]))
            column = 8 - len(set(Y[:, 1]))
            diag_1 = 8 - len(set(Y[:, 0] + Y[:, 1]))
            diag_2 = 8 - len(set(Y[:, 0] - Y[:, 1]))
            self.pop_fit[l] = 1 / (1 + row + column + diag_1 + diag_2)

    def __mutation(self):
        for i in range(self.population.shape[0]):
            if random.random() <= self.mut_prob:
                k = random.randint(0, np.shape(self.population)[1] - 1)
                self.population[i, k] = 0 if self.population[i, k] > 0 else 1

    def __rep(self):
        self.__fit_func()
        sum_f = self.pop_fit.sum()
        P = self.pop_fit * self.pop_size / sum_f

        while P.sum() >= self.pop_size:
            min_P = P.max()
            for i in P:
                if (i < min_P) & (i != 0):
                    min_P = i
            for i in range(np.shape(P)[0]):
                if P[i] == min_P:
                    P[i] -= 1
                if P.sum() < self.pop_size:
                    break
        while P.sum() < self.pop_size - 1:
            for i in range(np.shape(P)[0]):
                if P[i] == P.max():
                    P[i] += 1
                if P.sum() == self.pop_size - 1:
                    break
        X_rep = np.zeros(48)
        for i in range(np.shape(P)[0]):
            for j in range(int(round(P[i], 0))):
                X_rep = np.vstack((X_rep, self.population[i]))
        self.population = X_rep.copy()

    def __cross(self):
        array_shuffle = np.random.permutation(self.population.shape[0])
        while len(array_shuffle) > 0:
            for j in range(1, array_shuffle.shape[0]):
                if (self.population[array_shuffle[0], :] -
                    self.population[array_shuffle[j], :]).any():
                    if random.random() <= self.cross_prob:
                        K = random.randint(1, self.len_vector - 1)
                        A = self.population[array_shuffle[0]].copy()
                        B = self.population[array_shuffle[j]].copy()
                        array_shuffle = np.delete(array_shuffle, [j])
                        x = A[K:self.len_vector]
                        A[K:self.len_vector] = B[K:self.len_vector]
                        B[K:self.len_vector] = x
                        self.population = np.vstack((self.population, A))
                        self.population = np.vstack((self.population, B))
                    break
            array_shuffle = np.delete(array_shuffle, [0])

    def slove(self, min_fitness=0.9, max_epoch=500):
        epoch = 0
        flag = 0
        while (flag == 0) & (epoch < max_epoch):
            # Мутация
            self.__mutation()
            # Репликация
            self.__rep()
            # Кроссинговер
            self.__cross()
            # Вычисляем фитнесс-функци
            self.__fit_func()
            # сортируем
            sort = np.argsort(self.pop_fit, axis=0)
            sort = sort[::-1]
            X_sort = np.zeros((np.shape(sort)[0], self.len_vector))
            fit_sort = np.zeros(np.shape(sort)[0])
            i = 0
            for j in sort:
                X_sort[i, :] = self.population[j, :]
                fit_sort[i] = self.pop_fit[j]
                i += 1
            # удаляем блезницов
            X_not_repeat = X_sort[0:1, :]
            fit_not_repeat = fit_sort[0:1]
            for i in range(1, np.shape(X_sort)[0]):
                if (X_sort.shape[1] in np.sum(X_sort[i, :] == X_not_repeat, axis=1)) is False:
                    X_not_repeat = np.vstack((X_not_repeat, X_sort[i, :]))
                    fit_not_repeat = np.hstack((fit_not_repeat, fit_sort[i]))
            # отбираем N лучших
            self.population = X_not_repeat[0:self.pop_size - 1]
            self.pop_fit = fit_not_repeat[0:self.pop_size - 1]

            if self.pop_fit.max() >= min_fitness:
                flag = 1
            epoch += 1

        res_prom = self.population[0, :]
        res_dot = np.zeros((8, 2))
        for i in range(8):
            for j in [0, 1]:
                for k in [0, 1, 2]:
                    res_dot[i, j] += int(res_prom[i * 6 + j * 3 + k] * 2 ** k)
        res = np.array([])
        for i in range(64):
            res = np.hstack((res, ['*']))
        res.shape = (8, 8)
        for i in range(np.shape(res_dot)[0]):
            res[int(res_dot[i, 0]), int(res_dot[i, 1])] = 'Q'
        return self.pop_fit[0], epoch, res


count_epoch = np.array([])
for index in range(50):
    slover = Solver_8_queens()
    best_fit, epoch_num, visualization = slover.slove()
    ans = "No"
    if best_fit == 1.0:
        ans = "Yes"
        print(visualization)
        count_epoch = np.hstack((count_epoch, epoch_num))
    print(index, ans, best_fit)
print(count_epoch, count_epoch.shape[0])
