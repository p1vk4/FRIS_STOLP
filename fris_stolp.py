from sklearn import datasets
import numpy as np
import argparse
import time


class Fris_stolp():
    def __init__(self, X, y, l, gamma):
        self.X = X
        self.y = y
        self.l = l
        self.gamma = gamma
        self.omega = []
    
    @staticmethod
    def nn(x, sample):
        dist = []
        for u in sample:
            dist.append(np.linalg.norm(u - x))
        return sample[dist.index(min(dist))]
    
    @staticmethod
    def fris_function(u, x, xx):
        p_u_xx =  np.linalg.norm(u - xx)
        p_u_x = np.linalg.norm(u - x)
        return (p_u_xx - p_u_x) / (p_u_xx + p_u_x)
    
    def find_etalon(self, X_y, diff_X_l_x_y, current_omega):
        e_x = []
        for i, x in enumerate(X_y):
            summ = 0
            for j, u in enumerate(X_y):
                if i != j:
                    summ += self.fris_function(u, x, self.nn(u, current_omega))
            dx = 1 / (len(X_y) - 1) * summ
            summ = 0
            for v in diff_X_l_x_y:
                summ += self.fris_function(v, x, self.nn(v, current_omega))
            tx = (1 / len(diff_X_l_x_y)) * summ
            e_x.append(self.l * dx + (1 - self.l) * tx)
        return X_y[e_x.index(max(e_x))]


    def create_x_y(self, y_class):
        X_y = []
        diff_X_l_x_y = []
        for i in range(len(self.y)):
            if self.y[i] == y_class:
                X_y.append(self.X[i])
            else:
                diff_X_l_x_y.append(self.X[i])
        return X_y, diff_X_l_x_y
    
    def initialization(self):
        omega = []
        for y_class in np.unique(self.y):
            X_y, diff_X_l_x_y = self.create_x_y(y_class)
            omega.append(self.find_etalon(X_y, diff_X_l_x_y, diff_X_l_x_y))
        omega_c = []
        for y_class in np.unique(self.y):
            X_y, diff_X_l_x_y = self.create_x_y(y_class)
            omega_c.append([self.find_etalon(X_y, diff_X_l_x_y, [o for i, o in enumerate(omega) if i != y_class])])
        self.omega = omega_c

    def find_all_etalons(self):
        while(True):
            if len(self.X):
                old_len = sum([len(o) for o in self.omega])
                U = []
                for i in range(len(self.X)):
                    S = self.fris_function(self.X[i], self.nn(self.X[i], self.omega[self.y[i]]), self.nn(self.X[i], [o for j, o in enumerate(self.omega) if j != self.y[i]]))
                    if S > self.gamma:
                        U.append(self.X[i])    
                idxs = []
                for i in range(len(self.X)):
                    if len([True for u in U if (self.X[i] == u).all()]):
                        idxs.append(i)
                self.X = np.delete(self.X, idxs, axis = 0)
                self.y = np.delete(self.y, idxs)
                for y_class in np.unique(self.y):
                    X_y, diff_X_l_x_y = self.create_x_y(y_class)
                    if len(X_y) < 2 or len(diff_X_l_x_y) < 2:
                        continue
                    self.omega[y_class] = np.concatenate((self.omega[y_class],[self.find_etalon(X_y, diff_X_l_x_y, [o for j, o in enumerate(self.omega) if j != y_class])]), axis = 0)
                if sum([len(o) for o in self.omega]) == old_len:
                    break
            else:
                break
                
def main():
    parser = argparse.ArgumentParser(description='FRIS-STOLP')
    parser.add_argument('--l', type=float, default = 0.7,
        help='lambda')
    parser.add_argument('--gamma', type=float, default = 0.5,
        help='gamma')
    args = parser.parse_args()
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target

    fris_stolp = Fris_stolp(X, y, args.l, args.gamma)
    start_time = time.time()
    fris_stolp.initialization()
    print('*' * 10, 'initialization complete')
    fris_stolp.find_all_etalons()
    print('*' * 10, 'i found %d etalons' % sum([len(o) for o in fris_stolp.omega]))
    print("--- work time %s seconds ---" % (time.time() - start_time))

    
if __name__ == '__main__':
    main()
