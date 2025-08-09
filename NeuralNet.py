import time
import numpy as np

class CoreFunctions:
    @staticmethod
    def errorRate(Y: np.ndarray, Y_hat: np.ndarray):
        N = Y.shape[0]
        incorrect = 0

        for i in range(N):
            if Y[i, 0] != Y_hat[i, 0]: incorrect += 1
        
        return incorrect / N

    @staticmethod
    def addBiasFwd(u: np.ndarray):
        bias = np.ones((1, 1))
        v = np.vstack((bias, u))
        return v
    
    @staticmethod
    def addBiasBkd(u: np.ndarray, v: np.ndarray, dl_dv: np.ndarray):
        return dl_dv[1:, :]

    @staticmethod
    def linearFwd(u: np.ndarray, W: np.ndarray):
        return W @ u
    
    @staticmethod
    def linearBkd(u: np.ndarray, W: np.ndarray, v: np.ndarray, dl_dv: np.ndarray):
        dl_du = W.T @ dl_dv
        dl_dW = dl_dv @ u.T
                    
        return (dl_du, dl_dW)

    @staticmethod
    def tanhFwd(u: np.ndarray):
        return np.tanh(u)
    
    @staticmethod
    def tanhBkd(u: np.ndarray, v: np.ndarray, dl_dv: np.ndarray):
        return dl_dv * (1 - v**2)

    @staticmethod
    def softmaxFwd(u: np.ndarray):
        return np.exp(u) / np.sum(np.exp(u))
    
    @staticmethod
    def softmaxBkd(v: np.ndarray, dl_dv: np.ndarray):
        D = v.shape[0]
        k = np.eye(D)
        y_i = np.tile(v.T, (D, 1))
        y_j = np.tile(v, (1, D))
        dv_du = y_i * (k - y_j)
                    
        return dv_du.T @ dl_dv
    
    @staticmethod
    def NLLLossFwd(y: np.ndarray, y_prob: np.ndarray):
        return -np.sum(y * np.log(y_prob))
    
    @staticmethod
    def NLLLossBkd(y: np.ndarray, y_prob: np.ndarray):
        return -y / y_prob


class ForwardPass:
    def __init__(self):
        self.core = CoreFunctions()

    def SLNN(self, X: np.ndarray, W1: np.ndarray, W2: np.ndarray):
        Xbias = self.core.addBiasFwd(X) # shape (M+1, 1) 
        a1 = self.core.linearFwd(Xbias, W1) # shape (D1, 1)
        z1 = self.core.tanhFwd(a1) # shape (D1, 1)
        z1bias = self.core.addBiasFwd(z1) # shape (D1+1, 1)
        a2 = self.core.linearFwd(z1bias, W2) # shape (C, 1)
        Yprob = self.core.softmaxFwd(a2) # shape (C, 1)

        return {
            'X': X,
            'Xbias': Xbias,
            'a1': a1,
            'z1': z1,
            'z1bias': z1bias,
            'a2': a2,
            'y_prob': Yprob
        }

    def DLNN(self, X: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray):
        Xbias = self.core.addBiasFwd(X) # shape (M+1, 1) 
        a1 = self.core.linearFwd(Xbias, W1) # shape (D1, 1)
        z1 = self.core.tanhFwd(a1) # shape (D1, 1)
        z1bias = self.core.addBiasFwd(z1) # shape (D1+1, 1)
        a2 = self.core.linearFwd(z1bias, W2) # shape (D2, 1)
        z2 = self.core.tanhFwd(a2) # shape (D2, 1)
        z2bias = self.core.addBiasFwd(z2) # shape (D2+1, 1)
        a3 = self.core.linearFwd(z2bias, W3) # shape (C, 1)
        Yprob = self.core.softmaxFwd(a3) # shape (C, 1)

        return {
            'X': X,
            'Xbias': Xbias,
            'a1': a1,
            'z1': z1,
            'z1bias': z1bias,
            'a2': a2,
            'z2': z2,
            'z2bias': z2bias,
            'a3': a3,
            'y_prob': Yprob
        }
    
class BackwardPass:
    def __init__(self):
        self.core = CoreFunctions()

    def SLNN(self, X: np.ndarray, Y: np.ndarray, W1: np.ndarray, W2: np.ndarray, forward: dict[str, np.ndarray]):
        dl_dyprob = self.core.NLLLossBkd(y = Y, y_prob = forward['y_prob']) # shape (C, 1)
        dl_da2 = self.core.softmaxBkd(v = forward['y_prob'], dl_dv = dl_dyprob) # shape (C, 1)
        dl_dz1bias, dl_dW2 = self.core.linearBkd(u = forward['z1bias'], W = W2, v= forward['a2'], dl_dv= dl_da2) # shape (D1+1, 1) and (C, D1+1)
        dl_dz1 = self.core.addBiasBkd(u= forward['z1'], v= forward['z1bias'], dl_dv= dl_dz1bias) # shape (D1, 1)
        dl_da1 = self.core.tanhBkd(u= forward['a1'], v= forward['z1'], dl_dv= dl_dz1) # shape (D1, 1)
        dl_dXbias, dl_dW1 = self.core.linearBkd(u = forward['Xbias'], W= W1, v= forward['a1'], dl_dv= dl_da1) # shape (M+1, 1) and (D1, M+1)
        dl_dX = self.core.addBiasBkd(u= forward['X'], v= forward['Xbias'], dl_dv= dl_dXbias) # shape (M, 1) 

        return {
            'X': X,
            'dl_dyprob': dl_dyprob,
            'dl_da2' : dl_da2,
            'dl_dW2' : dl_dW2,
            'dl_dz1': dl_dz1,
            'dl_dz1bias': dl_dz1bias,
            'dl_da1': dl_da1,
            'dl_dW1': dl_dW1,
            'dl_dX': dl_dX,
            'dl_dXbias': dl_dXbias
        }
    
    def DLNN(self, X: np.ndarray, Y: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray, forward: dict[str, np.ndarray]):
        dl_dyprob = self.core.NLLLossBkd(y = Y, y_prob = forward['y_prob']) # shape (C, 1)
        dl_da3 = self.core.softmaxBkd(v = forward['y_prob'], dl_dv = dl_dyprob) # shape (C, 1)
        dl_dz2bias, dl_dW3 = self.core.linearBkd(u = forward['z2bias'], W = W3, v= forward['a3'], dl_dv= dl_da3) # shape (D2+1, 1) and (C, D2+1)
        dl_dz2 = self.core.addBiasBkd(u= forward['z2'], v= forward['z2bias'], dl_dv= dl_dz2bias) # shape (D2, 1)
        dl_da2 = self.core.tanhBkd(u= forward['a2'], v= forward['z2'], dl_dv= dl_dz2) # shape (D2, 1)
        dl_dz1bias, dl_dW2 = self.core.linearBkd(u = forward['z1bias'], W = W2, v= forward['a2'], dl_dv= dl_da2) # shape (D1+1, 1) and (D2, D1+1)    
        dl_dz1 = self.core.addBiasBkd(u= forward['z1'], v= forward['z1bias'], dl_dv= dl_dz1bias) # shape (D1, 1)
        dl_da1 = self.core.tanhBkd(u= forward['a1'], v= forward['z1'], dl_dv= dl_dz1) # shape (D1, 1)
        dl_dXbias, dl_dW1 = self.core.linearBkd(u = forward['Xbias'], W= W1, v= forward['a1'], dl_dv= dl_da1) # shape (M+1, 1) and (D1, M+1)
        dl_dX = self.core.addBiasBkd(u= forward['X'], v= forward['Xbias'], dl_dv= dl_dXbias) # shape (M, 1)

        return {
            'X': X,
            'dl_dyprob': dl_dyprob,
            'dl_da3': dl_da3,
            'dl_dW3': dl_dW3,
            'dl_dz2': dl_dz2,
            'dl_dz2bias': dl_dz2bias,
            'dl_da2' : dl_da2,
            'dl_dW2' : dl_dW2,
            'dl_dz1': dl_dz1,
            'dl_dz1bias': dl_dz1bias,
            'dl_da1': dl_da1,
            'dl_dW1': dl_dW1,
            'dl_dX': dl_dX,
            'dl_dXbias': dl_dXbias
        }
    
class NeuralNetModel:
    def __init__(self):
        self.core = CoreFunctions()
        self.fwd = ForwardPass()
        self.bkd = BackwardPass()

    def trainSLNN(self, X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, W1: np.ndarray, W2: np.ndarray, num_epochs: int, lr: float):
        (N_train, M), C = X_train.shape, 10
        best_W1, best_W2, best_val_loss = None, None, None
        train_losses, val_losses = [], []
        
        for e in range(num_epochs):
            train_loss = 0.0
            start = time.time()
            for i in range(N_train):
                
                # extract, reshape, and encode single datapoint
                X_i = X_train[i].reshape((M, 1))
                Y_i = Y_train[i].reshape((C, 1))

                # forward and backward computations
                forward = self.fwd.SLNN(X_i, W1, W2)
                gradient = self.bkd.SLNN(X_i, Y_i, W1, W2, forward)

                # update parameters
                W1 = W1 - lr * gradient['dl_dW1']
                W2 = W2 - lr * gradient['dl_dW2']

                # compute train loss 
                train_loss += self.core.NLLLossFwd(Y_i, forward['y_prob'])
                
            # compute val loss
            N_val = X_val.shape[0]
            val_loss = 0.0
            for i in range(N_val):
                X_i = X_val[i].reshape((M, 1))
                Y_i = Y_val[i].reshape((C, 1))
                forward = self.fwd.SLNN(X_i, W1, W2)
                val_loss += self.core.NLLLossFwd(Y_i, forward['y_prob'])
            
            print('Epoch %d, train loss=%0.4f, val loss=%0.4f, duration=%0.4fs' % (
                        e, train_loss/N_train, val_loss/N_val, time.time() - start))

            # save W1, Wb of the best val loss
            if best_val_loss is None or best_val_loss > val_loss: 
                best_val_loss = val_loss
                best_W1, best_W2 = W1, W2
                
            # save (average) train loss and val loss for training monitoring
            train_losses.append(train_loss/N_train)
            val_losses.append(val_loss/N_val)

        return {
            'best_W1'     : best_W1, 
            'best_W2'     : best_W2, 
            'train_losses': train_losses, 
            'val_losses'  : val_losses
        }
    
    def trainDLNN(self, X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray, num_epochs: int, lr: float):
        (N_train, M), C = X_train.shape, 10
        best_W1, best_W2, best_W3, best_val_loss = None, None, None, None
        train_losses, val_losses = [], []
        
        for e in range(num_epochs):
            train_loss = 0.0
            start = time.time()
            for i in range(N_train):
                
                # extract, reshape, and encode single datapoint
                X_i = X_train[i].reshape((M, 1))
                Y_i = Y_train[i].reshape((C, 1))

                # forward and backward computations
                forward = self.fwd.DLNN(X_i, W1, W2, W3)
                gradient = self.bkd.DLNN(X_i, Y_i, W1, W2, W3, forward)

                # update parameters
                W1 = W1 - lr * gradient['dl_dW1']
                W2 = W2 - lr * gradient['dl_dW2']
                W3 = W3 - lr * gradient['dl_dW3']

                # compute train loss 
                train_loss += self.core.NLLLossFwd(Y_i, forward['y_prob'])
                
            # compute val loss
            N_val = X_val.shape[0]
            val_loss = 0.0
            for i in range(N_val):
                X_i = X_val[i].reshape((M, 1))
                Y_i = Y_val[i].reshape((C, 1))
                forward = self.fwd.DLNN(X_i, W1, W2, W3)
                val_loss += self.core.NLLLossFwd(Y_i, forward['y_prob'])
            
            print('Epoch %d, train loss=%0.4f, val loss=%0.4f, duration=%0.4fs' % (
                        e, train_loss/N_train, val_loss/N_val, time.time() - start))

            # save W1, Wb of the best val loss
            if best_val_loss is None or best_val_loss > val_loss: 
                best_val_loss = val_loss
                best_W1, best_W2, best_W3 = W1, W2, W3
                
            # save (average) train loss and val loss for training monitoring
            train_losses.append(train_loss/N_train)
            val_losses.append(val_loss/N_val)

        return {
            'best_W1'     : best_W1, 
            'best_W2'     : best_W2, 
            'best_W3'     : best_W3, 
            'train_losses': train_losses, 
            'val_losses'  : val_losses
        }
    
    def predictSingle(self, X: np.ndarray, W1: np.ndarray, W2: np.ndarray):
        N, M = X.shape
        yHat = []
        for i in range(N):
            Xi = X[i, :].reshape((M, 1))
            neural = self.fwd.SLNN(Xi, W1, W2)
            Yi = np.argmax(neural['y_prob'], axis=0)
            yHat.append(Yi)

        return {
            'model': neural,
            'labels': np.array(yHat)
        }
    
    def predictDouble(self, X: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray):
        N, M = X.shape
        yHat = []
        for i in range(N):
            Xi = X[i, :].reshape((M, 1))
            neural = self.fwd.DLNN(Xi, W1, W2, W3)
            Yi = np.argmax(neural['y_prob'], axis=0)
            yHat.append(Yi)

        return {
            'model': neural,
            'labels': np.array(yHat)
        }