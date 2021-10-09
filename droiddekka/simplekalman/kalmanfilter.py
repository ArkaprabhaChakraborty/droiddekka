import numpy as np
import copy
from math import log, exp,sqrt



class simplekalmanfilter:
    def __init__(self, dim_x, dim_y):
        if dim_x <= 1:
            raise ValueError("number of state variables has to be greater than 1")
        if dim_y < 1:
            raise ValueError("number of measurement inputs has to be more than one")
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.X = np.zeros((dim_x,1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.A = np.eye(dim_x)
        self.B = np.zeros((dim_x,dim_x//2))
        self.u = np.zeros((dim_x//2,1))
        self.Y = np.zeros((dim_y,1))
        self.C = np.eye(dim_y)
        self.Z = np.zeros((dim_y,1))
        self.H = np.zeros((dim_y,dim_x))
        self.w = np.zeros((dim_x,1))
        self.R = np.zeros((dim_x//2,dim_x//2))
        self.dt = 0
        self.acc = 1

    def __repr__(self):
        return "\n".join([
            ("Simple kalman filter object"),
            (f"State shape: {self.X.shape}"),
            (f"Process Covariance shape: {self.P.shape}"),
            (f"Process Noise shape: {self.Q.shape}"),
            (f"State matrix: \n {self.X}"),
            (f"State transition matrix: \n {self.A}"),
            (f"transformation matrix: \n {self.H}"),
            (f"Measurement matrix: \n {self.Y}"),
            (f"Measurement noise covariance matrix: \n {self.R}"),

            ])
    
    def state_process_setter(self,X,dt,std_acc = 1,process_noise=0):
        print(X)
        if self.dim_x == 2:
            X = np.reshape(X,(self.dim_x,1))
            self.X = X
            self.dt = dt
            self.A[0][1] = self.dt  
            self.Q[0][0] = (self.dt ** 4)/4
            self.Q[0][1] = (self.dt ** 3)/2
            self.Q[1][0] = (self.dt ** 3)/2
            self.Q[1][1] = (self.dt ** 2)
            self.R[0][0] = (self.dt ** 4)/4
            self.Q = self.Q * std_acc
            self.acc = std_acc
            self.w = self.w + process_noise
            self.H[0][0] = 1
        elif self.dim_x == 4:
            X = np.reshape(X,(self.dim_x,1))
            self.X = X
            self.dt = dt
            self.A[0][2] = self.A[1][3] = self.dt
            self.Q[0][0] = self.Q[1][1] = (self.dt ** 4)//4
            self.Q[2][0] = self.Q[3][1] = self.Q[0][2] = self.Q[1][3] = (self.dt ** 3)//2
            self.Q[2][2] = self.Q[3][3] = self.dt ** 2
            self.Q = self.Q * std_acc
            self.acc = std_acc
            self.R[0][0] = self.R[1][1] = (self.dt ** 4)/4
            self.w = self.w + process_noise
            self.H[0][0] = self.H[1][1] =1
        #elif self.dim_x == 6:
        #    pass




        
    def predict(self):
        self.X = np.dot(self.A,self.X) + np.dot(self.B,self.u) + self.w
        self.P = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        X = self.X
        P = self.P        
        return (X,P)

    def update(self,Xm = None,z=None):
        if Xm is not None and np.array(Xm).shape!= self.Y.shape:
            print(Xm.shape)
            print(self.Y.shape)
            raise ValueError("Incorrect Y shape")
        if Xm is None:
            Xm = self.Y
        if z is not None and np.array(z).shape!= self.Z.shape:
            raise ValueError("Incorrect Z shape")
        if z is None:
            z = self.Z
        Xm = np.reshape(Xm,(self.dim_y,1))
        self.Y = np.dot(self.C,Xm) + z
        S = np.dot(self.H,np.dot(self.P,self.H.T)) + self.R
        try:
            K = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(S))
        except:
            K = np.dot(np.dot(self.P,self.H.T),np.linalg.cholesky(S))
        self.X = self.X + np.dot(K,(self.Y - np.dot(self.H,self.X)))
        self.P = np.dot(np.eye(self.dim_x) - np.dot(K,self.H),self.P)
        X = self.X
        
        return X

    def low_pass_predict(self, acc = 1., alpha = 0):
        if alpha > 1:
            raise ValueError("alpha cannot be more than 1")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")
        self.X = np.dot(self.A,self.X) + np.dot(self.B,self.u) + self.w
        if acc != self.acc:
            P_k = np.dot(np.dot(self.A,self.P),self.A.T) + (self.Q*acc/self.acc)
            self.P = alpha*self.P + (1-alpha)*P_k
        else:
            alpha = 0
            P_k = np.dot(np.dot(self.A,self.P),self.A.T) + (self.Q)
            self.P = alpha*self.P + (1-alpha)*P_k
        X = self.X
        P = self.P        
        return (X,P)






