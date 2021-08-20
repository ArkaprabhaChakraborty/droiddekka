import numpy as np
import copy
from math import log, exp,sqrt



class simplekalmanfilter:
    def __init__(self, dim_x, dim_y):
        if dim_x <= 1:
            raise ValueError("number of state variables has to be greater than 1")
        if dim_y <= 1:
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
    
    def state_process_setter(self,X,dt,process_noise=0):
        print(X)
        if self.dim_x == 2:
            self.X = X
            self.dt = dt
            self.A[0][1] = self.dt  
            self.Q[0][0] = (self.dt ** 4)//4
            self.Q[0][1] = (self.dt ** 3)//2
            self.Q[1][0] = (self.dt ** 3)//2
            self.Q[1][1] = (self.dt ** 2)
            self.R[0][0] = (self.dt ** 4)//4
            self.w = self.w + process_noise
            self.H[0][0] = 1
        elif self.dim_x == 4:
            self.X = X
            self.dt = dt
            self.A[0][2] = self.A[1][3] = self.dt
            self.Q[0][0] = self.Q[1][1] = (self.dt ** 4)//4
            self.Q[2][0] = self.Q[3][1] = self.Q[0][2] = self.Q[1][3] = (self.dt ** 3)//2
            self.Q[2][2] = self.Q[3][3] = self.dt ** 2
            self.R[0][0] = self.R[1][1] = (self.dt ** 4)//4
            self.w = self.w + process_noise
            self.H[0][0] = self.H[1][1] =1
        
    def predict(self):
        self.X = np.dot(self.A,self.X) + np.dot(self.B,self.u) + self.w
        self.P = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        X = self.X
        P = self.P        
        return (X,P)

    def update(self,Xm = None,z=None):
        if Xm is not None and np.array(Xm).shape!= self.Y.shape:
            raise ValueError("Incorrect Z shape")
        if Xm is None:
            Xm = self.Y
        if z is not None and np.array(z).shape!= self.Z.shape:
            raise ValueError("Incorrect Z shape")
        if z is None:
            z = self.Z
        
        self.Y = np.dot(self.C,Xm) + z
        S = np.dot(self.H,np.dot(self.P,self.H.T)) + self.R
        K = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(S))
        self.X = self.X + np.dot(K,(self.Y - np.dot(self.H,self.X)))
        self.P = np.dot(np.eye(self.dim_x) - np.dot(K,self.H),self.P)
        X = self.X[0:self.dim_y]
        
        return X






