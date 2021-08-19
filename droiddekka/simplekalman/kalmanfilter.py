import numpy as np
import copy
from math import log, exp,sqrt



class simplekalmanfilter:
    def __init__(self, dim_x, dim_y, dim_u=0):
        if dim_x <= 1:
            raise ValueError("number of state variables has to be greater than 1")
        if dim_y <= 1:
            raise ValueError("number of measurement inputs has to be more than one")
        if dim_u < 0:
            raise ValueError("control unit size must not be negative")
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_u = dim_u

        self.X = np.zeros((dim_x,1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.A = np.eye(dim_x)
        self.B = np.zeros((dim_x,dim_x//2))
        self.u = np.zeros((dim_u,1))
        self.Y = np.zeros((dim_y,1))
        self.C = np.zeros((dim_y,dim_x))
        self.Z = np.zeros((dim_y,1))
        self.H = np.zeros((dim_y,dim_x))
        self.w = np.zeros((dim_x,1))