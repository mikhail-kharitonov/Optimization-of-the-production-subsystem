from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import linprog

class ProductionSubsystem (ABC):
    m = 0
    n = 0
    Resource = 0
    # A = 0 
    def __init__(self, m,n):
        self.m = m
        self.n = n
        self.Resource = np.array([0.1, 100, 100, 100, 100])   
        #self.A = np.zeros((n+1, m), dtype=float) 
        
    @abstractmethod        
    def u (self):
        pass
    
    @abstractmethod
    def p (self):
        pass
    
    @abstractmethod
    def N(self):
        pass
    
    @abstractmethod
    def R(self):
        pass
    
    @abstractmethod
    def N_F(self):
        pass
    
    def formB (self):
        b=np.zeros(self.u(), dtype=float)
        for i in np.arange(0,self.u()):
            if i<self.m:
                b[i]=self.Resource[i];
        return b
    
    @abstractmethod    
    def formC (self):
        pass
    
    @abstractmethod
    def formA (self):
        pass

    def OptimalF(self):
        temp=linprog(self.formC(), A_ub=self.formA(), b_ub=self.formB(), method='revised simplex')
        return temp.fun