# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:50:01 2018

@author: YorozuyaSaint
"""

import numpy as np
from ad import adnumber
from ad.admath import *
import math

def nonlinear_newton(x, f, eps, max_iter, log=True):
    """
    calculates the roots based on newton's numerical method
    uses admath to call math functions
    params :
        x : initial values using adnumber object, e.g :  x=[0,0,0]
        f : list of lambda f(x), e.g : f=[lambda x : x**2+1, lambda x : x**3+admath.sin(x), lambda x : admath.cos(x**2)]
        eps : stopping criteria ||x(k)-x(k-1)||
        max_iter : max iteration
        log : log switch
    returns :
        x : roots approximation
        k : iteration
    """
    k=0
    if log:
        print("Init State")
        print("x =",x)
        print("eps = ",eps)
        print("max_iter = ",max_iter,"\n")
        print("iteration\tx_vector\terr")
    while(k<max_iter):
        x_prev = np.copy(x)
        x_ad = adnumber(x)
        y=np.zeros(len(f))
        J_f = np.zeros((len(f), len(x_ad)))
        for i in range(len(f)):
            y[i]=f[i](x_ad).x
            for j in range(len(x_ad)):
                J_f[i][j] = f[i](x_ad).d(x_ad[j])
        x-=np.matmul(np.linalg.inv(J_f), y) #x(k)=x(k-1)-J^(-1)*f(x)
        err=np.linalg.norm(x-x_prev)
        if log:
            print(str(k)+"\t", str(x)+"\t", str(err)+"\n")    
        if err<eps:
            break
        k+=1
        
    return x, k

if __name__ == "__main__":
 
    #example at page 641 Burden-Faires book
    x=np.array([0.1, 0.1, -0.1])
    f=[lambda x: (3*x[0]) - admath.cos(x[1]*x[2]) - 1/2,
       lambda x: x[0]**2 - (81*(x[1]+0.1)**2) + admath.sin(x[2]) + 1.06,
       lambda x: admath.exp(-x[0]*x[1]) + 20*x[2] + (10*math.pi-3)/3]
    x, k = nonlinear_newton(x, f, 1e-10, 100)

    
    
    