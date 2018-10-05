# -*- coding: utf-8 -*-
"""
created by @yorozuya
"""


import numpy as np


def linear_iterative_methods(A, b, x, eps, max_iter, method="gauss_seidel", log=True):
    D=np.diag(np.diag(A))
    T=A-D
    err=0
    k=1
    if log:
        print("Init state")
        print("A =\n",A)
        print("b =",b)
        print("x =",x)
        print("max_iter =",max_iter,", method =",method,"\n")
        print("iteration\tx_vector\terr")
    while(k<=max_iter):
        x_prev = np.copy(x)
        for i in range(len(A)):
            if method=="jacobian":
                x[i] = ((np.sum(-1*T[i]*x_prev))+b[i]) / A[i][i]
            elif method=="gauss_seidel":
                x[i] = ((np.sum(-1*T[i]*x))+b[i]) / A[i][i]
        err=np.linalg.norm(x-x_prev)/np.linalg.norm(x, np.inf)
        if log:
            print(str(k)+"\t", str(x)+"\t", str(err)+"\n")
        if(err<eps):
            break
        k+=1
    return x, k
        

if __name__ == "__main__":
    #example at page 451 Burden-Faires book
    x=np.zeros((4))
    A=np.array([[10,-1,2,0],[-1,11,-1,3],[2,-1,10,-1],[0,3,-1,8]])
    b=np.array([6, 25, -11, 15])
    x, k=linear_iterative_methods(A, b, x, 0.001, 20, method = 'gauss_seidel')
    
    #exercise : 9.b
    x=np.zeros((3))
    A=np.array([[2,-1,1], [2, 2, 2], [-1, -1, 2]])
    b=np.array([-1, 4, 5])
    x, k=linear_iterative_methods(A, b, x, 0.001, 100, method='jacobian')
    
    #exercise : 9.d
    x=np.zeros((3))
    A=np.array([[2,-1,1], [2, 2, 2], [-1, -1, 2]])
    b=np.array([-1, 4, 5])
    x, k=linear_iterative_methods(A, b, x, 0.00001, 100)
    
    #exercise : 10.b
    x=np.zeros((3))
    A=np.array([[1, 2, -2], [1, 1, 1], [2, 2, 1]])
    b=np.array([7, 2, 5])
    x, k=linear_iterative_methods(A, b, x, 0.00001, 100, method='jacobian')
    
    #exercise : 10.d
    x=np.zeros((3))
    A=np.array([[1, 2, -2], [1, 1, 1], [2, 2, 1]])
    b=np.array([7, 2, 5])
    x, k=linear_iterative_methods(A, b, x, 0.00001, 100)
    
    #exercise : 11.c
    x=np.zeros((3))
    A=np.array([[1, 0, -1], [-1/2, 1, -1/4], [1, -1/2, 1]])
    b=np.array([0.2, -1.425, 2])
    x, k=linear_iterative_methods(A, b, x, 0.01, 300)

    #exercise : 11.d
    x=np.zeros((3))
    A=np.array([[1, 0, -2], [-1/2, 1, -1/4], [1, -1/2, 1]])
    b=np.array([0.2, -1.425, 2])
    x, k=linear_iterative_methods(A, b, x, 0.01, 300)