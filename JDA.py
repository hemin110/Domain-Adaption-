# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 22:22:49 2017

@author: APAC
"""

import numpy as np
from scipy.linalg.misc import norm
from scipy.sparse.linalg import eigs

def JDA(Xs , Xt , Ys , Yt0 , k=100 , labda = 0.1 , ker = 'primal' , gamma = 1.0 , data = 'default'):
    print 'begin JDA'
    X = np.hstack((Xs , Xt))
    X = np.diag(1/np.sqrt(np.sum(X**2)))
    (m,n) = X.shape
    ns = Xs.shape[1]
    nt = Xt.shape[1]
    C = len(np.unique(Ys))
    # Construct MMD matrix
    e1 = 1/ns*np.ones((ns,1))
    e2 = 1/nt*np.ones((nt,1))
    e = np.vstack((e1,e2))
    M = np.dot(e,e.T)*C
    
    if any(Yt0) and len(Yt0)==nt:
        for c in np.reshape(np.unique(Ys) ,-1 ,1):
            e1 = np.zeros((ns,1))
            e1[Ys == c] = 1/len(Ys[Ys == c])
            e2 = np.zeros((nt,1))
            e2[Yt0 ==c] = -1/len(Yt0[Yt0 ==c])
            e = np.hstack((e1 ,e2))
            e = e[np.isinf(e) == 0]
            M = M+np.dot(e,e.T)
            
    M = M/norm(M ,ord = 'fro' )
    
    # Construct centering matrix
    H = np.eye(n) - 1/(n)*np.ones((n,n))
    
    #% Joint Distribution Adaptation: JDA
    if ker == 'primal':
        A = eigs(np.dot(np.dot(X,M),X.T)+labda*np.eye(m), k=k, M=np.dot(np.dot(X,H),X.T),  which='SM')
        Z = np.dot(A.T,X)
    else:
        pass
    
    print 'JDA TERMINATED'


