# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:01:45 2017

@author: Alex Daniel
"""
import numpy as np
import pandas as pd

def probnorm2(p1, p2):
    eta = 1e-12
    po = p1/(p1+p2+eta)
    return po

def probnorm3(p1, p2, p3):
    eta = 1e-12
    po = p1/(p1+p2+p3+eta)
    return po

def probnorm4(p1, p2, p3, p4):
    eta = 1e-12
    po = p1/(p1+p2+p3+p4+eta)
    return po

def probnorm6(p1, p2, p3, p4, p5, p6):
    eta = 1e-12
    po = p1/(p1+p2+p3+p4+p5+p6+eta)
    return po

def probnorm7(p1, p2, p3, p4, p5, p6, p7):
    eta = 1e-12
    po = p1/(p1+p2+p3+p4+p5+p6+p7+eta)
    return po

p = np.load('pred.npy')

head = pd.read_csv('head.csv', ',')
df = pd.DataFrame(data = np.zeros([p.shape[0], 38]), columns=head.columns)

df['GID'] = np.arange(p.shape[0])+2
df['1.1'] = np.sum(p[:,0:3], axis=1)
df['1.2'] = 1-p[:,158]-df['1.1']
df['1.3'] = p[:,158]

df['7.1'] = probnorm3(p[:,0], p[:,1], p[:,2])
df['7.2'] = probnorm3(p[:,1], p[:,0], p[:,2])
df['7.3'] = probnorm3(p[:,2], p[:,1], p[:,0])