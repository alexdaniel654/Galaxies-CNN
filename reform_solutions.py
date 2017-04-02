# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:10:33 2017

@author: Alex Daniel
"""

import pandas as pd
import numpy as np

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

data = pd.read_csv('training_solutions_rev1.csv',',')
datanorm = np.zeros(data.shape)



datanorm[:, 0] = data['GalaxyID']
datanorm[:, 1] = data['Class1.1']
datanorm[:, 2] = data['Class1.2']
datanorm[:, 3] = data['Class1.3']
datanorm[:, 4] = probnorm2(data['Class2.1'], data['Class2.2'])
datanorm[:, 5] = probnorm2(data['Class2.2'], data['Class2.1'])
datanorm[:, 6] = probnorm2(data['Class3.1'], data['Class3.2'])
datanorm[:, 7] = probnorm2(data['Class3.2'], data['Class3.1'])
datanorm[:, 8] = probnorm2(data['Class4.1'], data['Class4.2'])
datanorm[:, 9] = probnorm2(data['Class4.2'], data['Class4.1'])
datanorm[:, 10] = probnorm4(data['Class5.1'], data['Class5.2'], data['Class5.3'], data['Class5.4'])
datanorm[:, 11] = probnorm4(data['Class5.2'], data['Class5.1'], data['Class5.3'], data['Class5.4'])
datanorm[:, 12] = probnorm4(data['Class5.3'], data['Class5.2'], data['Class5.1'], data['Class5.4'])
datanorm[:, 13] = probnorm4(data['Class5.4'], data['Class5.2'], data['Class5.3'], data['Class5.1'])
datanorm[:, 14] = probnorm2(data['Class6.1'], data['Class6.2'])
datanorm[:, 15] = probnorm2(data['Class6.2'], data['Class6.1'])
datanorm[:, 16] = probnorm3(data['Class7.1'], data['Class7.2'], data['Class7.3'])
datanorm[:, 17] = probnorm3(data['Class7.2'], data['Class7.1'], data['Class7.3'])
datanorm[:, 18] = probnorm3(data['Class7.3'], data['Class7.2'], data['Class7.1'])
datanorm[:, 19] = probnorm7(data['Class8.1'], data['Class8.2'], data['Class8.3'], data['Class8.4'], data['Class8.5'], data['Class8.6'], data['Class8.7'])
datanorm[:, 20] = probnorm7(data['Class8.2'], data['Class8.1'], data['Class8.3'], data['Class8.4'], data['Class8.5'], data['Class8.6'], data['Class8.7'])
datanorm[:, 21] = probnorm7(data['Class8.3'], data['Class8.2'], data['Class8.1'], data['Class8.4'], data['Class8.5'], data['Class8.6'], data['Class8.7'])
datanorm[:, 22] = probnorm7(data['Class8.4'], data['Class8.2'], data['Class8.3'], data['Class8.1'], data['Class8.5'], data['Class8.6'], data['Class8.7'])
datanorm[:, 23] = probnorm7(data['Class8.5'], data['Class8.2'], data['Class8.3'], data['Class8.4'], data['Class8.1'], data['Class8.6'], data['Class8.7'])
datanorm[:, 24] = probnorm7(data['Class8.6'], data['Class8.2'], data['Class8.3'], data['Class8.4'], data['Class8.5'], data['Class8.1'], data['Class8.7'])
datanorm[:, 25] = probnorm7(data['Class8.7'], data['Class8.2'], data['Class8.3'], data['Class8.4'], data['Class8.5'], data['Class8.6'], data['Class8.1'])
datanorm[:, 26] = probnorm3(data['Class9.1'], data['Class9.2'], data['Class9.3'])
datanorm[:, 27] = probnorm3(data['Class9.2'], data['Class9.1'], data['Class9.3'])
datanorm[:, 28] = probnorm3(data['Class9.3'], data['Class9.2'], data['Class9.1'])
datanorm[:, 29] = probnorm3(data['Class10.1'], data['Class10.2'], data['Class10.3'])
datanorm[:, 30] = probnorm3(data['Class10.2'], data['Class10.1'], data['Class10.3'])
datanorm[:, 31] = probnorm3(data['Class10.3'], data['Class10.2'], data['Class10.1'])
datanorm[:, 32] = probnorm6(data['Class11.1'], data['Class11.2'], data['Class11.3'], data['Class11.4'], data['Class11.5'], data['Class11.6'])
datanorm[:, 33] = probnorm6(data['Class11.2'], data['Class11.1'], data['Class11.3'], data['Class11.4'], data['Class11.5'], data['Class11.6'])
datanorm[:, 34] = probnorm6(data['Class11.3'], data['Class11.2'], data['Class11.1'], data['Class11.4'], data['Class11.5'], data['Class11.6'])
datanorm[:, 35] = probnorm6(data['Class11.4'], data['Class11.2'], data['Class11.3'], data['Class11.1'], data['Class11.5'], data['Class11.6'])
datanorm[:, 36] = probnorm6(data['Class11.5'], data['Class11.2'], data['Class11.3'], data['Class11.4'], data['Class11.1'], data['Class11.6'])
datanorm[:, 37] = probnorm6(data['Class11.6'], data['Class11.2'], data['Class11.3'], data['Class11.4'], data['Class11.5'], data['Class11.1'])

head = pd.read_csv('head.csv', ',')
df = pd.DataFrame(data = datanorm, index = datanorm[:,0], columns=head.columns)

datareform = pd.DataFrame(data = np.zeros([data.shape[0], 158]), index = df['GID'])

datareform[0] = df['GID']
datareform[1] = df['1.1']*df['7.1']
datareform[2] = df['1.1']*df['7.2']
datareform[3] = df['1.1']*df['7.3']
datareform[4] = df['1.2']*df['2.1']*df['9.1']
datareform[5] = df['1.2']*df['2.1']*df['9.2']
datareform[6] = df['1.2']*df['2.1']*df['9.3']
datareform[7] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.1']*df['5.1']
datareform[8] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.1']*df['5.2']
datareform[9] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.1']*df['5.3']
datareform[10] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.1']*df['5.4']
datareform[11] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.2']*df['5.1']
datareform[12] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.2']*df['5.2']
datareform[13] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.2']*df['5.3']  
datareform[14] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.2']*df['5.4']
datareform[15] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.3']*df['5.1']
datareform[16] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.3']*df['5.2']
datareform[17] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.3']*df['5.3']
datareform[18] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.3']*df['5.4']
datareform[19] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.4']*df['5.1']
datareform[20] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.4']*df['5.2']
datareform[21] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.4']*df['5.3']
datareform[22] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.4']*df['5.4']
datareform[23] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.5']*df['5.1']
datareform[24] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.5']*df['5.2']
datareform[25] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.5']*df['5.3']
datareform[26] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.5']*df['5.4']
datareform[27] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.6']*df['5.1']
datareform[28] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.6']*df['5.2']
datareform[29] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.6']*df['5.3']
datareform[30] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.1']*df['11.6']*df['5.4']      
datareform[31] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.1']*df['5.1']
datareform[32] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.1']*df['5.2']
datareform[33] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.1']*df['5.3']
datareform[34] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.1']*df['5.4']
datareform[35] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.2']*df['5.1']
datareform[36] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.2']*df['5.2']
datareform[37] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.2']*df['5.3']  
datareform[38] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.2']*df['5.4']
datareform[39] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.3']*df['5.1']
datareform[40] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.3']*df['5.2']
datareform[41] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.3']*df['5.3']
datareform[42] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.3']*df['5.4']
datareform[43] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.4']*df['5.1']
datareform[44] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.4']*df['5.2']
datareform[45] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.4']*df['5.3']
datareform[46] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.4']*df['5.4']
datareform[47] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.5']*df['5.1']
datareform[48] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.5']*df['5.2']
datareform[49] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.5']*df['5.3']
datareform[50] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.5']*df['5.4']
datareform[51] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.6']*df['5.1']
datareform[52] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.6']*df['5.2']
datareform[53] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.6']*df['5.3']
datareform[54] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.2']*df['11.6']*df['5.4'] 
datareform[55] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.1']*df['5.1']
datareform[56] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.1']*df['5.2']
datareform[57] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.1']*df['5.3']
datareform[58] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.1']*df['5.4']
datareform[59] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.2']*df['5.1']
datareform[60] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.2']*df['5.2']
datareform[61] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.2']*df['5.3']  
datareform[62] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.2']*df['5.4']
datareform[63] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.3']*df['5.1']
datareform[64] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.3']*df['5.2']
datareform[65] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.3']*df['5.3']
datareform[66] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.3']*df['5.4']
datareform[67] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.4']*df['5.1']
datareform[68] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.4']*df['5.2']
datareform[69] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.4']*df['5.3']
datareform[70] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.4']*df['5.4']
datareform[71] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.5']*df['5.1']
datareform[72] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.5']*df['5.2']
datareform[73] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.5']*df['5.3']
datareform[74] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.5']*df['5.4']
datareform[75] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.6']*df['5.1']
datareform[76] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.6']*df['5.2']
datareform[77] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.6']*df['5.3']
datareform[78] = df['1.2']*df['2.2']*df['3.1']*df['4.1']*df['10.3']*df['11.6']*df['5.4'] 
datareform[79] = df['1.2']*df['2.2']*df['3.1']*df['4.2']*df['5.1']    
datareform[80] = df['1.2']*df['2.2']*df['3.1']*df['4.2']*df['5.2']
datareform[81] = df['1.2']*df['2.2']*df['3.1']*df['4.2']*df['5.3']
datareform[82] = df['1.2']*df['2.2']*df['3.1']*df['4.2']*df['5.4']
datareform[83] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.1']*df['5.1']
datareform[84] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.1']*df['5.2']
datareform[85] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.1']*df['5.3']
datareform[86] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.1']*df['5.4']
datareform[87] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.2']*df['5.1']
datareform[88] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.2']*df['5.2']
datareform[89] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.2']*df['5.3']  
datareform[90] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.2']*df['5.4']
datareform[91] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.3']*df['5.1']
datareform[92] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.3']*df['5.2']
datareform[93] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.3']*df['5.3']
datareform[94] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.3']*df['5.4']
datareform[95] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.4']*df['5.1']
datareform[96] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.4']*df['5.2']
datareform[97] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.4']*df['5.3']
datareform[98] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.4']*df['5.4']
datareform[99] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.5']*df['5.1']
datareform[100] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.5']*df['5.2']
datareform[101] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.5']*df['5.3']
datareform[102] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.5']*df['5.4']
datareform[103] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.6']*df['5.1']
datareform[104] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.6']*df['5.2']
datareform[105] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.6']*df['5.3']
datareform[106] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.1']*df['11.6']*df['5.4']      
datareform[107] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.1']*df['5.1']
datareform[108] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.1']*df['5.2']
datareform[119] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.1']*df['5.3']
datareform[110] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.1']*df['5.4']
datareform[111] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.2']*df['5.1']
datareform[112] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.2']*df['5.2']
datareform[113] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.2']*df['5.3']  
datareform[114] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.2']*df['5.4']
datareform[115] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.3']*df['5.1']
datareform[116] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.3']*df['5.2']
datareform[117] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.3']*df['5.3']
datareform[118] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.3']*df['5.4']
datareform[129] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.4']*df['5.1']
datareform[120] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.4']*df['5.2']
datareform[121] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.4']*df['5.3']
datareform[122] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.4']*df['5.4']
datareform[123] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.5']*df['5.1']
datareform[124] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.5']*df['5.2']
datareform[125] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.5']*df['5.3']
datareform[126] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.5']*df['5.4']
datareform[127] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.6']*df['5.1']
datareform[128] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.6']*df['5.2']
datareform[139] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.6']*df['5.3']
datareform[130] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.2']*df['11.6']*df['5.4'] 
datareform[131] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.1']*df['5.1']
datareform[132] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.1']*df['5.2']
datareform[133] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.1']*df['5.3']
datareform[134] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.1']*df['5.4']
datareform[135] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.2']*df['5.1']
datareform[136] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.2']*df['5.2']
datareform[137] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.2']*df['5.3']  
datareform[138] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.2']*df['5.4']
datareform[139] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.3']*df['5.1']
datareform[140] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.3']*df['5.2']
datareform[141] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.3']*df['5.3']
datareform[142] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.3']*df['5.4']
datareform[143] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.4']*df['5.1']
datareform[144] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.4']*df['5.2']
datareform[145] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.4']*df['5.3']
datareform[146] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.4']*df['5.4']
datareform[147] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.5']*df['5.1']
datareform[148] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.5']*df['5.2']
datareform[159] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.5']*df['5.3']
datareform[150] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.5']*df['5.4']
datareform[151] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.6']*df['5.1']
datareform[152] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.6']*df['5.2']
datareform[153] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.6']*df['5.3']
datareform[154] = df['1.2']*df['2.2']*df['3.2']*df['4.1']*df['10.3']*df['11.6']*df['5.4'] 
datareform[155] = df['1.2']*df['2.2']*df['3.2']*df['4.2']*df['5.1']    
datareform[156] = df['1.2']*df['2.2']*df['3.2']*df['4.2']*df['5.2']
datareform[157] = df['1.2']*df['2.2']*df['3.2']*df['4.2']*df['5.3']
datareform[158] = df['1.2']*df['2.2']*df['3.2']*df['4.2']*df['5.4']
datareform[159] = df['1.3']    

datareform.to_csv('train_reform.csv')