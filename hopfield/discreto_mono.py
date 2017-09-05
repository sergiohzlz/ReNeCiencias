#!/usr/env python

import numpy as np
from math import tanh

def crea_pesos(V, d, dbg=False):
    M = np.zeros((d,d))
    for s in range(len(V)):
        print "vector ",s,": ",V[s]
        v = V[s]
        for i in range(len(v)):
            for j in range(len(v)):
                M[i,j] += (2*v[i]-1)*(2*v[j]-1) if i!=j else 0
                if(dbg): print M
    return M

def actualiza(M,e,O):
    for n in O:
        print "mult ",M[:,n], " y ", e
        p = int(np.dot(M[:,n],e.T))
        e[n] = 1 if p>=0 else 0
        #print "p, ",p," tanh(p) ",tanh(p)
        #e[n] = redondea(tanh(p))
        print "n: ",n," atractor: "+str(e)


def redondea(val,ep = 0.5):
    s = -1 if val <0 else 1
    nx = int(abs(val))+1
    return nx*s if abs(abs(val)-nx)<=ep else int(val)

if __name__ =='__main__':
    V = [ (-1,1,1,-1,1), (1,-1,1,-1,1)]
    A = crea_pesos(V, 5)

    print "Ejemplo 1"
    P = [2,0,4,1,3]*2 #2 epocas para actualizar los 5 nodos
    e = np.array([1,1,1,1,1])
    print "El estimulo que introducimos a la red es: "+str(e)
    actualiza(A,e,P)
    print "El atractor en el que terminamos fue: "+str(e)

    print "Ejemplo 2"
    P = [1,3,2,4,0]*2
    e = np.array([1,1,1,1,1])
    print "El estimulo que introducimos a la red es: "+str(e)
    actualiza(A,e,P)
    print "El atractor en el que terminamos fue: "+str(e)
