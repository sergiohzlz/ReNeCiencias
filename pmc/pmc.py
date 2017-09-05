#!/usr/bin/python2.6
#-*- coding:utf-8 -*-

import numpy as np;
from random import uniform;

def leer_data_set(arch,n,m):
    """funcion para leer del archivo en el parametro
    un dataset en el que n columnas corresponden a la
    dimension de los datos de entrenamiento y m
    corresponden a la dimension del conjutno de salida"""
    f = open(arch,'rb');
    ss = f.readlines();
    ins = [];
    outs = [];
    for i in range(len(ss)):
        t = map(float, ss[i].split())
        ej1 = t[:n];
        ej2 = t[n:n+m];
        ins.append(ej1);
        outs.append(ej2);
    return [ins,outs];

        
class PMC(object):
    """Clase para implementar el perceptron multicapa
	al que se le pueden pasar los pesos y los umbrales 
	de las neuronas"""
	
	def __init__(self, ins, mids, outs, fmid='tanh', fout='lin', pin=1, b=True):
		"""Constructor:
		ins, mids, outs: cantidad de neuronas de 
		entrada, intermedias y de salida respectivamente"""
		self.bias = b
		if self.bias:
			self.nins = ins+1;
		else:
			self.nins = ins;
		self.nmids = mids;
		self.nouts = outs;
		if fmid=='tanh':
			self.fmid='tanh'
			self.actm=np.tanh
			self.dactm=lambda x: 1-tanh(x)**2
		elif fmid=='sigm':
			self.fmid='sigm'
			self.actm=lambda x: 1/(1+exp(-x))
			self.dactm=lambda x: (1/(1+exp(-x)))*(1-1/(1+exp(-x)))

		if fout=='lin'
			self.fout='lin'
			self.acto=lambda x: x
			self.dacto=lambda x: 1
		elif fout=='sigm':
			self.fout='sigm'
			self.acto=lambda x: 1/(1+exp(-x))
			self.dacto=lambda x: (1/(1+exp(-x)))*(1-1/(1+exp(-x)))
		elif fout=='tanh':
			self.fout='tanh'
			self.acto=np.tanh
			self.dacto=lambda x: 1-tanh(x)**2


		self.ue = np.ones((ins,))
		self.uo = np.ones((mids,));
		self.us = np.ones((outs,));

        #creamos los pesos
		self.wim = np.zeros((self.nins, self.nmids));
		self.wmo = np.zeros((self.nmids, self.nouts));

		##pesos capa media
		##acorde a la funcion
		if pin==1:
			if fmid=='tanh':
				exdr = np.sqrt(6/(self.ins+mids))
				exiz = -1*exdr
			elif fmid=='sigm':
				exdr = 4*np.sqrt(6/(self.ins+mids))
				exiz = -1*exdr
			for i in range(self.nins):
				for j in range(self.nmids):
					self.wim[i,j] = uniform(exiz,exdr)
			for j in range(self.nmids):
				for k in range(self.nouts):
					self.wmo[j,k] = uniform(-1.0, 1.0)
	 
 		##

	def arquitectura(self):
		"""regresa la arquitectura de la red"""
		if self.bias: return (self.nins-1,self.nmids, self.nouts);
		else: return (self.nins,self.nmids, self.nouts);

        def __cromosoma_act(self):
                """Regresa el cromosoma de la actual red neuronal
                en un arreglo unidimensional con el siguiente esquema

                [umbrales ocultos, umbrales salida, pesos e-o, pesos o-s]"""
                cr = []
                
                for j in self.uo:
                        cr.append(j);
                for j in self.us:
                        cr.append(j);
                for j in self.pesos_em:
                        cr.append(j);
                for j in self.pesos_ms:
                        cr.append(j);
                return cr[:];

        def __cromosoma_nvo(self, L):
                """Redefine el nuevo fenotipo a partir del genotipo definido en L
                que es una Lista con la siguiente configuracion

                L =  [umbrales ocultos, umbrales salida, pesos e-o, pesos o-s]"""
                self.uo = np.array(L[:self.nmids]);
                self.us = np.array(L[self.nmids:self.nmids+self.nouts]);
                self.pesos_em = L[self.nmids+self.nouts:self.nmids+self.nouts+self.nins*self.nmids];
                self.pesos_ms = L[self.nmids+self.nouts+self.nins*self.nmids:];

        @property
        def cromosoma(self): return self.__cromosoma_act();
        @cromosoma.setter
        def cromosoma(self, L): self.__cromosoma_nvo(L);
                
	def funact(self, x, param=2, c=1):
		"""Funcion de activacion usada por las neuronas, 
		si param = 1 entonces se usa la sigmoide
		si param = 2 entonces se usa la tanh"""
		if param==1:
			return 1/(1+np.exp(-c*x));
		elif param==2:
			return np.tanh(c*x);

	def error(self, estim, target):
		"""Metodo para calcular el error de lo obtenido con lo ejemplificado,
                lo que hace es sacar la norma infinito"""
		if len(target) != self.nouts:
			raise "Cantidad de ejemplos distintos de cantidad de neuronas de salida";
		else:
			error = 0.0;
			v1 = self.feed(estim);
			for i in range(len(target)):
				error += 0.5*((target[i] - v1)**2);

			return error;
##                        v1 = np.array(self.feed(estim));
##                        v2 = np.array(target);
##                        return v1[v1.argmax()] - v2[v2.argmax()];

	def __dpesos_em(self):
		"""regresa los pesos entre la capa de entrada y media en caso de
		que no vaya nada en el argumento, los regresa en un arreglo 
		de esta forma:
		
		[p11,p12,..,p21,...pnm]

		ie pij, donde i es la i-esima neurona de entrada 
		y j es la j-esima neurona intermedia"""
		arr = [];
                for i in range(self.nins):
                        for j in range(self.nmids):
                                arr.append(self.wim[i,j]);
                return arr

        def __pesos_em(self, wg):
                """Asigna el arreglo lineal wg a los pesos entre la capa de entrada
                e intermedia"""
                for i in range(self.nins):
                        for j in range(self.nmids):
                                self.wim[i,j] = wg[j+self.nmids*i];
                                
        @property
        def pesos_em(self): return self.__dpesos_em();
        @pesos_em.setter
        def pesos_em(self,wg): self.__pesos_em(wg);

	def __dpesos_ms(self):
		"""regresa los pesos entre la capa de media y de salida, en caso de
		que no vaya nada en el argumento, los regresa en un arreglo 
		de esta forma:
		
		[p11,p12,..,p21,...pms]

		ie pij, donde i es la i-esima neurona media
		y j es la j-esima neurona salida"""
		arr = [];
                for i in range(self.nmids):
                        for j in range(self.nouts):
                                arr.append(self.wmo[i,j]);
                return arr
        
        def __pesos_ms(self, wg):
                """Asigna el arreglo wg a los pesos entre la capa intermedia
                y de salida"""
                for i in range(self.nmids):
                        for j in range(self.nouts):
                                self.wmo[i,j] = wg[j+self.nouts*i];

        @property
        def pesos_ms(self): return self.__dpesos_ms();
        @pesos_ms.setter
        def pesos_ms(self, wg): self.__pesos_ms(wg);

	def ff(self, estimulo):
		"""Metodo para enviar estimulos hacia la capa de salida"""
		if self.bias:
                        cant = self.nins - 1
                else:
                        cant = self.nins
                
		if len(estimulo) != cant:
			raise "Cantidad de entradas diferente de neuronas de entrada";
		else:
			for i in range(len(estimulo)):
				self.ue[i] = estimulo[i];

			for j in range(len(self.uo)):
				suma = 0.0;
				for i in range(len(self.ue)):
					suma += self.wim[i,j]*self.ue[i];
				self.uo[j] = self.funact(suma);
		
			for j in range(len(self.us)):
				suma = 0.0;
				for i in range(len(self.uo)):
					suma += self.wmo[i,j]*self.uo[i];
				self.us[j] = self.funact(suma);
				

			return self.us[:]; #se regresa una copia de las unidades de salida

