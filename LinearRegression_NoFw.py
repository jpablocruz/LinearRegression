#Modelo de Regresion Lineal desde cero sin el uso de un "Framework"
#José Pablo Cruz Ramos - A01138740
from cmath import sqrt
from csv import reader
from random import randrange
import pandas as pd
import numpy as np

#Para crear la regresion lineal simple, necesitamos los datos de la varianza, la media y la covarianza.
#Debido a que no usamos librerias debemos crear funciones que calculen dichos datos.
def regresionLineal_smpl(train_set, set_prueba):
    datos_pred = list()
    beta0, beta1 = coef(train_set)
    print("Coeficientes:  b1: " + str(beta0) + " b1: " + str(beta1))
    print("")
    print( "Formula de regresion lineal:")
    print( "y = " + str(beta0) + " + " + str(beta1) + "x")
    print("")
    for i in set_prueba:
        estimado = beta0 +  beta1 * i[0]
        datos_pred.append(estimado)
    return datos_pred

def media(datos):
    return sum(datos) / float(len(datos))

def varianza(datos, media):
    return sum([(x - media) ** 2 for x in datos])

def varianza2(datos, media):
    sum = 0
    for x in datos:
        sum = sum +  ((x - media)** 2)
    return sum

def covarianza(datosX, mediaX, datosY, mediaY):
    covar = 0.0
    for i in range(len(datosX)):
        covar = covar + (datosX[i] - mediaX) * (datosY[i] - mediaY)
    return covar


def coef(datos):
    #cargamos los datos correctamente
    x = [row[0] for row in datos]
    y = [row[1] for row in datos]
    media_x, media_y = media(x), media(y)
    print(" ")
    var_x, var_y = varianza2(x, media_x), varianza2(y, media_y)
    covarianza_xy = covarianza(x, media_x, y, media_y)
    beta1 = covarianza(x, media_x, y, media_y) / varianza2(x, media_x)
    beta0 = media_y - beta1 * media_x
    return [beta0, beta1]

def error_rmse(real, estimado):
	errores = 0.0
	for i in range(len(real)):

		distancia_real_est = estimado[i] - real[i]
		errores = (distancia_real_est ** 2) + errores
	media_error = errores / float(len(real))
	return sqrt(media_error)

def train_test_muestra(datos, porcentaje):
    train_n = porcentaje * len(datos)
    train_set = list()
    auxDatos = list(datos)
    while len(train_set) < train_n:
        i = randrange(len(auxDatos)) #valor al azar a agarrar de la muestra
        train_set.append(auxDatos.pop(i))
    return train_set, auxDatos

def evaluate_algorithm(dataset, porcentaje): ########
	train, test = train_test_muestra(dataset, porcentaje)
	set_prueba = list()
	for datos_aux in test:
		datos_aux = list(datos_aux)
		datos_aux[-1] = None
		set_prueba.append(datos_aux)
	estimado = regresionLineal_smpl(train, set_prueba)
	real = [x[-1] for x in test]
	error = error_rmse(real, estimado)
	return error

def columna_a_numero(datos, columna):
	for x in datos:
		x[columna] = float(x[columna].strip())

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv = reader(file)
		for row in csv:
			if not row: continue
			dataset.append(row)
	return dataset

def r_sqrd(beta0,beta1,dataset):
    y = [row[1] for row in dataset]
    x = [row[0] for row in dataset]
    n = len(y)
    SsRes = 0
    Sstot = 0
    for i in range(n):
        y_est = beta0 + x[i] * beta1
        SsRes = SsRes + (y_est - y[i])**2
        Sstot = Sstot + (y[i] - media(y)) ** 2
    r_sq = 1 - SsRes/Sstot
    return r_sq
##main

datos = load_csv("Salary_Data.csv") #cargamos los datos y los transformamos de string a float
for i in range(len(datos[1])):
    columna_a_numero(datos, i)

porcentaje = 0.6 #en este caso dividiremos nuestros datos en muestras, una para el train y otra para el test, lo dividiremos con un 60% a 40%
print("- Analisis del modelo de regresion lineal -")
error_rmse = evaluate_algorithm(datos, porcentaje)
x = [row[0] for row in datos]
y = [row[1] for row in datos]
print("Root Mean Square Error: "+ str(error_rmse.real))
print("Media de la variable 'y' para comparar con rmse : " + str(media(y)))
rmse_norm = error_rmse / (max(x) - min(y)) #Calculamos el error rmse normalizado el cual se obtiene dividiendo el rmse entre la resta entre el valor maximo por el valor minimo
print("Normalizamos el RMSE y obtenemos un valor entre 0 a 1 de: " + str(abs(rmse_norm.real)) )
b0,b1 = coef(datos)
rcuadrada = r_sqrd(b0,b1,datos)
print("Valor de R cuadrada: " + str(rcuadrada))
print("")