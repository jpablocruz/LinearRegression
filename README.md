# LinearRegression
Este es un modelo de regresion lineal el cual sin el uso de ningun framework es capaz de generar predicciones correctamente
con base en un dataset el cual se le otorge.

Este algoritmo funciona con base en varias funciones las cuales se encargan de calcular diferentes valores que se necesitan para generar los coeficientes 
de una regresion lineal.

Entre estas funciones encontramos:
- varianza()
- covarianza()
- media()
- coef() [se encarga de generar los coeficientes de la funcion y, con base en las demas funciones señaladas]
- error_rmse() calcula el Root Mean Square Error para tener una metrica de precicion del modelo.
- r_sqrd() calcula la R cuadrada del modelo generado. 
- train_test_muestra() se encarga de separar los datos en un conjunto de prueba y otro de entrenamiento
  para poder hacer que aprenda nuestro modelo, en este caso optamos por dividir la muestra un 60 a 40 porciento.
- regresionLineal_smpl() se encarga de generar los datos estimados y los guarda en una lista la cual devuelve para que estos
  sean comparados con los datos reales en la formula del error rmse.
  
  Lo que el programa despliega es una serie de informacion relacionada al modelo. 
  
  Empezamos con los coeficientes que genera, pasando por la funcion con los coeficientes generados y finalmente las metricas 
  que demuestran si el modelo se ajusta correctamente a los datos. Mostrandonos la R cuadrada y la RMSE. 
  
  Conclusiones
  En este caso podemos observar que las metricas del modelo dan valores que nos dictan un modelo que se ajusta correctamente a los datos,
  entendemos con esto que las predicciones que genera con base en la muesta de prueba con la que entrena, es satisfactoria y genera predicciones acertadas.
  
  Esto lo podemos comprobar porque se obtiene una R cuadrada muy cercana al valor de 1 y un RMSE muy cercano a 0.
