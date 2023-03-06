import numpy as np

# Pide al usuario el número de restricciones y variables
num_restricciones = int(input("Ingrese el número de restricciones: "))
num_variables = int(input("Ingrese el número de variables: "))

# Crea la matriz de restricciones y el vector de coeficientes de la función objetivo
A = np.zeros((num_restricciones, num_variables))
b = np.zeros((num_restricciones))
c = np.zeros((num_variables))

# Pide al usuario los valores de cada restricción y coeficiente de la función objetivo
for i in range(num_restricciones):
    print(f"Ingrese los coeficientes de la restricción {i+1}:")
    for j in range(num_variables):
        A[i][j] = float(input(f"a{i+1}{j+1}: "))
    b[i] = float(input(f"Ingrese el valor de b{i+1}: "))
    
print("Ingrese los coeficientes de la función objetivo:")
for i in range(num_variables):
    c[i] = float(input(f"c{i+1}: "))

# Inicializa la solución
x = np.zeros((num_variables))

# Implementa el método simplex
while True:
    # Calcula la función objetivo
    z = np.dot(c, x)
    
    # Calcula las variables de holgura
    A_holgura = np.concatenate((A, np.identity(num_restricciones)), axis=1)
    c_holgura = np.concatenate((np.zeros((num_variables)), np.ones((num_restricciones))))
    
    # Encuentra la columna pivote
    pivot_column = np.argmax(c_holgura)
    
    # Si no hay columnas positivas, termina el algoritmo
    if c_holgura[pivot_column] <= 0:
        print("La solución es óptima")
        break
    
    # Encuentra la fila pivote
    ratios = np.divide(b, A_holgura[:, pivot_column])
    pivot_row = np.argmin(ratios)
    
    # Realiza el intercambio de fila y columna pivote
    A_holgura[pivot_row, :], b[pivot_row] = A_holgura[pivot_row, :] / A_holgura[pivot_row, pivot_column], b[pivot_row] / A_holgura[pivot_row, pivot_column]
    for i in range(num_restricciones):
        if i != pivot_row:
            ratio = A_holgura[i, pivot_column] / A_holgura[pivot_row, pivot_column]
            A_holgura[i, :] -= ratio * A_holgura[pivot_row, :]
            b[i] -= ratio * b[pivot_row]
    c_holgura -= c_holgura[pivot_column] * A_holgura[pivot_row, :]
    
    # Calcula la solución
    x = np.zeros((num_variables))
    for i in range(num_restricciones):
        if np.allclose(A_holgura[i, 0:num_variables], np.zeros((num_variables))):
            index = np.argmax(A_holgura[i, num_variables+1:])
            x[index] = b[i]
    
# Imprime la solución
print("La solución es:")
print(x)
print(f"El valor óptimo de la función objetivo es: {z}")