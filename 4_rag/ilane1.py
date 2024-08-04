def fibonacci(n):
    a, b = 0, 1
    fib = []
    while a<n:
        fib.append(a)
        a, b = b, a+b

    return fib
#print(fibonacci(3))










def classer(classeur, nombre):
    if nombre > 0:
        classeur["positif"].append(nombre)
    else:
        classeur["négatif"].append(nombre)

    return classeur
classeur = {"positif": [],
            "négatif": []}
#print(classer(classeur, 9))
