from pysr import PySRRegressor

estpySR = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "-", "*", "/", '^'],
    unary_operators=["sin", 'cos', "exp", "log", "abs", 'sqrt', 'square', 'cube', 'exp', 'abs', 'sinh'],
    progress=True,
    populations=12,
    precision=64,
    extra_sympy_mappings={'inv': lambda x: 1 / x},
    constraints={'^': (9, 1)},
    nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}},
    weight_optimize=0.001,
    maxsize=30
)


def symbolic_regression(data, target, features=None):
    y = data[target]  # Outputs
    if features is None:
        x = data.drop(columns=[target])
    else:
        x = data[features]  # Inputs

    estpySR.fit(x, y)

    return estpySR


# Doesnt handle disjoint data very well
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import random

    x = np.linspace(0, 100).reshape(-1, 1)
    #    y=np.empty(shape=len(x))
    #    for i in range(1,len(x)):
    #        if x[i]%2:
    #            y[i]= 1
    #        else:
    #            y[i]= 0
    y = 1 / np.sin(x + 1)

    df = pd.DataFrame({'x': x.ravel(), 'y': y.ravel()})
    result = symbolic_regression(df, target='y')
    print(result.equations_)
