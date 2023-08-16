from gplearn.genetic import SymbolicRegressor


def symbolic_regression(df, target, features=None):
    y = df[target] # 
    if features is None:
        x = df.drop(columns=[target])
    else:
        x = df[features] #Inputs

    est = SymbolicRegressor(
        population_size=5000,  # n
        generations=50,  # generational evolution
        stopping_criteria=0.01,  # evolutionstop
        p_crossover=0.7,  # backchecking
        p_subtree_mutation=0.1,  # mutation after winner
        p_hoist_mutation=0.05,  # Subtree of subtree mutation against winner, Avoid bloating
        p_point_mutation=0.1,  # Random node change on winner
        max_samples=0.9,  # percentage of samples to draw
        verbose=1,  # text
        parsimony_coefficient=0.01,  # bloat control to avoid large programs
        function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan'),  # functions to try
        random_state=8  # seed
    )
    est.fit(x, y)

    return str(est._program)


def format_expression(expr):
    replacements = {
        'add': '+',
        'sub': '-',
        'mul': '×',
        'div': '÷',
        'sqrt': '√',
        # add more
    }

    for key, value in replacements.items():
        expr = expr.replace(key, value)

    return expr


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import random

    x = np.arange(1, 50)
    y = np.empty(len(x))
    for i in range(0, len(x)):
        y[i] = 1 / (np.cos(i) + 1)
    df = pd.DataFrame({'x': x, 'y': y})
    res = symbolic_regression(df, target='y')
