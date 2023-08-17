from gplearn.genetic import SymbolicRegressor

est = SymbolicRegressor(
    population_size=5000,  # n
    generations=50,  # generational evolution
    stopping_criteria=0.0,  # evolutionstop
    p_crossover=0.7,  # backchecking
    p_subtree_mutation=0.1,  # mutation after winner
    p_hoist_mutation=0.1,  # Subtree of subtree mutation against winner, Avoid bloating
    p_point_mutation=0.1,  # Random node change on winner
    max_samples=0.8,  # percentage of samples to draw
    verbose=1,  # text
    parsimony_coefficient=0.01,  # bloat control to avoid large programs
    function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', 'abs'),
    # functions to try
    random_state=8,  # seed
    init_method='half and half'
)


def symbolic_regression(data, target, features=None):
    y = data[target]  # Outputs
    if features is None:
        x = data.drop(columns=[target])
    else:
        x = data[features]  # Inputs

    est.fit(x, y)

    return str(est._program)


def format_expression(expr):
    binary_operations = {
        'add': '+',
        'sub': '-',
        'mul': '×',
        'div': '÷'
    }
    unary_operations = {
        'sqrt': '√',
        'log': 'log',
        'inv': '1/',
        'sin': 'sin',
        'cos': 'cos',
        'tan': 'tan',
        'neg': 'neg',
        'abs': 'abs'
    }
    for unary_op, symbol in unary_operations.items():
        if expr.startswith(unary_op):
            inner_expr = extract_inner(expr[len(unary_op):])
            return f"{symbol}({format_expression(inner_expr)})"

    for binary_op, symbol in binary_operations.items():
        if expr.startswith(binary_op):
            inner_expr = extract_inner(expr[len(binary_op):])
            left, right = split_binary(inner_expr)
            return f"({format_expression(left)} {symbol} {format_expression(right)})"

    return expr


def extract_inner(expr):
    assert expr[0] == '(' and expr[-1] == ')', "Expression format is unexpected!"
    return expr[1:-1]


def split_binary(expr):
    depth = 0
    for i, char in enumerate(expr):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            return expr[:i], expr[i + 2:]


# Doesn't handle infinite or convergence of values to 0 or divergent functions or imaginary numbers, can produce
# functions that diverge, especially when x is an arange instead of linspace, might be problematic
# for whole number operations, doesn't handle exponents or power expressions well.
# Fails Rydberg formula because slope is too small, barely different from 0, results in value of 0.001
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import random

    x = np.linspace(0, 100).reshape(-1,1)
    #    y=np.empty(shape=len(x))
    #    for i in range(1,len(x)):
    #        if x[i]%2:
    #            y[i]= 1
    #        else:
    #            y[i]= 0
    y = 1/(10973731.6*(1/(x*x)-1/((x*x)+1)))

    df = pd.DataFrame({'x': x.ravel(), 'y': y.ravel()})
    res = symbolic_regression(df, target='y')
    print(res)
    print(format_expression(res))
