"""
Copyright (c) 2016 by MTech of Univ. of Michigan, Ann Arbor.
All rights reserved. Not to be used for commercial purposes.

This file provides the implementation of linear, lasso, and ridge regression methods
using the *linear_model* module from the *sklearn* package.
The methods are applied to a polynomial curve fitting problem of a 1D dataset.

Author: Daning Huang
Date: 10/02/2016
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def regs(odrs, model):
    """
    Solve regression problems using a series of polynomials of different orders *odrs*.
    The parameter *model* specifies the type of regression method.
    """
    s = []
    f = plt.figure()
    for idx in range(len(odrs)):
        odr = odrs[idx]
        X   = np.column_stack([x**i for i in range(odr+1)])
        model.fit(X, y)
        s.append(model.coef_)
    
        plt.subplot(2, 2, idx+1)
        plt.tight_layout()
        plt.plot(x, y, 'bo', fillstyle='none')
        plt.plot(x, np.sin(x), 'b--', linewidth=2)
        plt.plot(x, model.predict(X), 'r-', linewidth=2)
        plt.grid()
        plt.title('Order = {0:d}'.format(odr))
    return f, s

if __name__ == "__main__":
    # Generate sample dataset
    np.random.seed(10)
    x = np.arange(60, 300, 4) * np.pi/180.0
    y = np.sin(x) + np.random.normal(0, 0.15, len(x))

    # The polynomial orders to consider
    odrs = [6, 9, 12, 15]

    # Various regression methods
    f1, s1 = regs(odrs, linear_model.LinearRegression(normalize=True))
    f2, s2 = regs(odrs, linear_model.Lasso(alpha=0.0001, normalize=True, max_iter=20000))
    f3, s3 = regs(odrs, linear_model.Ridge(alpha=0.0001, normalize=True))

    # Comparison of regression coefficients
    # This figure illustrates the effect of lasso regression more clearly than the scipy results.
    # Linear regression leads to very large coefficients in higher-order terms.
    # Ridge regression drives down the coefficients of higher-order terms.
    # Lasso regression not only reduces the coefficients, but also forces some of the coefficients to be zero. That means some of the polynomial terms are even not necessary! Therefore, lasso provides an automatic way to select the most representative terms for the dataset.
    plt.figure()
    plt.semilogy(np.abs(s1[3]), 'bo', label='Linear')
    plt.semilogy(np.abs(s2[3]), 'rs', label='Lasso')
    plt.semilogy(np.abs(s3[3]), 'gv', label='Ridge')
    plt.grid()
    plt.xlabel('Order')
    plt.ylabel('Coefficients')
    plt.legend()
    
    plt.show()
