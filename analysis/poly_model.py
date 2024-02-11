import numpy as np
import matplotlib.pyplot as plt


def poly_fit(x_values, y_values, degree=60):
    # 使用 numpy 的 polyfit 函数进行多项式拟合
    coefficients = np.polyfit(x_values, y_values, degree)

    # 生成拟合的多项式函数
    fitted_curve = np.poly1d(coefficients)

    return fitted_curve(x_values)
