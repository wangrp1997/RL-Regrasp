# -*- coding: utf-8 -*-

import autograd.numpy as np
from autograd import jacobian


# 定义一个多变量向量函数
def vector_function(variables):
    x, y, z = variables
    return np.array([x ** 2 + y ** 3, np.sin(z), x * y])


# 计算向量函数的雅可比矩阵
jacobian_matrix = jacobian(vector_function)

# 输入变量的初始值
initial_variables = np.array([2.0, 3.0, 1.0])

# 梯度下降参数
learning_rate = 0.001
num_iterations = 1000

# 梯度下降优化
variables = initial_variables.copy()

for iteration in range(num_iterations):
    # 计算雅可比矩阵
    jacobian_result = jacobian_matrix(variables)

    # 计算目标函数的值
    objective_values = vector_function(variables)

    # 计算梯度
    gradient = np.dot(jacobian_result.T, objective_values)

    # 更新变量
    variables -= learning_rate * gradient

print("Optimized variables:", variables)


