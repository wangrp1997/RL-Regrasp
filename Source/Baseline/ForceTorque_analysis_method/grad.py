# -*- coding: utf-8 -*-

import autograd.numpy as np
from autograd import grad, jacobian, elementwise_grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def CoM_Estimate(joint_axes,T_ee2joint, end_effector_to_joints,
                           joint_torques_before, joint_torques_after):

    # 重新定义目标函数
    def objective_function(variables, T_ee2joint, joint_axes, end_effector_to_joints,
                           joint_torques_before, joint_torques_after):
        dx, dy, Go_z = variables
        Go = np.array([0, 0, -Go_z, 1])

        objective = 0
        for i, axis in enumerate(joint_axes):
            dr = end_effector_to_joints[i] + np.array([dx, dy, 0])
            Go = T_ee2joint[i] @ Go.reshape((4, 1))
            Torque_Go = np.cross(dr, Go[:3,0])
            torque_projection = np.dot(Torque_Go, axis) * axis
            # np.linalg.norm
            objective += (torque_projection + joint_torques_after[i] - joint_torques_before[i])**2
        return np.linalg.norm(objective)


    def com_estimate(objective_function,T_ee2joint, joint_axes, end_effector_to_joints,
                    joint_torques_before, joint_torques_after):
        # 使用Autograd计算带有额外参数的梯度
        elementwise_gradient_function = grad(objective_function)  # 指定参数位置为0

        # 初始猜测的值
        # initial_guess = np.array([np.random.rand() * 0.01, np.random.rand() * 0.01,
        #                 np.random.rand() * 0.01])
        # print("初始值：",initial_guess)
        initial_guess = np.array([0.0,0.1,10.])
        # 梯度下降参数
        learning_rate = 0.0001
        num_iterations = 3000

        # 梯度下降优化
        variables = initial_guess.copy()
        objective_values = []

        for iteration in range(num_iterations):
            # jacobian_result = gradient_function(variables,T_ee2joint, joint_axes, end_effector_to_joints,
            #                               joint_torques_before, joint_torques_after)
            elementwise_gradient = elementwise_gradient_function(variables,T_ee2joint, joint_axes, end_effector_to_joints,
                                          joint_torques_before, joint_torques_after)

            # 计算目标函数的值
            # objective_value = objective_function(variables, T_ee2joint, joint_axes, end_effector_to_joints,
            #                                       joint_torques_before, joint_torques_after)
            #
            # # 计算梯度
            # gradient = np.dot(jacobian_result.T, objective_value)

            variables -= learning_rate * elementwise_gradient

            # 计算并保存优化后的目标函数值
            optimized_value = objective_function(variables, T_ee2joint, joint_axes, end_effector_to_joints,
                                                  joint_torques_before, joint_torques_after)
            objective_values.append(optimized_value)

            print(f"Iteration {iteration + 1}, Optimized Value: {optimized_value}")

        # 输出最终优化结果
        print("Optimized variables:", variables)

        # 绘制目标函数值曲线
        plt.plot(range(num_iterations), objective_values, linewidth=5)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Objective Value Convergence')
        plt.grid(True)
        # 显示图形并暂停5秒
        plt.show(block=False)  # 使用 block=False 避免阻塞
        plt.pause(5)  # 暂停5秒

        # 关闭图形
        plt.close()
        return variables
    ret = com_estimate(objective_function, T_ee2joint, joint_axes, end_effector_to_joints,
                    joint_torques_before, joint_torques_after)

    return ret


# tcp_origin = np.array([0.1, 0.2, 0.3])
#
# # 定义关节坐标系原点在基坐标系中的坐标（示例）
# joint_origins = [
#     np.array([0, 0, 0]),    # 第一个关节坐标系原点在基坐标系中的坐标
#     np.array([0.0, 0.0, 0.089159]),  # 第二个关节坐标系原点在基坐标系中的坐标
#     np.array([0.425, 0, 0.10915]),  # 第三个关节坐标系原点在基坐标系中的坐标
#     np.array([0.39225, -0.093, 0.09465]),  # 第四个关节坐标系原点在基坐标系中的坐标
#     np.array([0.0, 0, 0.0823]),  # 第五个关节坐标系原点在基坐标系中的坐标
#     np.array([0.09465, 0, 0])  # 第六个关节坐标系（末端坐标系）原点在基坐标系中的坐标
# ]
#
# # 计算末端执行器中心到各个关节坐标系的位移向量
# end_effector_to_joints = []
#
# for joint_origin in joint_origins:
#     displacement_vector = tcp_origin - joint_origin
#     end_effector_to_joints.append(displacement_vector)  # dre-ii
# # print(end_effector_to_joints)
# # 定义关节旋转轴（示例，根据实际情况进行调整）
# joint_axes = [
#     np.array([0, 0, 1]),  # 第一个关节绕Z轴旋转
#     np.array([0, 1, 0]),  # 第二个关节绕Y轴旋转
#     np.array([0, 1, 0]),  # 第三个关节绕Y轴旋转
#     np.array([0, 1, 0]),  # 第四个关节绕Y轴旋转
#     np.array([0, 0, 1]),  # 第五个关节绕Z轴旋转
#     np.array([0, 1, 0])   # 第六个关节绕Y轴旋转
# ]
#
# # 生成随机的力矩大小
# torque_magnitudes = np.random.uniform(0.1, 10.0, len(joint_axes))
# torque_magnitudes1 = np.random.uniform(10.0, 15.0, len(joint_axes))
#
# # 生成与关节旋转轴对应的力矩向量
# joint_torques_before = []
# joint_torques_after = []
#
# for i, axis in enumerate(joint_axes):
#     torque_magnitude = torque_magnitudes[i]
#     torque_magnitude1 = torque_magnitudes1[i]
#
#     torque_vector = torque_magnitude * axis
#     torque_vector1 = torque_magnitude1 * axis
#
#     joint_torques_before.append(torque_vector)  # Tibeforegrasp
#     joint_torques_after.append(torque_vector1)  # Tiaftergrasp
#
# print(joint_axes)
# print(end_effector_to_joints)
# print(joint_torques_before)
# print(joint_torques_after)
# joint_axes = [np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0, 0, 1])]
# joint_axes = [
#     np.array([0, 0, 1]),  # 第一个关节绕Z轴旋转
#     np.array([0, 1, 0]),  # 第二个关节绕Y轴旋转
#     np.array([0, 1, 0]),  # 第三个关节绕Y轴旋转
#     np.array([0, 1, 0]),  # 第四个关节绕Y轴旋转
#     np.array([0, 0, 1]),  # 第五个关节绕Z轴旋转
#     np.array([0, 1, 0])   # 第六个关节绕Y轴旋转
# ]
# end_effector_to_joints = [np.array([0.69118251, 0.20081234, 0.23024074]), np.array([-0.14615369,  0.43799809,  0.23024074]), np.array([-0.06390232, -0.07161205,  0.23024074]), np.array([0.03779167, 0.00109074, 0.20822462]), np.array([0.00108604, 0.00642538, 0.0822082 ]), np.array([-6.17577484e-03,  2.07963160e-03, -9.18040241e-05])]
# joint_torques_before = [np.array([0., 0., 0.]), np.array([  0.        ,   0.        , -23.48294014]), np.array([  0.       ,   0.       , -14.3903755]), np.array([ 0.00000000e+00,  0.00000000e+00, -5.55111512e-15]), np.array([ 0.00000000e+00,  0.00000000e+00, -2.85687272e-17]), np.array([0.00000000e+00, 0.00000000e+00, 1.58190136e-16])]
# joint_torques_after = [np.array([0.0000000e+00, 0.0000000e+00, 4.4408921e-16]), np.array([  0.        ,   0.        , -24.23248756]), np.array([  0.        ,   0.        , -15.42885148]), np.array([0.00000000e+00, 0.00000000e+00, 6.43929354e-15]), np.array([0.00000000e+00, 0.00000000e+00, 4.06575815e-18]), np.array([ 0.00000000e+00,  0.00000000e+00, -4.04800697e-16])]
# r = CoM_Estimate(joint_axes, end_effector_to_joints, joint_torques_before, joint_torques_after)