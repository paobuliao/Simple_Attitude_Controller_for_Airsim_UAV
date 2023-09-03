import numpy as np
class Airsim_quadrotor():
    def __init__(self):
        self.J = np.array(
            [[0.006721, 0, 0], [0, 0.00804, 0], [0, 0, 0.014279]])
        self.Ixx = 0.006721
        self.Iyy = 0.00804
        self.Izz = 0.014279
        # Mass and Gravity
        self.m, self.g = 1, 9.82
        self.d = 0.2275

    def PD_Att_Controller(self, ang_now, ang_vel_now, ang_des):
        phi = float(ang_now[0])
        theta = float(ang_now[1])
        psi = float(ang_now[2])

        # PID gains unlimited
        Kp = np.array([[12, 0, 0],
                       [0, 12, 0],
                       [0, 0, 1]])*3
        Kd = np.array([[5, 0, 0],
                       [0, 5, 0],
                       [0, 0, .01]])*1

        angle_error = ang_des - ang_now
        # print('Erro angulo:', angle_error.T)
        ang_vel_error = np.zeros((3, 1)) - ang_vel_now
        # Compute Optimal Control Law

        T = np.array([[1/self.Ixx, np.sin(phi)*np.tan(theta)/self.Iyy, np.cos(phi)*np.tan(theta)/self.Izz],
                      [0, np.cos(phi)/self.Iyy, -np.sin(phi)/self.Izz],
                      [0, np.sin(phi)*np.cos(theta)/self.Iyy, np.cos(phi)*np.cos(theta)/self.Izz]])

        u = np.linalg.inv(T)@(Kp@angle_error + Kd@ang_vel_error)  # 求出期望的角速度
        tau = self.P_Angular_Vel_Controller(ang_vel_now, u)  # 用我的P期望角速度控制器求出力矩

        return tau

    def P_Angular_Vel_Controller(self, ang_vel_now, ang_vel_des):
        # PID gains unlimited
        Kp = np.array([[2, 0, 0],
                       [0, 2, 0],
                       [0, 0, 2]])*1
        Kd = np.array([[0.5, 0, 0],
                       [0, 0.5, 0],
                       [0, 0, 0.5]])*1

        ang_vel_error = ang_vel_des - ang_vel_now
        # Compute Optimal Control Law
        u = Kp@ang_vel_error
        tau = self.J@u+np.cross(ang_vel_now.transpose(),
                                (self.J@ang_vel_now).T).T
        return tau.reshape(3, 1)

    # 我现在有U, faid, thetad, psid
    # 转化为各旋翼呃转速
        # 力和力矩到电机控制的转换
    def fM2u(self, f, M):
        mat = np.array([[4.179446268, 4.179446268, 4.179446268, 4.179446268],
                        [-0.6723341164784, 0.6723341164784,
                            0.6723341164784, -0.6723341164784],
                        [0.6723341164784, -0.6723341164784,
                            0.6723341164784, -0.6723341164784],
                        [0.055562, 0.055562, -0.055562, -0.055562]])
        fM = np.vstack([f, M])
        u = np.dot(np.linalg.inv(mat), fM)
        u1 = u[0, 0]
        u2 = u[1, 0]
        u3 = u[2, 0]
        u4 = u[3, 0]
        return u1, u2, u3, u4
