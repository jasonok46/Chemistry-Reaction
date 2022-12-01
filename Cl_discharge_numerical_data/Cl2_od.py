import numpy as np
import math
from sympy import *
import time



# 저장해야할 txt 내용 
# 1. P_abs
# 2. pressue 
# 3. electron temperature  (Te)
# 4. densities (n)
# 5. Q_in 
# 6. Radius, length 
# 7. gas temerature 

class Explicit_method:
    def __init__(self, pressure, power, Radius, Length, Tp, Tn, Qin, gamma_rec, num_species = 9, num_reactions=41, dt = 1e-8):
        self.pi = 3.14159265
        self.amu = 1.66054e-27 # [kg]
        self.m_e = 9.1094e-31 # [kg]
        self.EC = 1.6022e-19 #[C], electron charge 
        self.EV = 1.6022e-19 #[J]
        self.dt = dt
        self.ZERO = 1e-20
        self.pressure = pressure
        self.power = power
        self.R = Radius 
        self.L = Length
        self.Tp = Tp
        self.Tn = Tn
        self.Qin = Qin
        self.gamma_rec =gamma_rec 
        self.Vol = self.pi * self.R**2 * self.L
        self.Area = 2*self.pi*(self.R**2 + self.R * self.L)
        
        self.S_pump = 0.014*self.Qin/self.pressure
        self.n_sp = num_species
        self.n_react = num_reactions
        self.Tg = self.calc_Tg(self.pressure, self.power)
        self.eta = 2*self.Tp/(self.Tp + self.Tn)

    def mass(self):
        m = np.zeros(self.n_sp)
        m[0] = 2*12.9676*self.amu
        m[1] = 2*12.9676*self.amu
        m[2] = 2*12.9676*self.amu
        m[3] = 2*12.9676*self.amu
        m[4] = 12.9676*self.amu
        m[5] = 2*12.9676*self.amu
        m[6] = 12.9676*self.amu
        m[7] = 12.9676*self.amu
        m[8] = self.m_e
        return m

    def init_dense(self):
        init_den = np.ones(self.n_sp)*1e16
        N0 = 3.2e19*self.pressure
        init_den[0] = N0
        init_den[1] = 1.e16
        init_den[2] = 1.e16
        init_den[3] = 1.e16
        init_den[4] = 1.e16
        init_den[5] = 1.e16
        init_den[6] = 1.e16
        init_den[7] = 1.e16
        init_den[8] = 1.e16
        return init_den

    def init_Te(self):
        init_Te = 3.0
        return init_Te

    def rc(self,A,B,C,T):
        return A*np.power(T,B)*np.exp(-C/T)
    
    def calc_Tg(self,pressure,power):
        sp = 1250.*(1.-np.exp(-0.091*pressure))+400.*np.exp(-0.337*pressure)
        Tg = (300. + sp * np.log(power/40.)/np.log(40.))/11604.
        return Tg
        
    def calc_rate_coefficient(self, Te,Tg):
        k = np.zeros(self.n_react)
        k[0] = self.rc(6.67e-14, -0.1, 8.67, Te)
        k[1] = self.rc(4.87e-14, 0.50, 12.17,Te)
        k[2] = self.rc(1.79e-13, 0., 24.88, Te)
        k[3] = self.rc(1.46e-16, 2.16, 21.42, Te)
        k[4] = self.rc(22.5e-16, -0.46, 2.82, Te) - self.rc(12.1e-16, 0., 0.99, Te) + 6.54e-16
        k[5] = self.rc(9.29e-15, -0.47, 2.83, Te) - self.rc(4.96e-15, 0., 0.99, Te) + 2.7e-15
        k[6] = self.rc(20.1e-15, -0.47, 2.83, Te) - self.rc(10.8e-15, 0., 0.97, Te) + 5.92e-15
        k[7] = self.rc(30.5e-15, -0.46, 2.82, Te) - self.rc(16.3e-15, 0., 0.99, Te) + 8.81e-15
        k[8] = self.rc(3.45e-16, 0.13, 19.7, Te)
        k[9] = self.rc(4.35e-16, -1.48, 0.79, Te)
        k[10] = self.rc(8.1e-17, -1.48, 0.68, Te)
        k[11] = self.rc(2.39e-17, -1.49, 0.64, Te)
        k[12] = self.rc(1.04e-15, -1.48, 0.73, Te)
        k[13] = self.rc(2.98e-16, -1.48, 0.67, Te)
        k[14] = self.rc(1.04e-15, -1.48, 0.73, Te)
        k[15] = self.rc(2.48e-14, 0.62, 12.76, Te)
        k[16] = self.rc(2.33e-15, 1.45, 2.48, Te)
        k[17] = self.rc(3.38e-15, 0.75, 25.28, Te)
        k[18] = self.rc(9.e-14, -0.5, 0., Te)
        k[19] = 5.e-14*np.power(300./Tg, 0.5)
        k[20] = 5.e-14*np.power(300./Tg, 0.5)
        k[21] = 5.4e-16
        k[22] = 3.5e-45*np.exp(810./Tg)
        k[23] = 8.75e-46*np.exp(810./Tg)
        k[24] = self.rc(1.1699e-13, 0.5085, 0.9266,Te)
        k[25] = self.rc(5.0375e-15, 0.3552, 4.1248, Te)
        k[26] = self.rc(6.1677e-15, 0.1158, 5.0214,Te)
        k[27] = self.rc(1.8779e-15, 0.5693, 5.6228, Te)
        k[28] = self.rc(1.5811e-15, 0.3792, 7.7542, Te)
        k[29] = self.rc(3.8130e-15, 0.2864, 6.4455, Te)
        k[30] = self.rc(9.5885e-15, 0.0225, 11.9027, Te)
        k[31] = self.rc(2.5668e-15, 0.0258, 12.4123, Te)
        k[32] = self.rc(8.5572e-14, 0.6575, 1.0712, Te)
        k[33] = self.rc(8.2546e-15, 0.2125, 9.9025, Te)
        k[34] = self.rc(2.0483e-14, -0.1959, 10.9701, Te)
        k[35] = self.rc(1.9755e-14, 0.1075, 11.4480,Te)
        k[36] = self.rc(1.2673e-15, 0.2059, 11.2638, Te)
        k[37] = self.rc(5.4283e-15, -0.1998, 12.4759, Te)
        k[38] = self.rc(1.0642e-14, 0.0898, 12.3539, Te)
        k[39] = self.rc(8.7963e-16, -0.0751, 16.7545, Te)
        k[40] = self.rc(6.7839e-15, 0.0206, 13.1525,Te)
        return k

    def calc_reaction_rate(self, den, Te):
        k = self.calc_rate_coefficient(Te, self.Tg*11604)
        R = np.zeros(self.n_react)
        R[0] = k[0]*den[8]*den[0]
        R[1] = k[1]*den[8]*den[0]
        R[2] = k[2]*den[8]*den[0]
        R[3] = k[3]*den[8]*den[0]
        R[4] = k[4]*den[8]*den[0]
        R[5] = k[5]*den[8]*den[1]
        R[6] = k[6]*den[8]*den[2]
        R[7] = k[7]*den[8]*den[3]
        R[8] = k[8]*den[8]*den[0]
        R[9] = k[9]*den[8]*den[0]
        R[10] = k[10]*den[8]*den[0]
        R[11] = k[11]*den[8]*den[0]
        R[12] = k[12]*den[8]*den[1]
        R[13] = k[13]*den[8]*den[1]
        R[14] = k[14]*den[8]*den[2]
        R[15] = k[15]*den[8]*den[4]
        R[16] = k[16]*den[8]*den[7]
        R[17] = k[17]*den[8]*den[7]
        R[18] = k[18]*den[8]*den[5]
        R[19] = k[19]*den[5]*den[7]
        R[20] = k[20]*den[6]*den[7]
        R[21] = k[21]*den[0]*den[6]
        R[22] = k[22]*den[4]**2*den[0]
        R[23] = k[23]*den[4]**2*den[4]
        R[24] = k[24]*den[8]*den[0]
        R[25] = k[25]*den[8]*den[0]
        R[26] = k[26]*den[8]*den[0]
        R[27] = k[27]*den[8]*den[0]
        R[28] = k[28]*den[8]*den[0]
        R[29] = k[29]*den[8]*den[0]
        R[30] = k[30]*den[8]*den[0]
        R[31] = k[31]*den[8]*den[0]
        R[32] = k[32]*den[8]*den[4]
        R[33] = k[33]*den[8]*den[4]
        R[34] = k[34]*den[8]*den[4]
        R[35] = k[35]*den[8]*den[4]
        R[36] = k[36]*den[8]*den[4]
        R[37] = k[37]*den[8]*den[4]
        R[38] = k[38]*den[8]*den[4]
        R[39] = k[39]*den[8]*den[4]
        R[40] = k[40]*den[8]*den[4]
        return R

    def funce(self, x, alpha_b, gamma):
        return np.exp((1. + x*alpha_b)*(1.-gamma)/(2.*(1. + x * alpha_b * gamma)))-x

    def calc_alpha_s(self, alpha_b, gamma):
        a = 0
        c = 10000
        iter = 0
        epsilon = 1e-8
        Y_a = self.funce(a,alpha_b, gamma)
        Y_c = self.funce(c,alpha_b, gamma)
        if(Y_a * Y_c >0):
            print('f(a) = {}\n'.format(Y_a))
            print('f(c) = {}\n'.format(Y_c))
        else:
            while True:
                iter = iter +1
                b = (a+c)*0.5
                Y_b = self.funce(b, alpha_b, gamma)
                if abs(b-a) < epsilon:
                    alpha_s= b* alpha_b
                    break 
                if Y_a * Y_b <=0.:
                    c=b
                    Y_c = Y_b
                else : 
                    a = b
                    Y_a = Y_b
        return alpha_s

    def cal_density(self, den, Te):
        m = self.mass()
        R = self.calc_reaction_rate(den, Te)

        alpha = den[7]/den[8]
        alpha_0  = 1.5 *alpha
        gamma = Te / self.Tn
        uB0_2i = np.sqrt(self.EV * Te/m[5]) 
        uB0_i = np.sqrt(self.EV * Te/m[6])
        lambda_2i = 1./(den[0]*200.e-20+ den[4]*150.e-20 + den[6]*75.e-20 + den[5]*100.e-20 + den[7]*75.e-20)
        lambda_i = 1./(den[0]*150.e-20 + den[4]*100.e-20 + den[6]*50.e-20 + den[5]*75.e-20 + den[7]*50.e-20)
        hLa_2i = 0.86 / (np.sqrt(3. + self.eta* self.L / (2. * lambda_2i)) * (1. + alpha_0))
        hRa_2i = 0.8 / (np.sqrt(4. + self.eta* self.R/lambda_2i) * (1. + alpha_0))
        hLa_i = 0.86 / (np.sqrt(3. + self.eta* self.L/(2*lambda_i)) * (1. + alpha_0))
        hRa_i = 0.8 / (np.sqrt(4. + self.eta* self.R/lambda_i)*(1. +alpha_0))
        v_2i = np.sqrt(8. * self.EC * self.Tp / (self.pi * m[5]))
        v_i  = np.sqrt(8. * self.EC * self.Tp / (self.pi * m[6]))
        Cl2i_star = 15. * self.eta*self.eta * v_2i/(56. * R[19] * lambda_2i + self.ZERO)
        Cli_star = 15. * self.eta* self.eta * v_i/(56. * R[20] * lambda_i + self.ZERO)
        hc_2i = 1./(np.sqrt(gamma) + np.sqrt(gamma)*(np.sqrt(Cl2i_star)*den[5]/np.power(den[7],1.5)))
        hc_i = 1./(np.sqrt(gamma) + np.sqrt(gamma)*(np.sqrt(Cli_star)*den[6]/np.power(den[7], 1.5)))
        hL_2i = np.sqrt(np.power(hLa_2i,2) + np.power(hc_2i,2))
        hR_2i = np.sqrt(np.power(hRa_2i,2) + np.power(hc_2i,2))
        hL_i = np.sqrt(np.power(hLa_i,2) + np.power(hc_i,2))
        hR_i = np.sqrt(np.power(hRa_i,2) + np.power(hc_i,2))
        Aeff_2i = 2*self.pi*(self.R*self.R*hL_2i + self.R*self.L*hR_2i)
        Aeff_i = 2*self.pi*(self.R*self.R*hL_i + self.R*self.L*hR_i)
        Lambda = np.power(np.power(self.pi/self.L,2) + np.power(2.405/self.R,2),-0.5)
        vthg = np.sqrt(8.*self.EV * self.Tg/(self.pi*m[4]))
        nu_m = (den[0]*75.e-20 + den[4]*50.e-20 + den[5]*150.e-20 + den[6]*100.e-20 + den[7]*100e-20)*vthg
        D_Cl = self.pi * vthg *vthg/(8*nu_m)
        k_Cl = 1./(np.power(Lambda,2)/D_Cl + 2.*self.Vol*(2. - self.gamma_rec)/(self.Area*vthg*self.gamma_rec))
        k_Cl2p = uB0_2i * Aeff_2i / self.Vol
        k_Clp = uB0_i * Aeff_i / self.Vol
        d_den = np.zeros(self.n_sp)
        d_den[0] = self.dt*(4.48e17*self.Qin/self.Vol + R[22] + R[23] - R[0] - R[1] - R[2] - R[3] - R[4] - R[8] - R[9] - R[10] - R[11] - R[21] - self.S_pump*den[0]/self.Vol + 0.5*k_Cl*den[4] + k_Cl2p*den[5])+den[0]
        d_den[1] = self.dt*(R[9] - R[5] - R[12] - R[13] - self.S_pump*den[1]/self.Vol)+den[1]
        d_den[2] = self.dt*(R[10] + R[12] - R[6] - R[14] - self.S_pump*den[2]/self.Vol)+den[2]
        d_den[3] = self.dt*(R[11] + R[13] + R[14] - R[7] - self.S_pump*den[3]/self.Vol)+den[3]
        d_den[4] = self.dt*(2*R[0] + R[2] + R[4] + R[5] + R[6] + R[7] + R[16] + 2.*R[18] + 3.*R[19] + 2*R[20] + R[21] - R[15] -2*R[22] - 2*R[23] - self.S_pump*den[4]/self.Vol - k_Cl*den[4] + k_Clp*den[6])+den[4]
        d_den[5] = self.dt*(R[1] + R[21] - R[18] - R[19] - self.S_pump*den[5]/self.Vol - k_Cl2p*den[5])+den[5]
        d_den[6] = self.dt*(R[2] + 2. *R[3] + R[8] + R[15] + R[17] - R[20] - R[21] - self.S_pump*den[6]/self.Vol - k_Clp*den[6])+den[6]
        d_den[7] = self.dt*(R[4] + R[5] + R[6] + R[7] + R[8] - R[16] - R[17] - R[19] - R[20])+den[7]
        d_den[8] = d_den[5] + d_den[6] - d_den[7]
        return d_den
    
    # def cal_energy(self,den, Te, write_txt = False):


    def cal_Te(self, den, Te, write_txt = False):
        R = self.calc_reaction_rate(den, Te)
        m = self.mass()
        alpha = den[7]/den[8]
        alpha_0  = 1.5 *alpha
        gamma = Te / self.Tn
        alpha_s = self.calc_alpha_s(alpha_b = alpha, gamma = gamma)
        Vp = 0.5 * (1.+alpha_s) * Te / (1.+gamma*alpha_s)
        d = np.exp(-(Vp/Te))
        uB_2i = np.sqrt(self.EV*Te*(1.+alpha_s) / (m[5]*(1.+alpha_s*gamma)))
        uB_i = np.sqrt(self.EV*Te*(1+alpha_s) / (m[6]*(1.+alpha_s*gamma)))
        uB0_2i = np.sqrt(self.EV * Te/m[5]) 
        uB0_i = np.sqrt(self.EV * Te/m[6])
        total_flux = (den[5]*uB_2i*d + den[6]*uB_i*d)*1.5
        vthi = np.sqrt(8*self.EC*Te/(self.pi*m[8]))
        Vs = Te*np.log((0.25*den[8]*vthi)/ (total_flux+self.ZERO))
        lambda_2i = 1./(den[0]*200.e-20+ den[4]*150.e-20 + den[6]*75.e-20 + den[5]*100.e-20 + den[7]*75.e-20)
        lambda_i = 1./(den[0]*150.e-20 + den[4]*100.e-20 + den[6]*50.e-20 + den[5]*75.e-20 + den[7]*50.e-20)
        hLa_2i = 0.86 / (np.sqrt(3. + self.eta* self.L / (2. * lambda_2i)) * (1. + alpha_0))
        hRa_2i = 0.8 / (np.sqrt(4. + self.eta* self.R/lambda_2i) * (1. + alpha_0))
        hLa_i = 0.86 / (np.sqrt(3. + self.eta* self.L/(2*lambda_i)) * (1. + alpha_0))
        hRa_i = 0.8 / (np.sqrt(4. + self.eta* self.R/lambda_i)*(1. +alpha_0))
        v_2i = np.sqrt(8. * self.EC * self.Tp / (self.pi * m[5]))
        v_i  = np.sqrt(8. * self.EC * self.Tp / (self.pi * m[6]))
        Cl2i_star = 15. * self.eta*self.eta * v_2i/(56. * R[19] * lambda_2i + self.ZERO)
        Cli_star = 15. * self.eta* self.eta * v_i/(56. * R[20] * lambda_i + self.ZERO)
        hc_2i = 1./(np.sqrt(gamma) + np.sqrt(gamma)*(np.sqrt(Cl2i_star)*den[5]/np.power(den[7],1.5)))
        hc_i = 1./(np.sqrt(gamma) + np.sqrt(gamma)*(np.sqrt(Cli_star)*den[6]/np.power(den[7], 1.5)))
        hL_2i = np.sqrt(np.power(hLa_2i,2) + np.power(hc_2i,2))
        hR_2i = np.sqrt(np.power(hRa_2i,2) + np.power(hc_2i,2))
        hL_i = np.sqrt(np.power(hLa_i,2) + np.power(hc_i,2))
        hR_i = np.sqrt(np.power(hRa_i,2) + np.power(hc_i,2))
        Aeff_2i = 2*self.pi*(self.R*self.R*hL_2i + self.R*self.L*hR_2i)
        Aeff_i = 2*self.pi*(self.R*self.R*hL_i + self.R*self.L*hR_i)
        source = 2. * self.power /(3.*self.EV*den[8]*self.Vol+self.ZERO)
        loss = 2.*(4.*R[0] + 11.48*R[1] + 15.5*R[2] + 29.92*R[3] + 0.39*R[4]
                    - 0.07*R[5] - 0.14*R[6] - 0.21*R[7] + 12.65*R[8] + 0.07*R[9]
                    + 0.14*R[10] + 0.21*R[11] + 0.07*R[12] + 0.13*R[13] +0.2*R[14]
                    + 12.96*R[15] + 3.61*R[16] + 16.57*R[17] - 7.48*R[18] 
                    + 3.24*R[25] + 4.04*R[26] + 6.23*R[27] + 6.86*R[28] + 6.8*R[29]
                    + 9.22*R[30] + 9.32*R[31] + 9.1*R[33] + 10.5*R[34] + 11.2*R[35]
                    + 11.4*R[36] + 11.8*R[37] + 12.0*R[38] + 12.1*R[39] + 12.4*R[40]
                    + R[24]*3*m[8]/m[0]*Te + R[32]*3*m[8]/m[4]*(Te))/(3*den[8]+self.ZERO)
        nu_loss = (uB0_2i*Aeff_2i*den[5] + uB0_i*Aeff_i*den[6])/(den[8]*self.Vol+self.ZERO)
        loss = 2.*(Vp + Vs + 2*Te)/3. * nu_loss +loss 
        d_Te = self.dt*(source - loss) + Te
        if write_txt:
            return alpha, alpha_s, Vs
        return d_Te

    def dmax(self,A,B):
        if A>B : 
            return A
        else :
            return B

    def main(self,maxiter, tol, ostep):
        den_old = self.init_dense()
        Te_old = self.init_Te()
        f.write("{},".format(0))
        for l in range(self.n_sp):
            f.write("{},".format(den_old[l]))
        f.write("{}\n".format(Te_old))
        start = time.time()
        # self.den_list = []
        # self.Te_list = []
        # self.den_list.append(den_old)
        # self.Te_list.append(Te_old)
        for i in range(maxiter):
            den_new = np.zeros(self.n_sp)
            Te_new = 0
            
            den_new = self.cal_density(den_old, Te_old)
            Te_new = self.cal_Te(den_new, Te_old)
            residual = 0
            for j in range(self.n_sp):
                residual = self.dmax(residual, abs((den_new[j]- den_old[j])/den_new[j]))
            residual = self.dmax(residual, abs((Te_new - Te_old)/Te_new))

            if residual < tol:
                f.write("{},".format((i+1)*self.dt))
                for l in range(self.n_sp):
                    f.write("{},".format(den_new[l]))
                f.write("{}\n".format(Te_new))
                return den_new, Te_new

            else:
                f.write("{},".format((i+1)*self.dt))
                for l in range(self.n_sp):
                    den_old[l] = den_new[l]
                    f.write("{},".format(den_old[l]))
                Te_old = Te_new
                f.write("{}\n".format(Te_old))
                # self.den_list.append(den_new)
                # self.Te_list.append(Te_new)
            if (i+1) % ostep ==0:
                print("{}/{} 번째 : Residual : {}, density_e : {}, Te : {}, time : {}".format(i+1, maxiter, residual, den_old[8], Te_old, time.time()- start))

# P_abs = float(input(" Absorbed Power : "))
# pressure = float(input("Pressure : "))
# Radius = float(input(" Radius : "))
# Length = float(input("Length : "))
# Tp = float(input("positive ion temperature : "))
# Tn = float(input("negative ion temperature : "))
# Qin = float(input("Input flow : "))
# gamma_rec = float(input("Gamma rec : "))
# P_abs = np.array([150,300,323,500,1000,2000])
# P_abs = np.array([300.0])
# P_abs = np.array([1000.0])
# pressure = 10**np.linspace(0,2,10)[6:]
# P_abs = np.array([20.0,40.0,60.0,120.0,150.0,220.0,300.0,440.0,500.0,700.0])
P_abs = 300.0
pressure = 10**np.linspace(0,2,10, endpoint = True)
# pressure = np.array([1.0])
Radius = 0.15
Length = 0.14
Tp = 0.1
Tn = 0.1
Qin = 20.0
gamma_rec = 0.06

# for i in range(len(P_abs)):
for j in range(len(pressure)):


    f=open("Cl_discharge_numerical_data\Cl_model(p({}),P_abs({}),R({}),L({}),gamma({}).txt".format(pressure[j], P_abs, Radius, Length, gamma_rec), 'w')

    Model = Explicit_method( pressure = pressure[j],
                            power = P_abs,
                            Radius = Radius,
                            Length = Length,
                            Tp = Tp,
                            Tn = Tn,
                            Qin = Qin,
                            gamma_rec = gamma_rec)

    den_fin, Te_fin = Model.main(maxiter = 500000000,
            tol = 1.e-8,
            ostep = 50000)
    f.close()

    f_ =open("Cl_discharge_numerical_data\Cl_result(p({}),P_abs({}),R({}),L({}),gamma({}).txt".format(pressure[j], P_abs[i], Radius, Length, gamma_rec), 'w')
    R = Model.calc_reaction_rate(den_fin, Te_fin)
    alpha, alpha_s, Vs = Model.cal_Te(den_fin, Te_fin, write_txt=True)
    Tg = Model.Tg

    #Pressure, power absorb
    f_.write("{},{}\n".format(P_abs[i], pressure[j]))

    #density 와 Te 값 입력
    for k in range(9):
        f_.write("{},".format(den_fin[k]))
    f_.write("{},".format(Te_fin))
    f_.write("{}\n".format(Tg))

    # Reaction 값들 입력
    for k in range(41):
        f_.write("{},".format(R[k]))
        if k==40:
            f_.write("\n")

    # alpha, alpha_s, Vs
    f_.write("{},{},{}".format(alpha, alpha_s, Vs))
    f_.close()



