import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp

tpf_file = 'Z2_ll.Combined.dat'
tpf_data = np.loadtxt(tpf_file, usecols=(0,1,2,3))

t_init = int(tpf_data[0,0])
t_fin = int(tpf_data[tpf_data.shape[0]-1,0])
T = int(t_fin)+1

#print(t_init,t_fin)

D = 18 # number of data points for each timeslice

# 'r' stands for raw
tpf_r = np.zeros(shape=(T,D))
tpf_r_avg = np.zeros(T)

# processing raw data
for t in range (T):
    for i in range (D*t,D*(t+1)):
        tpf_r[t,i%D] = tpf_data[i,2]
    #print(t_set)
    tpf_r_avg[t] = tpf_r[t,:D].mean()
    #print(t, tpf_r_avg[t])
    
#===============================================

# the effective mass plot
def eff_m(T,dat):
    eff_m = np.zeros(T-1)
    for t in range(T-1):
        eff_m[t] = np.abs(-np.log(dat[t+1]/dat[t]))
        #print(eff_m[t])
    return eff_m

T_arr = np.arange(T)
eff_mass = eff_m(T,tpf_r_avg)

#print(eff_mass)


plt.figure()
plt.scatter(T_arr[:T-1], eff_mass, linestyle="None")
plt.xlabel("T")
plt.ylabel("eff_m")

#==================================================

# fold data set and create COV matrix over folded set

T_2 = int(T/2)

# create folded data set
# 'f' stands for folded
tpf_f = np.zeros(shape=(T_2, D))
tpf_f[0,:] = tpf_r[0,:]
tpf_f_avg = np.zeros(T_2)
tpf_f_avg[0] = tpf_r_avg[0]

for t in range(1,T_2):
    tpf_f[t,:] = (tpf_r[t,:]+tpf_r[T-t,:])/2
    tpf_f_avg[t] = tpf_f[t,:].mean()
    #print(t,"\t", tpf_r_avg[t], "\t", tpf_f_avg[t])
    
# create COV over folded data
COV = np.zeros(shape=(T_2,T_2))

for t1 in range(T_2):
    for t2 in range(T_2):
        COV[t1,t2] = ((tpf_f[t1,:]-tpf_f_avg[t1]).dot(tpf_f[t2,:]-tpf_f_avg[t2]))/(D*(D-1))

np.savetxt("COV.csv", COV, delimiter="\t")

#=================================================
    
# correlated fit

fit_start = 7
fit_end = 21
fit_interval = fit_end-fit_start
print("fit range: ", fit_start, " - ", fit_end-1)

# truncating data set from both ends for fitting
T_fit_range = T_arr[fit_start:fit_end]
tpf_f_fit = tpf_f_avg[fit_start:fit_end]
COV_trunc = COV[fit_start:fit_end,fit_start:fit_end]

# Cholesky decomposition
L_inv = np.linalg.cholesky(COV_trunc)
L = np.linalg.inv(L_inv)

# fit ansatz
def c(params,t):
    return params[0]*np.exp(-params[1]*t)
    
# residual/difference vector
def diff(params):
    return tpf_f_fit - c(params, T_fit_range)
    
# LD
def LD(params):
    return L.dot(diff(params))

#guess parameters:
t_guess = int(fit_interval/2)
b0 = np.abs(np.log(tpf_f_fit[t_guess-1]/tpf_f_fit[t_guess]))
a0 = tpf_f_fit[t_guess]/np.exp(-b0*t_guess)
#print([a0,b0])

from scipy.optimize import least_squares

res = least_squares(LD, [a0,b0])
print(res.x)
[a,b] = res.x
chi_sq = ((LD([a,b])).T).dot(LD([a,b]))
print(chi_sq)

#=================================================
    
# TPF plot
plt.figure()
plt.scatter(T_arr, tpf_r_avg)
plt.xlabel("T")
plt.ylabel("TPF_val")

# TPF with fit in fit_range
plt.figure()
plt.scatter(T_fit_range, tpf_f_fit)
plt.plot(T_fit_range, a*np.exp(-b*T_fit_range),'r')
plt.xlabel("T")
plt.ylabel("(TPF_val)")


plt.show()




