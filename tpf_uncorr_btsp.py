import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy as sp
import csv

tpf_file = 'Z2_ll.Combined.dat'
tpf_data = np.loadtxt(tpf_file, usecols=(0,1,2,3))

t_init = int(tpf_data[0,0])
t_fin = int(tpf_data[tpf_data.shape[0]-1,0])
T = int(t_fin)

#print(t_init,t_fin)

D = 18 # number of data points for each timeslice

tpf_r = np.zeros(shape=(T,D))
tpf_r_avg = np.zeros(T)

# processing raw data
for t in range (T):
    avg=0
    for i in range (D*t,D*(t+1)):
        tpf_r[t,i%D] = tpf_data[i,2]
        avg = avg + tpf_r[t,i%D]
    #print(t_set)
    tpf_r_avg[t] = avg/D
    #print(t, tpf_r_avg[t])
    
#===============================================
#bootstrap analysis

K = 1000 # number of bootstrap samples

#generate bootstrap samples
btsp_samp = np.zeros(shape=(T,K,D))
tpf_btsp = np.zeros(T)

for t in range(T):
    for k in range(K):
        btsp_samp[t,k,:D] = random.choices(tpf_r[t,:D],k=D)
    tpf_btsp[t] = btsp_samp[t,:K,:D].mean()

#print(tpf_btsp)

    
#===============================================

# the effective mass array
def eff_m(T,dat):
    eff_m = np.zeros(T-1)
    for t in range(T-1):
        eff_m[t] = np.abs(-np.log(dat[t+1]/dat[t]))
        #print(eff_m[t])
    return eff_m

T_arr = np.array(list(range(T)))
eff_mass = eff_m(T,tpf_btsp)

#print(eff_mass)


plt.figure()
plt.scatter(T_arr[:T-1], eff_mass, linestyle="None")
plt.xlabel("T")
plt.ylabel("eff_m")


#================================================

# fold over set
T_2 = int(T/2)
#print(T)

tpf_f = np.zeros(T_2)
tpf_f[0] = tpf_btsp[0]
#err_f = np.zeros(T_2)
#err_f[0] = err[0]

for i in range (1,T_2):
    tpf_f[i] = (tpf_btsp[i]+tpf_btsp[T-i])/2
    #err_f[i] = err[i]+err[T-i]
    #print(tpf_f[i],err_f[i])
    
#==================================================
    
    
T_2_arr = np.array(list(range(T_2)))

fit_start = 7
fit_end = 21

print("fit range: ", fit_start, " - ", fit_end)

#truncating data set from both ends for fitting
T_fit_range = T_arr[fit_start:fit_end]
tpf_f_fit = tpf_f[fit_start:fit_end]


# nonlinear fit
from scipy.optimize import curve_fit

# fitting data to exponential ansatz
def expo1(t,a,b):
    return a*np.exp(-b*t)

popt, pcov = curve_fit(expo1, T_fit_range, tpf_f_fit)

a,b = popt
perr = np.sqrt(np.diag(pcov))
print("a=",a,"\t err=", perr[0],"\nb=",b,"\t err=",perr[1])

#=================================================
    
#TPF plot with errorbars
plt.figure()
plt.scatter(T_arr, tpf_r_avg)
plt.xlabel("T")
plt.ylabel("TPF_val")

plt.figure()
plt.scatter(T_fit_range, tpf_f[fit_start:fit_end])
plt.plot(T_fit_range, a*np.exp(-b*T_fit_range),'r')
plt.xlabel("T")
plt.ylabel("(TPF_val)")


plt.show()




