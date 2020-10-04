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

#print(t_init,t_fin)

# statistical bootstrap at each timeslice

D = 18 # number of data points for each timeslice
K = 1000 # number of samples

tpf_val = np.zeros(t_fin-t_init+1)
err = np.zeros(t_fin-t_init+1)



for t in range (t_init,t_fin+1):
    t_set = np.zeros(18)
    for i in range (D*t,D*(t+1)):
        t_set[i-(D*t)] = tpf_data[i,2]
    #print(t_set)
    set_avg = float(np.sum(t_set))/D
    
    btsp = 0
    btsp_err_sq = 0
    for j in range (0,K):
        j_avg = (float(np.sum(random.choices(t_set,k=D)))/D)
        btsp = btsp + j_avg
        btsp_err_sq = btsp_err_sq + (j_avg-set_avg)**2
        
    tpf_val[t] = float(btsp)/float(K)
    err[t] = np.sqrt(float(btsp_err_sq)/float(K))
    
    print(t,tpf_val[t], err[t])

"""# btsp avgs write to file
data = [t, tpf_val, err]

with open("btsp_tpf.dat", "w+") as out:
    for x in zip(*data):
        out.write("{0}\t{1}\n".format(*x))
out.close()"""


# fitting data to exponential ansatz
def expo1(t,a,b,c):
    return a + b*exp(-c*t)

# fold over set
T = int(t_fin)+1
T_2 = int(T/2)
#print(T)

tpf_f = np.zeros(T_2)
tpf_f[0] = tpf_val[0]
err_f = np.zeros(T_2)
err_f[0] = err[0]

ln_tpf_f = np.zeros(T_2) #to store the ln of vals



for i in range (1,T_2):
    tpf_f[i] = (tpf_val[i]+tpf_val[T-i])/2
    err_f[i] = err[i]+err[T-i]
    #print(tpf_f[i],err_f[i])
    ln_tpf_f[i] = np.log(tpf_f[i])
    
    
T_arr = np.array(list(range(T)))
T_2_arr = np.array(list(range(T_2)))

fit_start = 4
fit_end = 28

print("fit range: ", fit_start, " - ", fit_end)

#truncating data set from both ends for fitting
T_fit_range = T_arr[fit_start:fit_end]
ln_tpf_f_fit = ln_tpf_f[fit_start:fit_end]

    
"""#linear fit for ln(y) = a + b*t using polyfit
[b1,a1] = np.polyfit(T_fit_range,ln_tpf_f_fit,1)
print("energy1 = ", -b1)"""

#linear fit using LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(T_fit_range.reshape(-1,1),ln_tpf_f_fit)
a2 = model.intercept_
b2 = model.coef_[0]

print("int = ", a2, "slope = ",b2)

gof = r2_score(ln_tpf_f_fit, model.predict(T_fit_range.reshape(-1,1)))

print("gof = ", gof)


#TPF plot with errorbars
plt.figure()
plt.errorbar(T_arr, tpf_val, yerr=err, linestyle="None")
plt.xlabel("T")
plt.ylabel("TPF_val")

plt.figure()
plt.scatter(T_fit_range, np.log(tpf_f[fit_start:fit_end]))
plt.plot(T_fit_range, (a2 + b2*T_fit_range),'r')
plt.title(gof)
plt.xlabel("T")
plt.ylabel("ln(TPF_val)")

plt.figure()
plt.scatter(T_2_arr, tpf_f, linestyle="None")
plt.plot(T_fit_range, np.exp(a2 + b2*T_fit_range), 'r')
plt.xlabel("T")
plt.ylabel("TPF_val")


plt.show()




