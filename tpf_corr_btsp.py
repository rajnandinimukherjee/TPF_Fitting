import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp

tpf_file = 'Z2_ll.Combined.dat'
tpf_data = np.loadtxt(tpf_file, usecols=(0,1,2,3))

t_init = int(tpf_data[0,0])
t_fin = int(tpf_data[tpf_data.shape[0]-1,0])
T = (t_fin-t_init)+1

#print(t_init,t_fin)

D = 18 # number of data points for each timeslice

tpf_r = np.zeros(shape=(T,D))
tpf_r_avg = np.zeros(T)

# processing raw data
for t in range (T):
    for i in range (D*t,D*(t+1)):
        tpf_r[t,i%D] = tpf_data[i,2]
    #print(t_set)
    tpf_r_avg[t] = tpf_r[t,:D].mean()
    #print(t, tpf_r_avg[t])
    
#============================================
#bootstrap analysis

K = 10000 # number of bootstrap samples

#generate bootstrap samples
btsp_samp = np.zeros(shape=(T,K,D))
samp_avg = np.zeros(shape=(T,K))
tpf_btsp = np.zeros(T)

for t in range(T):
    for k in range(K):
        btsp_samp[t,k,:D] = random.choices(tpf_r[t,:D],k=D)
        samp_avg[t,k] = btsp_samp[t,k,:].mean()
    tpf_btsp[t] = btsp_samp[t,:,:].mean()

#print(tpf_btsp)

#===============================================

# the effective mass plot
def eff_m(T,dat):
    eff_m = np.zeros(T-1)
    for t in range(T-1):
        eff_m[t] = np.abs(-np.log(dat[t+1]/dat[t]))
        #print(eff_m[t])
    return eff_m

T_arr = np.arange(T)
eff_mass = eff_m(T,tpf_btsp)

#print(eff_mass)


plt.figure()
plt.scatter(T_arr[:T-1], eff_mass, linestyle="None")
plt.xlabel("T")
plt.ylabel("eff_m")

    
#==================================================

# calculating covariance matrix COV for each bootstrap sample

COV = np.zeros(shape=(K,T,T))

for k in range(K):
    for t1 in range(T):
        for t2 in range(T):
            COV[k,t1,t2] = ((btsp_samp[t1,k,:D]-tpf_btsp[t1]).dot((btsp_samp[t2,k,:D]-tpf_btsp[t2])))/(D*(D-1))
        
#np.savetxt("COV.csv", COV, delimiter="\t")

#==================================================
    
# correlated fit on each bootstrap sample

T_2 = int(T/2)

fit_start = 7
fit_end = 21

print("fit range: ", fit_start, " - ", fit_end-1)

# truncating data set from both ends for fitting
T_fit_range = T_arr[fit_start:fit_end]
COV_trunc = COV[:,fit_start:fit_end, fit_start:fit_end]

params_dist = np.zeros(shape=(K,2))

for k in range(K):
    
    # fold over each bootstrap sample
    tpf_f = np.zeros(T_2)
    tpf_f[0] = samp_avg[0,k]
    #err_f = np.zeros(T_2)
    #err_f[0] = err[0]

    for i in range (1,T_2):
        tpf_f[i] = (samp_avg[i,k]+samp_avg[T-i,k])/2
        
    tpf_f_fit = tpf_f[fit_start:fit_end]
    
    
    # Cholesky decomposition
    L_inv = np.linalg.cholesky(COV_trunc[k,:,:])
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
    t_guess = int((fit_end-fit_start)/2)
    b0 = np.abs(np.log(tpf_f_fit[t_guess-1]/tpf_f_fit[t_guess]))
    a0 = tpf_f_fit[t_guess]/np.exp(-b0*t_guess)
    #print([a0,b0])

    from scipy.optimize import least_squares

    res = least_squares(LD, [a0,b0])
    params_dist[k,:] = res.x
    #chi_sq = ((LD(res.x)).T).dot(LD(res.x))
    #print(chi_sq)
    
#====================================================
    
# histogram of fit parameters over K bootstrap samples
plt.figure()
a_data = params_dist[:,0]
a = a_data.mean()
a_bins = np.arange(min(a_data), max(a_data),(max(a_data)-min(a_data))/500)
plt.hist(a_data, bins=a_bins, alpha=0.5)
plt.axvline(a, color='k')

plt.figure()
b_data = params_dist[:,1]
b = b_data.mean()
b_bins = np.arange(b-0.05,b+0.05,0.1/500)
plt.hist(b_data, bins=b_bins, alpha=0.5)
plt.axvline(b, color='k')

print("a = ", a, "\nb = ", b)

plt.show()




