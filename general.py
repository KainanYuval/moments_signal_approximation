import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import dft
from numpy.linalg import inv
import nevergrad as ng


check_sanity = True
save_fig = True
plot_fig = False
budget = 2000
# ///////////////////////////////////////////// define signal
L = 20
t = np.linspace(0, L, L)
noise_var = 1


construct_f_from_fft_elements = True

if construct_f_from_fft_elements:
    fft_elements = np.random.randint(1,3,L)
    fft_elements[1] = 5
    fft_elements[4] = 5
    clean_signal = abs(np.fft.ifft(fft_elements))   # f(t)
    clean_signal[0] = 0.4
    clean_signal = 4*clean_signal
else:
    def f(x):
        h = 0.1
        sigma = 0.6
        return (np.exp(-((x+h-5)/(2*sigma))**2) - np.exp(-((x-h-5)/(2*sigma))**2))/(2*h)
    clean_signal = np.zeros(L)
    clean_signal[6:10] = 0.35
    clean_signal[11:14] = -0.35
    #clean_signal = f(t)

if check_sanity:
    if (min(abs(np.fft.fft(clean_signal))) - 0.0) < 0.001:
        print('vanishing fft coeffitions - wont work')

P_signal = np.mean(np.square(clean_signal))
SNR = np.mean(np.square(clean_signal))/noise_var**0.5
print('SNR = ' + str(SNR))

# /////////////// define p
random_rotation = True
if random_rotation:
    p = np.random.randint(0, 100, L)

uniform_rotation = False
if uniform_rotation:
    p = [1]*L

zero_rotation = False
if zero_rotation:
    p = [0] * L
    p[0] = 1

p = (float(sum(p))**-1)*np.array(p)

if check_sanity:
    is_periodic = False
    for i in range(1, L, 1):
        if (p == np.roll(p, i)).all():
            is_periodic = True
            break
    if is_periodic:
        print('p is periodic!!')


# ///////////////////////////////////////////// ALGOs
def sample():
    s = np.random.choice(L, 1, p=p)[0]
    noise = np.random.normal(0, noise_var**0.5, L)
    noisy_signal = np.add(np.roll(clean_signal, s), noise)
    return noisy_signal


def get_M1_M2(observations, noise_var=None):
    N = len(observations)
    M1 = np.mean(observations, axis=0)
    if noise_var is None:
        noise_var = np.var(observations) # TODO understand exactly why the sqrt operation
    M2 = -np.diag([noise_var]*L) + (1/N)*np.transpose(observations) @ observations
    return M1, M2


def get_F_invF():
    F = dft(L)
    F = np.mat(F)
    invF = inv(F)
    return F, invF


def get_circular_matrix(u):
    mat = []
    for i in range(len(u)):
        mat.append(np.roll(u, i))
    return np.transpose(np.mat(mat))


def algo_1(M1, M2, F, invF, L):
    # normalize Fx
    Px = L* np.diag(F @ M2 @ invF)
    if check_sanity:
        Px_clean = np.abs(np.fft.fft(clean_signal)) ** 2
        if 2*np.linalg.norm(Px_clean - Px)/(np.linalg.norm(Px)+np.linalg.norm(Px_clean)) > 0.01:
            print('bad power spectrum approximation')
        else:
            print('power spectrum proparly approximated')
    Px_sqrt = np.sqrt((Px.real.clip(min=0.001)))

    Dx = np.diag(Px_sqrt**-1)
    Q = invF @ Dx @ F
    M2_tild = Q @ M2 @ Q.H

    # get x and p
    eig_values, eig_vectors = np.linalg.eig(M2_tild)
    complex_value_per_eig_vector = []
    for i in range(L):
        complex_value_per_eig_vector.append(np.sum( np.square(np.fft.fft(eig_vectors[i]).imag)))
    index = complex_value_per_eig_vector.index(min(complex_value_per_eig_vector))
    v = np.transpose(eig_vectors[index])

    arg1 = np.squeeze(np.array(F @ v))
    arg2 = np.multiply(Px_sqrt, arg1)
    v_tild = invF @ arg2
    v_tild = np.squeeze(v_tild)
    x = (np.sum(M1)/np.sum(v_tild)) * v_tild.__array__()[0]

    rho = np.linalg.inv(get_circular_matrix(x)) @ np.array(M1)[0]

    return x, rho


def EM_optimization_DFO(M1, M2, L):
    def loss(x_concate_rho):
        lambda0 = 1/(L*(1+3*noise_var))
        x = x_concate_rho[:L]
        rho = x_concate_rho[L:]
        rho = rho*(1/np.sum(rho))
        Cx = get_circular_matrix(x)
        Drho = np.diag(rho)
        arg1 = np.linalg.norm(M2 - Cx @ Drho @ np.transpose(Cx))
        arg2 = np.linalg.norm(M1 - Cx @ rho)
        return arg1**2 + lambda0*arg2**2
    optimizer = ng.optimizers.CMandAS2(instrumentation=2*L, budget=2*L*budget)
    recommendation = optimizer.optimize(loss)
    x_tild = recommendation.data[:L]
    rho = recommendation.data[L:]

    return x_tild, rho


def algo_wrapper(N, use_LS):
    samples = []
    for i in range(N):
        samples.append(sample())
    samples = np.mat(samples)
    M1, M2 = get_M1_M2(samples, noise_var=noise_var)
    if use_LS:
        return EM_optimization_DFO(M1, M2, L)
    else:
        F, invF = get_F_invF()
        return algo_1(M1, M2, F, invF, L)


def check_min_error(x, x_tild):
    min_error = 10000000000
    for i in range(L):
        err = np.linalg.norm(np.roll(x_tild, i) -x)**2
        err = err/(np.linalg.norm(x)**2)
        if  err < min_error:
            min_error = err
            index = i
    return err, index


def experiment(N_min, N_max, number_of_rep, deltaN, use_LS):
    N_list = []
    err = []
    for N in range(N_min, N_max, deltaN):
        for rep in range(number_of_rep):
            x_tild, rho = algo_wrapper(N, use_LS=use_LS)
            N_list.append(N)
            err1 = check_min_error(clean_signal, x_tild)[0]
            print('err = ' + str(err1))
            err.append(err1)

    sns.lineplot(x=N_list, y=err, label='error vs N')
    if use_LS:
        name = 'LS signal approximation, MSE vs N , sigma = ' + str(noise_var)
    else:
        name = 'spectral algo, MSE vs N , sigma = ' + str(noise_var)

    plt.title(name)
    if save_fig:
        plt.legend()
        dir = r'experiments/'
        name = dir +str(name) + ".png"
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(name, bbox_inches='tight')
    if plot_fig:
        plt.show()
    plt.close()


def graphic_test(N, use_LS):

    save_fig = True
    plot_signal_and_sample = False
    if plot_signal_and_sample:
        sns.lineplot(x=t, y=clean_signal, label='clean')
        sns.lineplot(x=t, y=sample(), label='sample')
        plt.show()
    x_tild, rho = algo_wrapper(N, use_LS)
    rho = rho*(1/np.sum(rho))
    err, roll_index = check_min_error(clean_signal, x_tild)
    print('err = ' + str(err))
    f, axes = plt.subplots(1, 2)
    if use_LS:
        type = 'LS algo, ES optimizer'
    else:
        type = 'spectral algo'
    if err < 0.0001:
        MSE ='0'
    else:
        MSE = "%.4f" % err
    name = type + ', N = ' + str(N) + ' , sigma = ' + str(noise_var) + ' , MSE = ' + MSE

    plt.title(name)

    sns.lineplot(x=t, y=clean_signal, label='x_clean', ax=axes[0])
    sns.lineplot(x=t, y=np.roll(x_tild, roll_index), label='x_approximation', ax=axes[0])

    sns.lineplot(x=range(L), y=p, label='p_clean', ax=axes[1])
    sns.lineplot(x=range(L), y=rho, label='p_recovered', ax=axes[1])

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    plt.savefig("myplot.png", dpi=100)
    if save_fig:
        plt.legend()
        dir = r'experiments/'
        name = dir +str(name) + ".png"
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(name, figsize=(15.0, 10.0))
    if plot_fig:
        plt.show()
    plt.close()

if __name__ == '__main__':
    MSE_vs_N = False
    if MSE_vs_N:
        experiment(N_min=100,
                   N_max=100000,
                   number_of_rep=5,
                   deltaN=(100000 - 100) // 6,
                   use_LS=True)
    else:
        use_LS = [True, False]
        #use_LS = [False]
        for use in use_LS:
            graphic_test(N=1000, use_LS=use)
            graphic_test(N=10000, use_LS=use)
            graphic_test(N=100000, use_LS=use)


