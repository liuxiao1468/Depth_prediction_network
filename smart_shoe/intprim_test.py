import numpy as np
import cv2
import pickle
import itertools
from scipy import fftpack


# N = 6

heel = []
meta_1 = []
meta_2 = []
toe = []

with open('1-1.txt') as f:
    for lines in itertools.zip_longest(*[f]*6):
    	# count = count+1
    	# print(lines)
    	# print(len(lines))
    	x = lines[1].split(" ")
    	heel.append(x[1])
    	x = lines[2].split(" ")
    	meta_1.append(x[1])
    	x = lines[3].split(" ")
    	meta_2.append(x[1])
    	x = lines[4].split(" ")
    	toe.append(x[1])


count = 0
with open('1-2.txt') as f:
    for lines in itertools.zip_longest(*[f]*8):
        count = count+1
        # print(lines)
        # print(len(lines))
        # x = lines[1].split(" ")
        # heel.append(x[1])
        # x = lines[2].split(" ")
        # meta_1.append(x[1])
        # x = lines[3].split(" ")
        # meta_2.append(x[1])
        # x = lines[4].split(" ")
        # toe.append(x[1])
print("acc_reading: ", count)

  	
# print(heel)
heel = [int(i) for i in heel]
meta_1 = [int(i) for i in meta_1]
meta_2 = [int(i) for i in meta_2]
toe = [int(i) for i in toe]

print("original: ", len(heel))
start = [ n for n,i in enumerate(heel) if i>10 ][0]
heel = heel[start:]
meta_1 = meta_1[start:]
meta_2 = meta_2[start:]
toe = toe[start:]
print("processed heel: ", len(heel))

pickle_in = open("depth_save_1.pickle","rb")
depth_save = pickle.load(pickle_in)
# pickle_in = open("depth_save_1.pickle","rb")
# depth_save_1 = pickle.load(pickle_in)


length = len(depth_save)
print("depth_data: ",len(depth_save))
stepsize = int(np.floor(len(heel)/len(depth_save)))
print(stepsize)

idx = list(range(1, length+1))
idx = [i * stepsize for i in idx]

p_heel = [heel[i] for i in idx]
p_meta_1 = [meta_1[i] for i in idx]    
p_meta_2 = [meta_2[i] for i in idx]    
p_toe = [toe[i] for i in idx] 


import matplotlib.pyplot as plt
x = np.arange(1, length+1)
y1 = np.asarray(p_heel)
y2 = np.asarray(p_meta_1)
y3 = np.asarray(p_meta_2)
y4 = np.asarray(p_toe)

y5 = np.asarray(depth_save)
# y6 = np.asarray(depth_save_1)
sample_num = 1250

x = x[0:sample_num]
y1 = y1[0:sample_num]
y2 = y2[0:sample_num]
y3 = y3[0:sample_num]
y4 = y4[0:sample_num]
y5 = y5[0:sample_num]


def fft_filter(signal):
    sig_fft = fftpack.fft(signal)
    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)**2

    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(y5.size, d=0.0166)
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    filtered_sig = fftpack.ifft(high_freq_fft)
    filtered_sig = filtered_sig.real
    # real=np.isreal(filtered_sig)
    # real_array=filtered_sig[real]  
    return filtered_sig


y5 = fft_filter(y5)
y1 = fft_filter(y1)
y4 = fft_filter(y4)


# plt.figure(1)
# plt.grid(True)
# plt.plot(x, y1,'-',color = '#afaf76',label='heel',zorder=3)
# plt.plot(x, y2,'-',color = '#3a9b85',label='meta_1',zorder=3)
# plt.plot(x, y3,'-',color = '#9effe9',label='meta_2',zorder=3)
# plt.plot(x, y4,'-',color = '#eba5eb',label='toe',zorder=3)
# plt.title('Shoe sensor readings')
# plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.1))
# plt.figure(2)
# plt.grid(True)
# plt.plot(x, y5,'-',color = '#3a9b85',label='depth_1',zorder=3)
# # plt.plot(x, y6,'-',color = '#eba5eb',label='depth_2',zorder=3)
# plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.1))
# plt.title('Depth camera readings')
# # plt.ylim(10,25)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import intprim
import copy

from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import SR1

training_trajectories = []
for i in range(int(sample_num/50)):
    dof1 = [y5[i*50:(i+1)*50]]
    dof2 = [y1[i*50:(i+1)*50]]
    dof3 = [y4[i*50:(i+1)*50]]
    temp_data = np.concatenate((dof1, dof2, dof3), axis=0)
    training_trajectories.append(temp_data)


fig1 = plt.figure(1,figsize=(30,15))
mean_phase = np.linspace(0,1,50).reshape(1,-1)
ax1 = fig1.add_subplot(311)
ax1.set_ylabel('Observable Variable')
ax1.plot(mean_phase.T,np.asarray(training_trajectories)[:,0,:].T)

ax2 = fig1.add_subplot(312)
ax2.set_ylabel('Latent Variable')
ax2.plot(mean_phase.T,np.asarray(training_trajectories)[:,1,:].T)

ax3 = fig1.add_subplot(313)
ax3.set_ylabel('Latent Control')
ax3.set_xlabel('Phase')
ax3.plot(mean_phase.T,np.asarray(training_trajectories)[:,2,:].T)
plt.show()



# Define the data axis names.
dof_names = np.array(["Observation", "Latent Variable", "Latent Control"])

# Decompose the handwriting trajectories to a basis space with 10 uniformly distributed Gaussian functions and a variance of 0.1.
basis_model = intprim.basis.GaussianModel(10, .005, dof_names) #0.1

# Initialize a BIP instance.
primitive = intprim.BayesianInteractionPrimitive(basis_model)

# Train the model.
for trajectory in training_trajectories:
    primitive.add_demonstration(trajectory)


# Calculate sample mean and covariance of demonstrations
demo_mean, demo_cov = primitive.get_basis_weight_parameters()

# Set an observation noise for the demonstrations.
observation_noise = np.diag([0.0001, 1000000**2, 1000000**2])

# Define the active DoF
active_dofs = np.array([0])

# Compute the phase mean and phase velocities from the demonstrations.
phase_velocity_mean, phase_velocity_var = intprim.examples.get_phase_stats(training_trajectories)

# Define a filter to use. Here we use an ensemble Kalman filter
filter = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
    basis_model = basis_model,
    initial_phase_mean = [0.0, phase_velocity_mean],
    initial_phase_var = [1e-5, phase_velocity_var],
    proc_var = 1e-7,
    initial_ensemble = primitive.basis_weights)
# Create test trajectories.
test_trajectories = [training_trajectories[5]]
# Explicitly zero out the x-axis values to illustrate that they are not being used.
test_trajectory_partial = np.array(test_trajectories[0], copy = True)
test_trajectory_partial[1:3,:] = 0.0


# We set a copy of the filter here so we can re-use it later. The filter maintains internal state information and should not be used more than once.
primitive.set_filter(copy.deepcopy(filter))

# Calculate the mean trajectory for plotting.
mean_trajectory = primitive.get_mean_trajectory()




prev_observed_index = 0
observed_index=20
inferred_trajectory,phase,traj_mean,var = primitive.generate_probable_trajectory_recursive(
    test_trajectory_partial[:, prev_observed_index:observed_index],
    observation_noise,
    active_dofs,
    num_samples = test_trajectory_partial.shape[1] - observed_index)

upper_b = traj_mean + np.sqrt(np.diag(var))
lower_b = traj_mean - np.sqrt(np.diag(var))
dictlist = list(basis_model.computed_basis_values.values())
latent_upper_bound=[]
latent_lower_bound=[]
control_upper_bound=[]
control_lower_bound=[]
for i in range (0, 100):
    latent_upper_bound.append(np.dot(dictlist[i],upper_b[10:20]))
    latent_lower_bound.append(np.dot(dictlist[i],lower_b[10:20]))
    control_upper_bound.append(np.dot(dictlist[i],upper_b[20:30]))
    control_lower_bound.append(np.dot(dictlist[i],lower_b[20:30]))

mean_phase = np.linspace(0,1,100)
test_phase = np.linspace(0,phase,test_trajectory_partial[:, :observed_index].shape[1])
infer_phase = np.linspace(phase,1,inferred_trajectory.shape[1])
centers = primitive.basis_model.centers

fig1 = plt.figure(2,figsize=(30,15))
ax1 = fig1.add_subplot(311)
ax1.plot(mean_phase,mean_trajectory[0],'k')
ax1.plot(test_phase,test_trajectory_partial[0, :observed_index],'b')
ax1.plot(infer_phase, inferred_trajectory[0],'r--')
ax1.set_ylabel('Observation')
ax1.grid(True)

ax2 = fig1.add_subplot(312)
ax2.fill_between(mean_phase, latent_upper_bound, latent_lower_bound, color='r', alpha=0.1)
ax2.plot(mean_phase,mean_trajectory[1],'k')
ax2.plot(infer_phase, inferred_trajectory[1],'r--')
ax2.set_ylabel('Latent Variable')
ax2.grid(True)

ax3 = fig1.add_subplot(313)
ax3.fill_between(mean_phase, control_upper_bound, control_lower_bound, color='r', alpha=0.1)
ax3.plot(mean_phase,mean_trajectory[2],'k')
ax3.plot(infer_phase, inferred_trajectory[2],'r--')
ax3.set_ylabel('Control')
ax3.set_xlabel('Phase')
ax3.grid(True)

plt.show()
