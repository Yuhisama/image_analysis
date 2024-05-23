import numpy as np
import matplotlib.pyplot as plt
import secrets
import math
from scipy.stats import norm
# Logistic Map function
def logistic_map(r, x):
    return r * x * (1 - x)
def logistic_map_with_counter(r, x, counter):
    return (r * x * (1 - x)*counter) % 1
def logistic_map_2d(x,y,u1,u2,r1,r2):
    x_next = x*u1*(1-x) + r1*(y*y)
    y_next = y*u2*(1-y) + r2*((x*x)+(x*y))
    return x_next, y_next
# Tent Map function
def tent_map(x, r, counter):
    if x < 0.5:
        return (r * x * counter) % 1
    else:
        return (r * (1 - x) * counter) % 1
# Sin Map function
def sin_map(x, r, counter):
    return (r * math.sin(math.pi * x) * counter) % 1

# Parameters
def generate_parameters():
    random_number = secrets.token_hex(16)
    r_values = (int(random_number[12:28], 16) / (2**64)) *5
    x_initial = int(random_number[0:12], 16) / (2**48)  # Initial value of x
    counter_i0 = int(random_number[28:], 16)
    return r_values , x_initial , counter_i0
# Number of iterations to observe
iterations = 10000  # Number of iterations to observe

# Create a list to store the x values
x_values = []
y_values = []
# Iterate through different parameter values
for i in range(1):
    r_values, x_initial, counter_i0 = generate_parameters()
    r = r_values
    x = x_initial
    counter_x = counter_i0
    counter_y = counter_i0
    # Iterate to reach a stable state
    for _ in range(100):
        x = logistic_map_with_counter(r, x, counter_x)
        y = logistic_map_with_counter(r, x, counter_y)
        # x = tent_map(r, x, counter)
        # x = sin_map(r, x, counter)
        counter_y = counter_y + 1
    # Collect data points after reaching a stable state
    for _ in range(iterations):
        x = logistic_map_with_counter(r, x, counter_x)
        y = logistic_map_with_counter(r, x, counter_y)
        # x = tent_map(r, x, counter)
        # x = sin_map(r, x, counter)
        x_values.append([r, x])
        y_values.append([r, y])
        counter_y = counter_y + 1
    # print(f'loading {i}')


# # Convert the data to a NumPy array
x_values = np.array(x_values)
y_values = np.array(y_values)
values = [x_values, y_values]
# plot Analysis chart
def Probabilty_Density(values):
    # hist:對應的bins值有幾個; bins = 10000 ,range=(0,1):指的是0~1之間取10000個間格
    hist, bins = np.histogram(values[0][0:iterations, 1], bins=1000, range=(0, 1))
    plt.figure(figsize=(8,4))
    plt.plot(bins[:-1], hist/iterations, 'm-', label=f'r value : {values[i][0][0]}')
    plt.title('Probability Density', fontsize = 12)
    plt.xlabel('Xn', fontsize = 10)
    plt.ylabel('P(x)', fontsize = 10)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.show()
Probabilty_Density(values)





def Sensitivity_Analysis(values):
    fig, axes = plt.subplots(1,2, figsize=(8,2))
    fig.suptitle('Sensitivity Analysis')
    plot_name = ["My purpose", "Logistic with counter"]
    for i in range(len(values)):
        ax = axes[i]
        ax.plot(np.arange(iterations), values[i][0:iterations,1],'m-', label = f'r value :{values[i][0][0]}')
        ax.set_title(f'{plot_name[i]}', fontsize = 12)
        ax.set_xlabel('Iterations', fontsize = 10)
        ax.set_ylabel('Xn', fontsize = 10)
        ax.legend()
        # ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.show()
# Sensitivity_Analysis(values)
def Bifurcation_Diagram(values):
    fig, axes = plt.subplots(1,2, figsize=(8,2))
    fig.suptitle('Bifurcation_Diagra')
    plot_name = ["My purpose", "Logistic with counter"]
    for i in range(len(values)):
        ax = axes[i]
        ax.scatter(values[i][:, 0], values[i][:, 1], s=0.1, cmap='viridis', marker='.')
        ax.set_title(f'{plot_name[i]}', fontsize = 12)
        ax.set_xlabel('Parameter r', fontsize = 10)
        ax.set_ylabel('Xn', fontsize = 10)
        ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.show()
# Bifurcation_Diagram(values)