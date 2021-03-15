import numpy as np
import matplotlib.pyplot as plt

x_data = [90.,88.,79.,60.,70.,83.,86.,97.,91.,87.]
y_data = [85.,84.,79.5,70.,75.,81.5,83.,88.5,85.5,83.5]
#? y_data = b + w * x_data

'''
#! closed-form solution

x = np.arrange(-200,-100,1) #? bias
y = np. arrange(-5,5,0.1) #? weight
z = np.zeros((len(x),len(y)))
x,y = np.meshgrid(x,y)
for i in range(len(x)):
    b = x[i]
    w = y[i]
    z[i][j] = 0
    for n in range(len(x_data)):
        z[j][i] = z[j][i] + (y_data - b - w*x_data)**2
    z[j][i] = z[j][i]/len(x_data)
'''

#TODO: randomly choose an initial point (random weight and random bias)
b = -120 #* initial bias
w = -4 #* initial weight
lr = 2 #* learning rate
iteration = 100000 #? will iterate for 100 thousands times

#TODO: store initial values for plotting
b_history = [b]
w_history = [w]

#? initail seperated learning rate of weight and bias
lr_b = 0
lr_w = 0

#TODO: iterations
for i in range(iteration):

    b_grad = 0.0 #* bias gradient 新的b點位移預測
    w_grad = 0.0 #* weight gradient 新的w點位移預測

    for n in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[n] - (b + w*x_data[n]))*1.0
        #! 新的dradient descent = 舊的gradient descent - 估測誤差的平方和( L(w,b) )對bias做偏微分
        w_grad = w_grad - 2.0*(y_data[n] - (b + w*x_data[n]))*x_data[n]
        #! 新的dradient descent = 舊的gradient descent - 估測誤差的平方和( L(w,b) )對weight做偏微分

    #TODO: Adagrad 修改 learning rate
    #? 賦予b跟w客製化的learning rate
    lr_b = lr_b + b_grad**2
    lr_w = lr_w + w_grad**2
    #? update parameters.
    b = b - lr/np.sqrt(lr_b)*b_grad
    w = w - lr/np.sqrt(lr_w)*w_grad

    #? store parameters for plotting
    b_history.append(b)
    w_history.append(w)

#TODO: plot the figure
print('Minimize point: bias = ',b_history[-1],' weight = ',w_history[-1])
plt.plot([40],[0.5],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'ro')
plt.show()
