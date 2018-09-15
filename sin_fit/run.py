import Network
import numpy as np
import matplotlib.pyplot as plt

N = 100000

X = np.random.rand(N)*2.0*np.pi
Y = np.sin(X)


training_data = [[np.array([[x]]), y] for x, y in zip(X, Y)]

input_size = 1
hidden_size = 10
hidden_layer_num = 10
output_size = 1

Neural_structure = [input_size]
for l in range(hidden_layer_num):
    Neural_structure.append(hidden_size)
Neural_structure.append(output_size)

net = Network.Network(Neural_structure, cost=Network.CrossEntropyCost())

print('Layer # = %d'%(hidden_layer_num))
print('NN structure:',Neural_structure)


epochs = 10
mini_batch_size = 10
eta = 0.1

net.SGD(training_data, epochs, mini_batch_size, eta, 
        lmbda = 5.0,
        )

x = []
sin = []
for z in np.linspace(0.0, 2.0*np.pi, 100):
    x.append(z)
    sin.append((net.feedforward(np.array([[z]]))[0]))
    
Y = Y[np.argsort(X)]
X = np.sort(X)
plt.title("Epoch = %d" %epochs)
plt.plot(X, Y, label='training_data', linestyle='dashed')
plt.plot(x, sin, label='Neural_Network')
plt.legend()
plt.savefig('result_epoch=%d.png'%epochs)
# plt.show()
