import mnist_loader

import network_cross_entropy as network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

input_size = 28 * 28
hidden_size = 1000
hidden_layer_num = 1
output_size = 10

Neural_structure = [input_size]
for l in range(hidden_layer_num):
    Neural_structure.append(hidden_size)
Neural_structure.append(output_size)

net = network.Network(Neural_structure)

print('Layer # = %d'%(hidden_layer_num))
print('NN structure:',Neural_structure)


epochs = 100
mini_batch_size = 10
eta = 3.0

net.SGD(training_data[:10], epochs, mini_batch_size, eta, test_data=test_data)
