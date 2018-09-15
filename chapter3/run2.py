import mnist_loader

import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

input_size = 28 * 28
hidden_size = 30
hidden_layer_num = 1
output_size = 10

Neural_structure = [input_size]
for l in range(hidden_layer_num):
    Neural_structure.append(hidden_size)
Neural_structure.append(output_size)

net = network2.Network(Neural_structure, cost=network2.CrossEntropyCost())

print('Layer # = %d'%(hidden_layer_num))
print('NN structure:',Neural_structure)


epochs = 30
mini_batch_size = 10
eta = 0.5

net.SGD(training_data[:100], epochs, mini_batch_size, eta, 
        lmbda = 5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True)
