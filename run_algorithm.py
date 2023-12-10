import Hyperparameters
import Hyperparameters as hp
import numpy as np

hyperparameters = hp.Hyperparameters(x_dimension=10,
                                     nn_width=10,
                                     nn_activation="relu",
                                     dal_num_iterations=20,
                                     max_error=0.3)

# Network
network = hp.BasicNN(input_dim=hyperparameters.x_dimension,
                     width=hyperparameters.nn_width,
                     activation=hyperparameters.nn_activation)

# Dataset
distribution = hp.Distribution(x_dimension)



num_dataset = 1000
unlabelled_dataset = distribution.sample(num_dataset)

labelled_dataset = np.array([]) # x, y pairs
labelled_weights = np.array([]) # u
sampling_algorithm = hp.SamplingProcedure(distribution, x_dimension)
num_sampling_iterations = hyperparameters.dal_num_iterations
for i in range(num_sampling_iterations):
    weight, sample = sampling_algorithm.sample(unlabelled_dataset)

hp.train_network(network, labelled_dataset, labelled_weights)
# define testing set