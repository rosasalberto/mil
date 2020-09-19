import os

import numpy as np

current_file = os.path.abspath(os.path.dirname(__file__))

def load():
    # load dataset of mnist bags that bag class == 1 if contains digit 7, data is normalized [0,1]
    train_bags, train_bag_labels, train_instance_labels, test_bags, test_bag_labels, test_instances_labels = np.load(current_file + './npy/mnist_bags_7.npy', allow_pickle=True)
    return (train_bags, train_bag_labels, train_instance_labels), (test_bags, test_bag_labels, test_instances_labels)

def load_42():
    # load dataset of mnist bags that bag class == 1 if contains digit 4 and 2 consecutive, data is normalized [0,1]
    train_bags, train_bag_labels, train_instance_labels, test_bags, test_bag_labels, test_instances_labels = np.load(current_file + './npy/mnist_bags_42.npy', allow_pickle=True)
    return (train_bags, train_bag_labels, train_instance_labels), (test_bags, test_bag_labels, test_instances_labels)
    
def load_2_and_3():
    # load dataset of mnist bags that bag class == 1 if contains digit 2 and 3, data is normalized [0,1]
    train_bags, train_bag_labels, train_instance_labels, test_bags, test_bag_labels, test_instances_labels = np.load(current_file + './npy/mnist_bags_2_and_3.npy', allow_pickle=True)
    return (train_bags, train_bag_labels, train_instance_labels), (test_bags, test_bag_labels, test_instances_labels)