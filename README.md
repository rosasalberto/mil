# mil: multiple instance learning library for Python
---
When working on a research problem, I found myself with the [multiple instance learning (MIL)](https://en.wikipedia.org/wiki/Multiple_instance_learning) framework, which I found quite interesting and unique. After carefully reviewing the literature, I decided to try few of the algorithms on the problem I was working on, but for my surprise, there was no standard, easy, and updated MIL library for any programming language. So... here we are. <br/>
The mil library tries to achieve reproducible and productive research using the MIL framework.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Contributing](#contributing)
- [To-do-list](#to-do-list)
- [License](#license)

---
### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install mil.

```bash
pip install mil
```

### Features

The overall implementation tries to be user-friendly as possible. That's why most of it constructed on top of [sklearn](https://scikit-learn.org/) and [tensorflow.keras](tensorflow.org/api_docs/python/tf/keras).

- [mil.data](#data)
- [mil.bag_representation](#bag_representation)
- [mil.dimensionality_reduction](#dimensionality_reduction)
- [mil.metrics](#metrics)
- [mil.models](#models)
- [mil.preprocessing](#preprocessing)
- [mil.utils](#utils)
- [mil.validators](#validators)
- [mil.trainer](#trainer)

#### data
Very well known datasets of the multiple instance learning framework have been added to the library. For each of the datasets a train and test split 
has been done for reproducibility purposes. The API is similar to the tensorflow datasets in order to create and experiment in a fast and easy way.

```python
# importing all the datasets modules
from mil.data.datasets import musk1, musk2, protein, elephant, corel_dogs, \
                              ucsb_breast_cancer, web_recommendation_1, birds_brown_creeper, \
                              mnist_bags
# load the musk1 dataset
(bags_train, y_train), (bags_test, y_test) = musk1.load()
```
Also, the mnist_bags dataset has been created. The principal reason of creating this dataset is to have a good benchmark to evaluate the instances predictions. Or more specifically, if we can classify correctly a bag, can we detect which instance/s caused this classification? 
In the mnist_bags dataset, there are 3 different types of problems with their own dataset.

1) The bag 'b' is positive if the instance '7' is contained in 'b' <br/>
![mnist_bags](imgs/mnist_bags.png)
2) The bag 'b' is positive if the instance '2' and '3' are contained in 'b'  <br/>
![mnist_bags_2_and_3](imgs/mnist_bags_2_and_3.png)
3) The bag 'b' is positive if the instance '4' and '2' are located in consecutive instances in 'b'  <br/>
![mnist_bags_4_2](imgs/mnist_bags_4_2.png)

#### bag_representation
In multiple instance learning, bag representation is the technique that consists in obtaining a unique vector representing all the bag.
The classes implemented in the mil.bag_representation are used only for this finality.

- MILESMapping
- DiscriminativeMapping
- ArithmeticMeanBagRepresentation
- MedianBagRepresentation
- GeometricMeanBagRepresentation
- MinBagRepresentation
- MaxBagRepresentation
- MeanMinMaxBagRepresentation

**datasets**

```python
from mil.data.datasets import musk1

(bags_train, y_train), (bags_test, y_test) = musk1.load()
```

mnist bags dataset.

#### dimensionality_reduction
#### metrics
#### models
#### preprocessing
#### utils
#### validators
#### trainer

### Usage

```python

```

### Contributing
Pull requests are welcome. Priority things are on [To-do-list](#to-do-list). For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

### To-do-list
Pending tasks to do:
- [ ] Implement other algorithms, such as the SVM based ones.
- [ ] Make C/C++ extension of the APR algorithm to run faster.
- [ ] Make C/C++ extension of the MILESMapping algorithm to run faster.
- [ ] MILESMapping generates a simetric matrix of bag instance similarity, optimize it to only calculate half matrix and apply other possible optimizations to reduce time and space complexity.
- [ ] Implement get_positive_instances for MILES model.
- [ ] Implement Tuner class for hyperparameter tuning.
- [ ] Implement Callbacks for using on Trainer.
- [ ] Add one cycle learning rate to use on optimizers of KerasClassifiers models.
- [ ] On trainer, implement to get the best validation loss for calculating the metrics, right now when evaluating a model, the metrics are the ones from the last epoch.

### License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
- **[MIT license](http://opensource.org/licenses/mit-license.php)**
