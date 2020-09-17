# mil: multiple instance learning library for Python
---
When working on a research problem, I found myself with the [multiple instance learning (MIL)](https://en.wikipedia.org/wiki/Multiple_instance_learning) framework, which I found quite interesting and unique. After carefully reviewing the literature, I decided to try few of the algorithms on the problem I was working on, but for my surprise, there was no standard, easy, and updated MIL library for any programming language. So... here we are. <br/>
The mil library tries to achieve reproducible and productive research using the MIL framework.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [To-do-list](#to-do-list)
- [License](#license)

---
### Installation

### Features

The overall implementation tries to be as user-friendly as possible. That's why most of it constructed on top of [sklearn](https://scikit-learn.org/) and [tensorflow.keras](tensorflow.org/api_docs/python/tf/keras).

- [mil.bag_representation](#bag_representation)
- [mil.data](#data)
- [mil.dimensionality_reduction](#dimensionality_reduction)
- [mil.metrics](#metrics)
- [mil.models](#models)
- [mil.preprocessing](#preprocessing)
- [mil.utils](#utils)
- [mil.validators](#validators)
- [mil.trainer](#trainer)

#### bag_representation
#### data
#### dimensionality_reduction
#### metrics
#### models
#### preprocessing
#### utils
#### validators
#### trainer

### Contributing
Pull requests are welcome. Priority things are on [To-do-list](#to-do-list). For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

### To-do-list
Pending tasks to do:
- [ ] Implement other algorithms, such as the SVM based ones.
- [ ] Make C++ extension of the APR algorithm to run faster.
- [ ] Make C++ extension of the MILESMapping algorithm to run faster.
- [ ] MILESMapping generates a simetric matrix of bag instance similarity, optimize it to only calculate half matrix and apply other possible optimizations to reduce time and space complexity.
- [ ] Implement get_positive_instances for MILES model.
- [ ] Implement Tuner class for hyperparameter tuning.
- [ ] Implement Callbacks for using on Trainer.
- [ ] Add one cycle learning rate to use on optimizers of KerasClassifiers models.
- [ ] On trainer, implement to get the best validation loss for calculating the metrics, right now when evaluating a model, the metrics are the ones from the last epoch.

### License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
- **[MIT license](http://opensource.org/licenses/mit-license.php)**
