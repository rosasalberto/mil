# Multiple Instance Learning (MIL) library for Python

---

## Table of Contents (Optional)

- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [Team](#team)
- [FAQ](#faq)
- [Support](#support)
- [Pending tasks](#pendingtasks)
- [License](#license)

---

### Features

### Pending tasks

- [ ] Implement other algorithms, such as the SVM based ones.
- [ ] Make C++ extension of the APR algorithm to run faster.
- [ ] Make C++ extension of the MILESMapping algorithm to run faster.
- [ ] MILESMapping generates a simetric matrix of bag instance similarity, optimize it to only calculate half matrix and apply other possible optimizations to reduce time and space complexity.
- [ ] Implement get_positive_instances for MILES model.
- [ ] Implement hyperparameter tuner class.
- [ ] Implement Callbacks for using on Trainer.
- [ ] Add one cycle learning rate to use on optimizers of KerasClassifiers models.
- [ ] On trainer, implement to get the best validation loss for calculating the metrics, right now when evaluating a model, the metrics are the ones from the last epoch.

### License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
- **[MIT license](http://opensource.org/licenses/mit-license.php)**
