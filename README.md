# Mini-Workloads for Deep Learning Experiments

This repo contains miniature deep learning workloads for use in experiments
and simple profiling.  Example scripts are provided to profile workloads using
the Intel VTune Profiler.

All code is under the Apache 2.0 License (see
https://www.apache.org/licenses/LICENSE-2.0), unless a source file lists
otherwise.


## Models Provided


### cnn-cifar10-tf2

A CNN model that runs over the CIFAR10 dataset, implemented in TensorFlow 2
Keras.  This model was derived from a standard TensorFlow 2.x example -- see
the model script for details.

A Horovod version is provided for experiments with simple distributed runs.


### cnn-cifar10-pytorch

A CNN model that runs over the CIFAR10 dataset, implemented in PyTorch.  This
model was translated from the cnn-cifar10-tf2 model


### simple-mnist-tf1

A simple NN which runs over the MNIST dataset, implemented in TF 1.1x.  This
model was derived from a CI test in the open-source TensorFlow nGraph project.


### simple-mnist-tf2

A simple NN which runs over the MNIST dataset, implemented in TF 2.x.  This
model was derived from a standard TensorFlow 2.x example -- see the model
script for details.


## Notes

For all models, a training script (without any TensorBoard profiling) is
provided which trains the model over a short period, then saves the model out.
The matching inference script uses the saved model, so the training script
must be run first.

Scripts are also provided that run training (and also inference, for some)
with TensorBoard profiling activated.  The intention is that you can diff the
scripts without profiling, and those with profiling, to see how the profiling
code was added.

Shell scripts are provided to run the Intel VTune Profiler using the models
with TensorBoard profiling added.  This collects both types of profiling
(model-level TensorBoard profiling and hardware-level VTune profilng)
simultaneously, so they can be correlated.  See the following article to see
how to use a VTune custom collector to incorporate TensorFlow timelines into
VTune's timeline display:

https://software.intel.com/content/www/us/en/develop/articles/profiling-tensorflow-workloads-with-intel-vtune-amplifier.html

Note that Horovod's timeline capabilities also generate a chrome::/tracking
compatible JSON file, and can be incorporated into VTune's timeline in a
similar fashion.
