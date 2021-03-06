# Early-Stop-On-Baseline
Custom EarlyStopping Callback for Keras (TF2+) to prioritize baseline and use patience after baseline was reached. 
According to: https://theailearner.com/tag/baseline-early-stopping-keras/, keras built-in EarlyStopping Callback waits a {patience} number of epochs until the model reaches the {baseline} value. If this does not happen, the training is stopped. 
I needed a way to train a model for many epochs, but stop the training after the {baseline} is reached and no better result appears after the {patience} number of epochs. 
The differences between EarlyStopping and EarlyStopOnBaseline:
difference | EarlyStopping | EarlyStopOnBaseline
--- | --- | ---
weights restoration | present | not implemented (yet)
mode choice | manual or auto | auto
metrics | built-in and custom | (val) loss / acc (tested on loss only)
verbosity | choice available | none to write home about

Requirements: Tensorflow 2+
