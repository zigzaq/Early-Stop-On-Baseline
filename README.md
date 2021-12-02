# Early-Stop-On-Baseline
Custom EarlyStopping Callback for Keras (TF2+) to prioritize baseline and use patience after baseline was reached. 
According to: https://theailearner.com/tag/baseline-early-stopping-keras/, keras built-in EarlyStopping Callback waits a {patience} number of epochs until the model reaches the {baseline} value. If this does not happen, the training is stopped. 
I needed a way to train a model for many epochs, but stop the training after the {baseline} is reached and no better result appears after the {patience} number of epochs. This is it.
The differences between EarlyStopping and EarlyStopOnBaseline:
Attempt | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
