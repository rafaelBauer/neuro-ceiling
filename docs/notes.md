
Workflow
Load dataset or sample data from EEG


## Notes
We will use supervised learning to learn the EEG &rarr; Actions decoder. First, we will use
publicly available and labeled datasets to train a NN.

Afterward, we try to train a NN using our own datasets, sampled with the cap.

The EEG will generate "noisy" action labels due to the noise in the EEG. The idea is to check
if the CEILing framework can be used with these noisy labels.

### EEG
#### Learning NN
We will first sample and store labeled datasets and have offline supervised learning of an
NN.

There can be two sources of data:
 - Braindecode public available datasets
 - Our own sampled dataset using the EEG cap + gamepad

-[ ] Do we need to push the EEG data to ROS?

#### Using NN to decode actions
We want to get a "live stream" of data (EEG) sampling, "decode" and action, and then
feed it to the Action-Motion policy from the CEILing.

#### Sampling from "Cap"
 - PC password: neuro
 - SW password: 0000
 - Sampling rate: 1 kHz
 - Amplifier: EEG64-CY-261
 - Raw data.

### Action &rarr; Motion
The second step is to use the CEILing framework to train Actions to Motion.
To perform the feedback we have:

 - Action &rarr; Motion
   - Evaluative feedback (human): ??
   - Corrective feedback (human): ??
   - Comparison feedback (human): ??

