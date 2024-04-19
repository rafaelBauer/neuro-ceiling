import logging
from typing import List

import braindecode

from neuroceiling.configuration import NeuroCeilingConfig, DatasetConfig
from neuroceiling.dataaquisition import IDataset, DatasetFactory

# Necessary for preprocessing
from braindecode.preprocessing import preprocess, Preprocessor, exponential_moving_standardize
from numpy import multiply

# Necessary for creating windows
from braindecode.preprocessing import create_windows_from_events
from braindecode.datasets import WindowsDataset, BaseConcatDataset

# Necessary to create model
import torch
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

# Necessary to train the model
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier

# Necessary to plot results
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# Necessary to plot Confusion Matrix
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix


def load_dataset(configuration: DatasetConfig) -> IDataset:
    # Load dataset
    _dataset: IDataset = DatasetFactory.get_dataset(configuration.dataset_config)
    _dataset.load_dataset()
    return _dataset


def preprocess_dataset(_dataset: IDataset):
    """
    Taken from https://github.com/braindecode/braindecode/blob/master/examples/model_building/plot_bcic_iv_2a_moabb_trial.py#L72

    Now we apply preprocessing like bandpass filtering to our dataset. You
    can either apply functions provided by
    `mne.Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`__ or
    `mne.Epochs <https://mne.tools/0.11/generated/mne.Epochs.html#mne.Epochs>`__
    or apply your own functions, either to the MNE object or the underlying
    numpy array.

    note::
       Generally, braindecode prepocessing is directly applied to the loaded
       data, and not applied on-the-fly as transformations, such as in
       PyTorch-libraries like
       `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`__.

    :param _dataset:
    :return:
    """

    low_cut_hz = 4.  # low-cut frequency for filtering
    high_cut_hz = 38.  # high-cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    # Factor to convert from V to uV
    factor = 1e6

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                     factor_new=factor_new, init_block_size=init_block_size)
    ]

    # Transform the data
    preprocess(_dataset.raw_dataset, preprocessors, n_jobs=-1)


def extract_compute_windows(_dataset: IDataset) -> WindowsDataset:
    """
    Taken from https://github.com/braindecode/braindecode/blob/a5a2b049652da88cd143a50b05e7b6f4a443f6a4/examples/model_building/plot_bcic_iv_2a_moabb_trial.py#L108

    # Now we extract compute windows from the signals, these will be the inputs
    # to the deep networks during training. In the case of trialwise
    # decoding, we just have to decide if we want to include some part
    # before and/or after the trial. For our work with this dataset,
    # it was often beneficial to also include the 500 ms before the trial.


    :param _dataset:
    :return:
    """
    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = _dataset.raw_dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in _dataset.raw_dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    return create_windows_from_events(
        _dataset.raw_dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )


def prepare_training_validation(_windows_dataset: WindowsDataset) -> tuple[BaseConcatDataset, BaseConcatDataset]:
    """
    Taken from https://github.com/braindecode/braindecode/blob/master/examples/model_building/plot_bcic_iv_2a_moabb_trial.py#L137
    :param _windows_dataset:
    :return:
    """
    splitted = _windows_dataset.split('session')
    return splitted['session_T'], splitted['session_E']


def create_model(n_classes: int, n_chans: int, input_window_samples: int, cuda: bool) -> torch.nn.Module:
    """
    Taken from https://github.com/braindecode/braindecode/blob/master/examples/model_building/plot_bcic_iv_2a_moabb_trial.py#L203C9-L203C9

    # Now we create the deep learning model! Braindecode comes with some
    # predefined convolutional neural network architectures for raw
    # time-domain EEG. Here, we use the shallow ConvNet model from [3]_. These models are
    # pure `PyTorch <https://pytorch.org>`__ deep learning models, therefore
    # to use your own model, it just has to be a normal PyTorch
    # `nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__.

    :param n_classes:
    :param cuda:
    :param n_chans:
    :param input_window_samples:
    :return:
    """
    if cuda:
        torch.backends.cudnn.benchmark = True
    # Set random seed to be able to roughly reproduce results
    # Note that with cudnn benchmark set to True, GPU indeterminism
    # may still make results substantially different between runs.
    # To obtain more consistent results at the cost of increased computation time,
    # you can set `cudnn_benchmark=False` in `set_random_seeds`
    # or remove `torch.backends.cudnn.benchmark = True`
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    _model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
    )

    # Display torchinfo table describing the model
    print(_model)

    # Send model to GPU
    if cuda:
        _model = _model.cuda()
    return _model


def train_model(_model: torch.nn.Module, batch_size: int, n_epochs: int, learning_rate: float, weight_decay: float,
                classes: list[int], _training_set: BaseConcatDataset,
                _validation_set: BaseConcatDataset, device: str) -> braindecode.classifier.EEGClassifier:
    """
    Taken from https://github.com/braindecode/braindecode/blob/master/examples/model_building/plot_bcic_iv_2a_moabb_trial.py#L203C9-L203C9

    # Now we will train the network! ``EEGClassifier`` is a Braindecode object
    # responsible for managing the training of neural networks. It inherits
    # from skorch `NeuralNetClassifier <https://skorch.readthedocs.io/en/stable/classifier.html#>`__,
    # so the training logic is the same as in `Skorch <https://skorch.readthedocs.io/en/stable/>`__.
    #


    ######################################################################
    # .. note::
    #    In this tutorial, we use some default parameters that we
    #    have found to work well for motor decoding, however we strongly
    #    encourage you to perform your own hyperparameter optimization using
    #    cross validation on your training data.
    #
    :return:
    """
    # We found these values to be good for the shallow network:
    # lr = 0.0625 * 0.01
    # weight_decay = 0

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001

    _clf = EEGClassifier(
        _model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(_validation_set),  # using valid_set for validation
        optimizer__lr=learning_rate,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
        classes=classes,
    )
    # Model training for the specified number of epochs. `Y` is None as it is
    # already supplied in the dataset.
    _ = _clf.fit(_training_set, y=None, epochs=n_epochs)

    return _clf


if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    dataset_config: DatasetConfig = DatasetConfig.load("run/DatasetConfig.json")
    dataset: IDataset = load_dataset(dataset_config)

    if not dataset.raw_dataset:
        logging.error('!!!! No raw dataset !!!!!!')
        exit()

    preprocess_dataset(dataset)

    windows_dataset: WindowsDataset = extract_compute_windows(dataset)

    training_set, validation_set = prepare_training_validation(windows_dataset)

    N_CLASSES = 4
    CLASSES = list(range(N_CLASSES))
    CUDA: bool = torch.cuda.is_available()
    model: torch.nn.Module = create_model(N_CLASSES, training_set[0][0].shape[0], training_set[0][0].shape[1], CUDA)

    BATCH_SIZE = 64
    N_EPOCHS = 10
    DEVICE = 'cuda' if CUDA else 'cpu'
    LEARNING_RATE = 0.0625 * 0.01
    WEIGHT_DECAY = 0
    clf: braindecode.classifier.EEGClassifier = (train_model(model, BATCH_SIZE, N_EPOCHS,
                                                             LEARNING_RATE, WEIGHT_DECAY, CLASSES,
                                                             training_set, validation_set, DEVICE))


    # plot
    # Extract loss and accuracy values for plotting from history object
    results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
    df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                      index=clf.history[:, 'epoch'])

    # get percent of misclass for better visual comparison to loss
    df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                   valid_misclass=100 - 100 * df.valid_accuracy)

    fig, ax1 = plt.subplots(figsize=(8, 3))
    df.loc[:, ['train_loss', 'valid_loss']].plot(
        ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
    ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    df.loc[:, ['train_misclass', 'valid_misclass']].plot(
        ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
    ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
    ax1.set_xlabel("Epoch", fontsize=14)

    # where some data has already been plotted to ax
    handles: list[Line2D] = [Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'),
                             Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid')]
    plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
    plt.tight_layout()

    ######################################################################
    # Plotting a  Confusion Matrix
    # ----------------------------
    #

    # generate confusion matrices
    # get the targets
    y_true = validation_set.get_metadata().target
    y_pred = clf.predict(validation_set)

    # generating confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)

    # add class labels
    # label_dict is class_name : str -> i_class : int
    label_dict = windows_dataset.datasets[0].window_kwargs[0][1]['mapping']
    # sort the labels by values (values are integer class labels)
    labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]

    # plot the basic conf. matrix
    confusion_matrix_fig = plot_confusion_matrix(confusion_mat, class_names=labels)

    confusion_matrix_fig.show()
    plt.show()
