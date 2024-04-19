from __future__ import annotations

import braindecode.datasets
import numpy as np
from numpy import ndarray

from .dataset import IDataset, DatasetBaseConfig
import logging

from braindecode.datasets import create_from_X_y

import pyxdf

import mne


class XdfFileConfig(DatasetBaseConfig):
    """ Configuration of an XDF File

        This configuration object is used by
    """

    def __init__(self):
        super().__init__("XdfFile")
        self.filename: str = ""

class XdfFile(IDataset):
    def __init__(self, config: XdfFileConfig):
        super().__init__(config)
        self.__filename = config.filename

    def load_dataset(self) -> None:
        logging.info("Loading dataset: %s" % self.TYPE_NAME)

        eeg_stream, markers_stream = self.__parse_streams()

        if not eeg_stream:
            logging.error("No EEG stream found")
            return
        elif not markers_stream:
            logging.error("No markers stream found")
            return

        num_channels: int = int(eeg_stream['info']['channel_count'][0]) - 1
        sfreq: float = float(eeg_stream['info']['nominal_srate'][0])

        data: ndarray = eeg_stream["time_series"].T
        data = np.delete(data, -1, 0)  # remove last element since it's the count of samples
        channel_types = ["eeg" for x in range(num_channels)]
        info = mne.create_info(num_channels, sfreq, channel_types)
        mne_dataset: mne.io.RawArray = mne.io.RawArray(data, info)

        mne_dataset.set_annotations(self.__create_annotations_from_marker(markers_stream, eeg_stream["time_stamps"].min()))

        # TODO: set channel names??
        mne_dataset.plot(duration=1)
        # subject_id = 22
        # event_codes = [5, 6, 9, 10, 13, 14]
        # # This will download the files if you don't have them yet,
        # # and then return the paths to the files.
        # physionet_paths = mne.datasets.eegbci.load_data(
        #     subject_id, event_codes, update_path=False)
        #
        # # Load each of the files
        # parts = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto')
        #          for path in physionet_paths]
        # parts[0].plot()
        # X = [raw.get_data() for raw in parts]
        # y = event_codes
        # sfreq = parts[0].info["sfreq"]
        # ch_names = parts[0].info["ch_names"]
        # windows_dataset = create_from_X_y(
        #     X, y, drop_last_window=False, sfreq=sfreq, ch_names=ch_names,
        #     window_stride_samples=500,
        #     window_size_samples=500,
        # )

        self._raw_dataset = braindecode.datasets.create_from_mne_raw(
            [mne_dataset],
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            window_size_samples=500,
            window_stride_samples=500,
            drop_last_window=False,
            accepted_bads_ratio=0.072
            # descriptions=descriptions,
        )

    def __parse_streams(self) -> [dict, dict]:
        streams, header = pyxdf.load_xdf(self.__filename)
        logging.info("Streams present:")

        eeg_stream: dict = dict()
        markers_stream: dict = dict()
        for stream in streams:
            time_stamps: ndarray = stream['time_stamps']
            name: str = stream['info']['name'][0]
            if time_stamps.size == 0:
                logging.warning("Skipping stream %s, since there is no data in it" % name)
                continue

            stream_type: str = stream['info']['type'][0]

            logging.info("    Name              : %s" % name)
            logging.info("    Type              : %s" % stream_type)
            logging.info("    Num channels      : %s" % stream['info']['channel_count'][0])
            logging.info("    Sampling frequency: %s" % stream['info']['nominal_srate'][0])
            logging.info("-------------------------------------------------------")
            # logging.info("Loading dataset: %s" % self.TYPE_NAME)

            if stream_type == 'EEG':
                eeg_stream = stream
            elif stream_type == "markers":
                markers_stream = stream

        return eeg_stream, markers_stream

    def __create_annotations_from_marker(self, markers_stream: dict, start_time_eeg: float = 0) -> mne.Annotations:
        events: list[dict] = [{'1.0': 'Right hand', '-1.0': 'Left hand'},
                              {'1.0': 'Right foot', '-1.0': 'Left foot'},
                              {'1.0': 'A', '-1.0': 'B'},
                              {'1.0': 'C', '-1.0': 'D'},
                              {'1.0': 'E', '-1.0': 'F'},
                              {'1.0': 'G', '-1.0': 'H'},
                              {'1.0': 'I', '-1.0': 'J'},
                              {'1.0': 'K', '-1.0': 'L'}]
        get_event_description = lambda index, value: events[index].get(str(value))

        recorded_events: list[tuple] = list()

        old_value: list = [0.0 for x in range(markers_stream['time_series'][0].shape[0])]
        time_stamps = markers_stream['time_stamps'] - start_time_eeg
        # event_time_stamp: list = [0 for x in range(old_value)]
        event_statuses: list = [(False, 0) for x in range(markers_stream['time_series'][0].shape[0])]    # container if there is an event running, and its index in the onset, duration and description arrays
        for idx, data in enumerate(markers_stream['time_series']):
            changed_indexes: ndarray = np.nonzero(old_value != data)[0]
            for changed_index in changed_indexes:
                is_event_running = event_statuses[changed_index][0]
                new_event_description = get_event_description(changed_index, data[changed_index])
                if is_event_running:
                    # stop current event and check if there is a new one
                    event_index = event_statuses[changed_index][1]
                    current_event = list(recorded_events[event_index])
                    current_event[1] = time_stamps[idx] - current_event[0]
                    recorded_events[event_index] = tuple(current_event)
                    event_statuses[changed_index] = (False, 0)
                    #if there is no new event, we just
                    if new_event_description:
                        event: tuple = (time_stamps[idx], 0, new_event_description)
                        recorded_events.append(event)
                        event_statuses[changed_index] = (True, len(recorded_events) - 1)
                else:
                    # Store time stamp of when the event starts
                    # event_time_stamp[changed_index] = time_stamps[idx]
                    # Event with onset (start time), duration, description - Maybe create MneAnnotation type to facilitate.
                    # The duration will have to be updated, but set to 0 for its creation
                    event: tuple = (time_stamps[idx], 0, new_event_description)
                    recorded_events.append(event)
                    event_statuses[changed_index] = (True, len(recorded_events) - 1)

                # # If there was no event before, now it must have
                # if old_value[changed_index] == 0:
                #     # Store time stamp of when the event starts
                #     event_time_stamp[changed_index] = time_stamps[idx]
                #     # start_event
                #     # Event with onset, duration, description - Maybe create MneAnnotation type to facilitate.
                #     # The duration will have to be updated, but set to 0 for its creation
                #     event: tuple = (event_time_stamp[changed_index], 0, get_event_description(changed_index, data[changed_index]))
                #     np.append(recorded_events, event)
                #     # np.append(onset, )
                #     # np.append(duration, 0)  # just to "save" the spot in the array
                #     # np.append(description, )
                #
                # else:
                #     # It can have stopped the event or started a new one
                #     if data[changed_index] == 0:
                #         # end event
                #         pass
                #     else:
                #         # end event
                #         # start_event (transition from -1 to 1 or vice-versa)
                #         pass
                old_value = data

        onset: ndarray = ndarray(0, dtype=float)         # The starting time of annotations in seconds after orig_time
        duration: ndarray = ndarray(0, dtype=float)      # Durations of the annotations in seconds.
        description: ndarray = ndarray(0, dtype=str)   # Array of strings containing description for each annotation.
        for event in recorded_events:
            onset = np.append(onset, event[0])
            duration = np.append(duration, event[1])
            description = np.append(description, event[2])

        return mne.Annotations(onset, duration, description)
