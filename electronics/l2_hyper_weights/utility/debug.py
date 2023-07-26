import os
import json

import tensorflow as tf
from tensorflow.python.client import timeline


class TFTimeline(object):

    def __init__(self, dump_file, batch_start_id=0, batch_end_id=0):
        self.dump_file = dump_file
        self.batch_start_id = batch_start_id
        self.batch_end_id = batch_end_id
        self.cur_batch = 0
        self.chrome_trace_dict = None

        self._END_ = False

    def _update(self):
        cur_trace_dict = json.loads(self.trace.generate_chrome_trace_format())
        if self.chrome_trace_dict is None:
            self.chrome_trace_dict = cur_trace_dict
        else:
            for event in cur_trace_dict['traceEvents']:
                if 'ts' in event:
                    self.chrome_trace_dict['traceEvents'].append(event)
        self.cur_batch += 1

    def _save(self, file_path=None):
        if self.chrome_trace_dict is None:
            return
        if not os.path.exists(os.path.split(self.dump_file)[0]):
            os.makedirs(os.path.split(self.dump_file)[0])
        with open(self.dump_file, 'w') as f:
            json.dump(self.chrome_trace_dict, f, indent=1)

    def collect(self, run_metadata):
        if self._END_:
            return
        if self.cur_batch < self.batch_start_id:
            self.cur_batch += 1
            return
        if not isinstance(run_metadata, tf.RunMetadata):
            raise TypeError('Invalid run_metadata')
        self.trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        if self.cur_batch <= self.batch_end_id:
            self._update()
        else:
            self._save()
            self._END_ = True
