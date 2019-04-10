from __future__ import division
import mxnet as mx
import numpy as np


class DKAccuracy(mx.metric.EvalMetric):
    def __init__(self, indexes=[0, ], names=['accuracy', ]):
        self.names = names
        self.indexes = indexes
        self.sum_metric_list = [0.0, ] * len(self.names)
        self.num_inst_list = [0.0, ] * len(self.names)
        super(DKAccuracy, self).__init__('accuracy')
    
    def update(self, labels, preds):
        for i in range(len(self.indexes)):
            label=labels[0]
            pred_label = preds[self.indexes[i]]
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax_channel(pred_label)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            self.sum_metric_list[i] += (pred_label.flat == label.flat).sum()
            self.num_inst_list[i] += len(pred_label.flat)
    
    def reset(self):
        self.sum_metric_list = [0.0, ] * len(self.names)
        self.num_inst_list = [0.0, ] * len(self.names)

    def get(self):
        return self.names, [self.sum_metric_list[i] / self.num_inst_list[i] for i in range(len(self.sum_metric_list))]

class DKAccuracyTopK(mx.metric.EvalMetric):
    def __init__(self, top_k=1, indexes=[0,],names=['top_k_accuracy',],
                 output_names=None, label_names=None):
        self.top_k = top_k
        self.names = names
        self.indexes = indexes
        self.sum_metric_list = [0.0, ] * len(self.names)
        self.num_inst_list = [0.0, ] * len(self.names)
        super(DKAccuracyTopK, self).__init__('top_k_accuracy')

        assert(self.top_k > 1), 'Please use Accuracy if top_k is no more than 1'
        for i in range(len(self.indexes)):
            self.names[i] += '_%d' % self.top_k

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        for i in range(len(self.indexes)):
            #import pudb; pudb.set_trace()
            label=labels[0]
            pred_label = preds[self.indexes[i]]
            pred_label = np.argsort(pred_label.asnumpy().astype('float32'), axis=1)
            label = label.asnumpy().astype('int32')
            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric_list[i] += (pred_label.flat == label.flat).sum()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    self.sum_metric_list[i] += (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
                self.num_inst_list[i] += num_samples
    def reset(self):
        self.sum_metric_list = [0.0, ] * len(self.names)
        self.num_inst_list = [0.0, ] * len(self.names)
    def get(self):
        return self.names, [self.sum_metric_list[i] / self.num_inst_list[i] for i in range(len(self.sum_metric_list))]
