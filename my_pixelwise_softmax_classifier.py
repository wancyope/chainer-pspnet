import chainer
from chainer import cuda
import chainer.functions as F
from chainer import reporter

import numpy as np


class MyPixelwiseSoftmaxClassifier(chainer.Chain):

    def __init__(self, predictor, ignore_label=-1, class_weight=None):
        super(MyPixelwiseSoftmaxClassifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
        self.ignore_label = ignore_label
        if class_weight is not None:
            self.class_weight = np.asarray(class_weight, dtype=np.float32)
        else:
            self.class_weight = class_weight

    def to_cpu(self):
        super(PixelwiseSoftmaxClassifier, self).to_cpu()
        if self.class_weight is not None:
            self.class_weight = cuda.to_cpu(self.class_weight)

    def to_gpu(self, device=None):
        super(MyPixelwiseSoftmaxClassifier, self).to_gpu(device)
        if self.class_weight is not None:
            self.class_weight = cuda.to_gpu(self.class_weight, device)

    def __call__(self, x, t):
        self.y = self.predictor(x)
        if chainer.config.train:
            self.aux_loss = F.softmax_cross_entropy(
                self.y[0], t, class_weight=self.class_weight,
                ignore_label=self.ignore_label)

            self.loss = F.softmax_cross_entropy(
                self.y[1], t, class_weight=self.class_weight,
                ignore_label=self.ignore_label)
        
            reporter.report({'loss': (self.aux_loss*0.4)+self.loss}, self)
            return (self.aux_loss*0.4) + self.loss
        else:
            self.loss = F.softmax_cross_entropy(
                self.y, t, class_weight=self.class_weight,
                ignore_label=self.ignore_label)
            reporter.report({'loss': self.loss}, self)
            return self.loss
        
