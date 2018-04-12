#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import numpy as np
import chainer
from chainer import cuda
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.links import PixelwiseSoftmaxClassifier
from my_pixelwise_softmax_classifier import MyPixelwiseSoftmaxClassifier
from datasets.voc import voc_transformed
import pspnet
import pspnet_fine_custom
from poly import PolynomialShift

class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret
                                        
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='GPU id')
    parser.add_argument('-o','--out', type=str, default='result')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--batchsize', '-B', type=int, default=2)
    parser.add_argument('--val_batchsize', '-b', type=int, default=6)
    args = parser.parse_args()
    # Triggers
    
    key = "iteration"
    log_trigger = (100, 'iteration')
    validation_trigger = (1000, 'iteration')
    val2_trigger = (10000, 'iteration')
    end_trigger = (1000000, 'iteration')    
    
    # 1. dataset
    dataset_train = voc_transformed.VOCTransformedDataset("/home/naka/.chainer/dataset/pfnet/chainercv/voc/VOCdevkit/VOC2012/",split='train',rotate=True,fliplr=True,n_class=2,ignore_labels=False,crop_size=(473,473))
    dataset_valid = voc_transformed.VOCTransformedDataset("/home/naka/.chainer/dataset/pfnet/chainercv/voc/VOCdevkit/VOC2012/",split='val',n_class=2,ignore_labels=False,crop_size=(473,473))

    train_iter = chainer.iterators.MultiprocessIterator(dataset_train, batch_size=args.batchsize)
    val_iter = chainer.iterators.MultiprocessIterator(dataset_valid, batch_size=args.val_batchsize, repeat=False, shuffle=False)
    
    # 2. model

    model = pspnet_fine_custom.PSPNet(pretrained_model='voc2012')
    model = MyPixelwiseSoftmaxClassifier(model)
    
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  
    # 3. optimizer

    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0001))

    # Updater
    
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
     
    # Trainer
    
    trainer = training.Trainer(updater, end_trigger, out=args.out)
    trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu),
                                      trigger=validation_trigger)
    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(PolynomialShift(stop_trigger=trainer.stop_trigger,len_dataset = len(trainer.updater.get_iterator('main').dataset),batchsize = trainer.updater.get_iterator('main').batch_size))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss'], x_key=key,
            file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['validation/main/loss'], x_key=key,
            file_name='val_loss.png'))
        #trainer.extend(extensions.PlotReport(
            #['validation_1/main/miou'], x_key=key,
            #file_name='miou.png'))
    #trainer.extend(extensions.snapshot(filename='snapshot_'+key+'-{.updater.'+key+'}'), trigger=val2_trigger)
    trainer.extend(extensions.snapshot(filename="snapshot_"+key+'-{.updater.'+key+'}'),trigger=val2_trigger)
    trainer.extend(extensions.snapshot_object(
        model, filename='model_'+key+'-{.updater.'+key+'}'),
        trigger=validation_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time', 'lr',
         'main/loss', 'validation/main/loss', 'validation_1/main/miou',
         'validation_1/main/mean_class_accuracy',
         'validation_1/main/pixel_accuracy']),
        trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    #trainer.extend(SemanticSegmentationEvaluator(val_iter, model.predictor, ),trigger=validation_trigger)
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

if __name__ == '__main__':
    main()
