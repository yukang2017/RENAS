import mxnet as mx
import logging
import os
import time
import lr_scheduler 
from metric_evl import DKAccuracy
from metric_evl import DKAccuracyTopK


def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))


def _get_warmup_lr_step_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    warmup = args.warmup if hasattr(args,'warmup') else False
    if warmup:
        warmup_step=args.warmup_epoch*epoch_size
        warmup_lr = args.warmup_lr
    else:
        warmup_step=0
        warmup_lr = lr

    warmup_linear = args.warmup_linear if hasattr(args,'warmup_linear') else False
    return (lr,lr_scheduler.WarmupMultiFactorScheduler(step=steps, factor=args.lr_factor, 
                   warmup = warmup, warmup_linear = warmup_linear, warmup_lr = warmup_lr, warmup_step = warmup_step))

def _get_lr_poly_scheduler(args, kv,adjust_num_update = False):
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    all_update = epoch_size*args.num_epochs
    if not args.max_num_update:
        max_num_update = all_update
    else:
        max_num_update =args.max_num_update
    if adjust_num_update:
        all_poly_update = max_num_update if max_num_update > all_update else all_update
    else:
        all_poly_update = max_num_update
    lr = args.lr
    power = args.power
    if not args.stop_factor_lr:
        stop_factor_lr = 1e-8
    else: 
        stop_factor_lr = args.stop_factor_lr
    return (lr, lr_scheduler.PolyScheduler(max_num_update=all_poly_update, power = power, stop_factor_lr=stop_factor_lr))

def _get_warmup_lr_poly_scheduler(args, kv,adjust_num_update = False):
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    all_update = epoch_size*args.num_epochs
    if not args.max_num_update:
        max_num_update = all_update
    else:
        max_num_update =args.max_num_update
    if adjust_num_update:
        all_poly_update = max_num_update if max_num_update > all_update else all_update
    else:
        all_poly_update = max_num_update
    lr = args.lr
    power = args.power
    if not args.stop_factor_lr:
        stop_factor_lr = 1e-8
    else: 
        stop_factor_lr = args.stop_factor_lr
    warmup = args.warmup if hasattr(args,'warmup') else False
    if warmup:
        warmup_step=args.warmup_epoch*epoch_size
        warmup_lr = args.warmup_lr
        warmup_end_lr = args.warmup_end_lr
    else:
        warmup_step=0
        warmup_lr = lr
        warmup_end_lr = lr
    warmup_linear = args.warmup_linear if hasattr(args,'warmup_linear') else False
    return (lr, lr_scheduler.WarmupPolyScheduler(max_num_update=all_poly_update, power = power, stop_factor_lr=stop_factor_lr,
                warmup = warmup, warmup_linear = warmup_linear, warmup_lr = warmup_lr, warmup_step = warmup_step))
def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    #dst_dir = os.path.dirname(args.model_prefix)
    #if not os.path.isdir(dst_dir):
    #    os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--model-type', type=str,default='nasnetAmobile',
                       help='model-type: nasnetAmobile or nasnetAlarge')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=120,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,default='30,60,90',
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--warmup', action ='store_true', help='warmup for lr scheduler')
    train.add_argument('--warmup-linear', action ='store_true', help='warmup linearor lr scheduler, if fasle, warmup keep a constant lr')
    train.add_argument('--warmup-lr', type=float, default=0.1,
                       help='warmup start learning rate')
    train.add_argument('--warmup-end-lr', type=float, default=0.1,
                       help='warmup end  learning rate')
    train.add_argument('--warmup-epoch', type=int, default=5,
                       help='warmup-step')
    train.add_argument('--lr-scheduler', type=str,default='step',
                       help='lr scheduler: step, poly, linearwarmup')
    train.add_argument('--max-num-update', type=int, default=450000,
                       help='max num of updaet')
    train.add_argument('--power', type=float, default=1.0,
                       help='power for polyscheduler') 
    train.add_argument('--stop-factor-lr', type=float, default=0.0001,
                       help='stop-factor-lr for poly') 
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.00004,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=256,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=5,
                       help='report the top-k accuracy. 0 means no report.')
    #train.add_argument('--label-smooth', action ='store_true', help='label smooth')
    train.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    train.add_argument('--dtype', type=str, default='float32',
                       help='precision: float32 or float16')
    return train

def fit(args, network, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    #import pudb; pudb.set_trace()
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)
    # data iterators
    (train, val) = data_loader(args, kv)
    #network().save("nasnet-symbol.json")
    #assert False
    if args.test_io:
        tic = time.time()
        for i, batch in enumerate(train):
            for j in batch.data:
                j.wait_to_read()
            if (i+1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, args.disp_batches*args.batch_size/(time.time()-tic)))
                tic = time.time()

        return

    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        #if sym is not None:
        #    assert sym.tojson() == network().tojson()

    # save model
    checkpoint = _save_model(args, kv.rank)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    #import pudb;pudb.set_trace()
    if args.lr_scheduler=='step':
        lr, lr_scheduler = _get_warmup_lr_step_scheduler(args, kv)
        #lr, lr_scheduler = _get_lr_scheduler(args, kv)
    elif args.lr_scheduler=='poly':
        
        lr, lr_scheduler = _get_warmup_lr_poly_scheduler(args, kv)
    else:
        raise ValueError("no support lr_scheduler type {}".format(args.lr_scheduler))
    # create model

    # create model
    model = mx.mod.Module(
        context       = devs,
        symbol        = network()
    )
    #import pudb; pudb.set_trace()
    lr_scheduler  = lr_scheduler
    optimizer_params = {
            'learning_rate': lr,
            'momentum' : args.mom,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler,
            'clip_gradient': 10}

    optimizers_using_momentum = ['sgd', 'dcasgd', 'nag']
    if not args.optimizer in optimizers_using_momentum:
        del optimizer_params['momentum']
        #del optimizer_params['multi_precision']
    if optimizers_using_momentum=='nadm':
            optimizer_params['beta1']=0.9
            optimizer_params['beta2']=0.999
            optimizer_params['schedule_decay']=0.004
    if optimizers_using_momentum=='rmsprop':
            optimizer_params['gamma1']=0.9
            optimizer_params['gamma2']=0.9
            optimizer_params['centered']=False

    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

   # monitor = mx.mon.Monitor(args.monitor, pattern=".*conv.*")
   # if args.network == 'alexnet':
   #     # AlexNet will not converge using Xavier
   #     initializer = mx.init.Normal()
   # else:
   #     initializer = mx.init.Xavier(
   #         rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),

    initializer = mx.initializer.Mixed(['.*fc.*','.*'],[mx.init.Normal(0.01),
           mx.init.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=2)])
    #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="avg",magnitude=2)
    # evaluation metrices
    #if args.top_k > 0:
    #    eval_metrics.append(mx.metric.create(DKAccuracyTopK(top_k=5, output_names=['predict'],label_names=['softmax'])))
    use_aux_head=args.use_aux
    if use_aux_head:
        eval_metrics =[DKAccuracy(indexes=[0,2],names=['aux_head_accuracy','backbone_accuracy'])]
        if args.top_k > 0:
            eval_metrics.append(mx.metric.create(DKAccuracyTopK(indexes=[0,2],
                 names=['aux_head','backbone'],top_k=5, output_names=['softmax_out_aux_head','softmax_out_backbone'],label_names=['softmax'])))
    else:
        eval_metrics =[DKAccuracy(indexes=[0,2],names=['backbone_accuracy'])]
        if args.top_k > 0:
            eval_metrics.append(mx.metric.create(DKAccuracyTopK(indexes=[0,2],
                names=['backbone'],top_k=5, output_names=['softmax_out_backbone'],label_names=['softmax'])))

    #use_aux_head=True
    #if use_aux_head:
    #    eval_metric_aux=[]
    #    eval_metric_backbone = []
    #    eval_metric=mx.metric.CompositeEvalMetric()
    #    eval_metric_aux.append(mx.metric.create(mx.metric.Accuracy('accuracy')))
    #    if args.top_k > 0:
    #        eval_metric_aux.append(mx.metric.create(mx.metric.TopKAccuracy(name='top_k_accuracy',top_k=args.top_k)))

    #    eval_metric_backbone.append(mx.metric.create(mx.metric.Accuracy('accuracy')))
    #    if args.top_k > 0:
    #        eval_metric_backbone.append(mx.metric.create(mx.metric.TopKAccuracy(name='top_k_accuracy', top_k=args.top_k)))
    #    for child_metric in [eval_metric_aux,eval_metric_backbone]:
    #        eval_metric.add(child_metric)
    #else:
    #    eval_metric =[]
    #    eval_metric.append(mx.metric.create(mx.metric.Accuracy('accuracy')))
    #    if args.top_k > 0:
    #        eval_metric.append(mx.metric.create(mx.metric.TopKAccuracy(name='top_k_accuracy', top_k=args.top_k)))

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]
    #import pudb;pudb.set_trace()
    # run
    model.fit(train,
        begin_epoch        = args.load_epoch if args.load_epoch else 0,
        num_epoch          = args.num_epochs,
        eval_data          = val,
        eval_metric        =  eval_metrics,
        kvstore            = kv,
        optimizer          = args.optimizer,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True,
        monitor            = monitor)
