from nasnet_utils import *
import numpy as np
#from utils import get_ops_params
#from calops import *

def Dilate_conv2d(data,
                  out_channels,
                  kernel,
                  pad,
                  stride=(1,1),
                  bias=False,
                  name=None,
                  workspace=512):

    dilate = mx.sym.Activation(data=data,
                               act_type='relu',
                               name=name + '_relu')
    dilate = mx.sym.Convolution(data=dilate,
                                num_filter=out_channels,
                                kernel=kernel,
                                pad=pad,
                                stride=stride,
                                no_bias=False if bias else True,
                                #num_group=in_channels,
                                workspace=workspace,
                                name=name +'_conv2d')
    dilate = mx.sym.BatchNorm(data=dilate,
                              fix_gamma=fix_gamma,
                              eps=eps,
                              momentum=bn_mom,
                              use_global_stats=use_global_stats,
                              name=name+'_conv2d_bn')
    return dilate


def Asymmetric_conv2d(data,
                      in_channels,
                      out_channels,
                      kernel,
                      pad,
                      stride=(1,1),
                      bias=False,
                      bn_out=True,
                      act_out=True,
                      name=None,
                      workspace=512):
	#asymmetric1
    asym1 = mx.sym.Activation(data=data,
                              act_type='relu',
                              name=name+'_sep1_relu')

    asym1 = mx.sym.Convolution(data=asym1,
                               num_filter=in_channels,
                               kernel=(1,kernel),
                               pad=(0,pad),
                               stride=stride,
                               no_bias=False if bias else True,
                               #num_group=in_channels,
                               workspace=workspace,
                               name=name +'_conv2d_asym1')
    if bn_out:
        asym1 = mx.sym.BatchNorm(data=asym1,
                                 fix_gamma=fix_gamma,
                                 eps=eps,
                                 momentum=bn_mom,
                                 use_global_stats=use_global_stats,
                                 name=name+'_conv2d_asym1_bn')
    if act_out:
        asym1 = mx.sym.Activation(data=asym1,
                                  act_type='relu',
                                  name=name+'_conv2d_asym1_relu')
    #asymmetric2
    asym2 = mx.sym.Convolution(data=asym1,
                               num_filter=out_channels,
                               kernel=(kernel, 1),
                               stride=(1, 1),
                               pad=(pad, 0),
                               #num_group=1,
                               no_bias=False if bias else True,
                               workspace=workspace,
                               name=name+'_conv2d_asym2')

    asym2 = mx.sym.BatchNorm(data=asym2,
                             fix_gamma=fix_gamma,
                             eps=eps,
                             momentum=bn_mom,
                             use_global_stats=use_global_stats,
                             name=name + '_conv2d_asym2_bn')
    return asym2


def choose_layer(x, layer_type, num_filter, block_index, i, stride=(1,1)):
    if layer_type == 0:  # separable conv 3x3
        layer = Stack_separabels(data           =x,
                                 in_channels    =num_filter,
                                 out_channels   =num_filter,
                                 kernel         =(3, 3),
                                 pad            =(1, 1),
                                 stride         =stride,
                                 name           ='%s(%s)'% (str(block_index), '%02d'%(i + 1)))

    elif layer_type == 1:  # separable conv 5x5
        layer = Stack_separabels(data           =x,
                                 in_channels    =num_filter,
                                 out_channels   =num_filter,
                                 kernel         =(5, 5),
                                 pad            =(2, 2),
                                 stride         =stride,
                                 name           ='%s(%s)'% (str(block_index), '%02d'%(i + 1)))

    elif layer_type == 2:  # separable conv 7x7
        layer = Stack_separabels(data           =x,
                                 in_channels    =num_filter,
                                 out_channels   =num_filter,
                                 kernel         =(7, 7),
                                 pad            =(3, 3),
                                 stride         =stride,
                                 name           ='%s(%s)'% (str(block_index), '%02d'%(i + 1)))

    elif layer_type == 3:  # Avg_Pool 3x3
        layer = mx.sym.Pooling(data             =x,
                               kernel           =(3, 3),
                               stride           =stride,
                               pad              =(1, 1),
                               pool_type        ="avg",
                               name             ='%s(%s)avg_pooling'% (str(block_index), '%02d'%(i + 1)))

    elif layer_type == 4:  # Max_Pool 3x3
        layer = mx.sym.Pooling(data             =x,
                               kernel           =(3, 3),
                               stride           =stride,
                               pad              =(1, 1),
                               pool_type        ="max",
                               name             ='%s(%s)avg_pooling'% (str(block_index), '%02d'%(i + 1)))

    elif layer_type == 5:  # Conv 3x3
        layer = Dilate_conv2d(data           =x,
                              out_channels   =num_filter,
                              kernel         =(3, 3),
                              pad            =(1, 1),
                              stride         =stride,
                              name           ='%s(%s)'% (str(block_index), '%02d'%(i + 1)))

    elif layer_type == 6:  # Conv 1x7->7x1
        layer = Asymmetric_conv2d(data           =x,
                                  in_channels    =num_filter,
                                  out_channels   =num_filter,
                                  kernel         = 7,
                                  pad            = 3,
                                  stride         =stride,
                                  name           ='%s(%s)'% (str(block_index), '%02d'%(i + 1)))

    else:  # Identity layer_type == 7
        layer = mx.sym.identity(x) if stride == (1,1) else mx.sym.Pooling(data             =x,
                                                         kernel           =(3, 3),
                                                         stride           =stride,
                                                         pad              =(1, 1),
                                                         pool_type        ="max",
                                                         name             ='%s(%s)max_pooling'% (str(block_index), '%02d'%(i + 1)))
    return layer


def Cell(config_code,
         data_pre,
         data_cur,
         out_filters,
         cell_index=0,
         dim_match=True,
         bias=False,
         name=None,
         cell_type='Normal'):

    h1=Pre_layer_reduction(data_pre=data_pre,
                           data_cur=data_cur,
                           output_filters=out_filters,
                           stride=(1,1) if dim_match else (2,2),
                           bias=bias,
                           name=name+'_data_pre')

    h2=Squeeze_channels(data=data_cur,
                        output_channels=out_filters,
                        bias=False,
                        name=name+'_data_cur')

    layers = [h1,h2]

    connect_to_terminal = np.arange(len(config_code) / 2) + 2

    for i in range(0,len(config_code),2):
        type_left  = config_code[i][0]
        input_left = config_code[i][1]
        type_rigt  = config_code[i+1][0]
        input_rigt = config_code[i+1][1]

        if input_left>=i/2+2 or input_rigt>=i/2+2:
            print ('Wrong Block Config!')
            return None

        layer_left = choose_layer(x             =layers[input_left],
                                  layer_type    =type_left,
                                  num_filter    =out_filters,
                                  block_index   =cell_index,
                                  i             =i,
                                  stride        =(2,2) if cell_type is 'Reduction' and input_left<=1 else (1,1))

        layer_rigt = choose_layer(x             =layers[input_rigt],
                                  layer_type    =type_rigt,
                                  num_filter    =out_filters,
                                  block_index   =cell_index,
                                  i             =i+1,
                                  stride        =(2,2) if cell_type is 'Reduction' and input_rigt<=1 else (1,1))

        layers.append(layer_left+layer_rigt)
        connect_to_terminal = connect_to_terminal[np.where(connect_to_terminal != input_left)]
        connect_to_terminal = connect_to_terminal[np.where(connect_to_terminal != input_rigt)]

    #connect_to_terminal = np.append(connect_to_terminal, 0 if cell_type is 'Normal' else 3)          #small trick
    if cell_type is 'Normal':
        connect_to_terminal = np.append(connect_to_terminal, 0)

    if len(connect_to_terminal) == 1:
        output = layers[connect_to_terminal[0]]
    else:
        layers_con = [layers[int(a)] for a in connect_to_terminal]
        output = mx.sym.concat(*layers_con,name=name+'_out_concat')
    return output,data_cur


def renasnet_backbone(input_data,
                      normal_code,
                      reduct_code,
                      stem_filters,
                      num_cells,
                      num_conv_filters,
                      use_aux_head=True,
                      filter_scale=2,
                      num_reduction_layers=2,
                      is_training=True,
                      dense_drop_ratio=0.5,
                      net_type        = 'ImageNet',
                      skip_reduction_layer_input=0,
                      ):

    item_reduction=num_reduction_layers+1
    repeat_cell=int(num_cells/item_reduction)

    input_shape = (256, 3, 32, 32) if net_type=='CIFAR-10' else (256, 3, 224, 224)
    if net_type=='CIFAR-10':
        conv = mx.sym.Convolution(data=input_data, num_filter=stem_filters, kernel=(3, 3), stride=(1, 1), pad=(1, 1),no_bias=True, name='conv0')
        bn = mx.sym.BatchNorm(data=conv, name='conv0_bn', fix_gamma=False)
        #relu = mx.sym.Activation(data=bn, act_type='relu', name='conv0_relu')
        #state1
        p = None
        x = bn

    else:
        conv0_data = mx.sym.Convolution(data=input_data,
                                    num_filter=stem_filters,
                                    kernel=(3,3),
                                    stride=(2,2),
                                    no_bias= True,
                                    workspace=512,
                                    name='nasnet_conv0')

        conv0_data = mx.sym.BatchNorm(data=conv0_data,
                                      fix_gamma=fix_gamma,
                                      eps=eps,
                                      momentum=bn_mom,
                                      use_global_stats=use_global_stats,
                                      name='nasnet_conv0_bn')
        filters_scale_ratio=filter_scale**(-2)
        x, p = Cell(config_code=reduct_code,
                    data_pre=None,
                    data_cur=conv0_data,
                    out_filters=int(num_conv_filters * filters_scale_ratio),
                    cell_index=-1,
                    dim_match=True,
                    name='nasnet_stem0',
                    cell_type='Reduction')
        filters_scale_ratio = filter_scale ** (-1)
        x, p = Cell(config_code=reduct_code,
                    data_pre=p,
                    data_cur=x,
                    out_filters=int(num_conv_filters * filters_scale_ratio),
                    cell_index=0,
                    dim_match=False,
                    name='nasnet_stem1',
                    cell_type='Reduction')

    filters_scale_ratio = filter_scale ** (0)
    for cells in range(repeat_cell):
        x,p = Cell(config_code=normal_code,
                   data_pre   =p,
                   data_cur   =x,
                   out_filters=num_conv_filters*filters_scale_ratio,
                   cell_index =repeat_cell*(0)+cells+1,
                   dim_match  =False if cells == 0 else True,
                   #name       ='Normal%d'%(self.repeat_cell*(0)+i+1),
                   name       ='nomcell_stage1_{}'.format(cells),
                   cell_type  ='Normal')

    filters_scale_ratio = filter_scale ** (1)
    x,p = Cell(config_code=reduct_code,
               data_pre   =p,
               data_cur   =x,
               out_filters=num_conv_filters*filters_scale_ratio,
               cell_index =repeat_cell*(1)+1,
               name       ='redcell_stage1',
               cell_type  ='Reduction')

    #state2
    for cells in range(repeat_cell):
        x,p = Cell(config_code=normal_code,
                   data_pre   =p,
                   data_cur   =x,
                   out_filters=num_conv_filters*filters_scale_ratio,
                   cell_index =repeat_cell*(1)+cells+2,
                   dim_match  =False if cells == 0 else True,
                   #name       ='Normal%d'%(self.repeat_cell*(1)+i+2),
                   name='nomcell_stage2_{}'.format(cells),
                   cell_type  ='Normal')

    if use_aux_head and is_training:
        aux_head_data = Auxiliary_head(data=x,
                                       # shape=au_head_shape,
                                       out_channels=768,
                                       bias=False,
                                       name='nasnet_auxiliary_head')
        aux_head_data = mx.sym.Dropout(data=aux_head_data,
                                       p=dense_drop_ratio)

    filters_scale_ratio = filter_scale ** (2)
    x,p = Cell(config_code=reduct_code,
               data_pre   =p,
               data_cur   =x,
               out_filters=num_conv_filters*filters_scale_ratio,
               cell_index =repeat_cell*(2)+2,
               name       ='redcell_stage2',
               cell_type  ='Reduction')

    #state3
    for cells in range(repeat_cell):
        x,p = Cell(config_code=normal_code,
                   data_pre   =p,
                   data_cur   =x,
                   out_filters=num_conv_filters*filters_scale_ratio,
                   cell_index =repeat_cell*(2)+cells+3,
                   dim_match=False if cells == 0 else True,
                   #name       ='Normal%d'%(self.repeat_cell*(2)+i+3),
                   name='nomcell_stage3_{}'.format(cells),
                   cell_type  ='Normal')

    backbone_data = mx.sym.Activation(data=x,
                                      act_type='relu',
                                      name='nasnet_backbone_outdata_relu')
    backbone_data_shape = get_inshape(backbone_data, input_shape)
    global_pool_size = (int(backbone_data_shape[0][2]),int(backbone_data_shape[0][3]))
    #global_pool_size = (8,8)

    backbone_data = mx.sym.Pooling(data=backbone_data,
                                   global_pool=True,
                                   kernel=global_pool_size,
                                   pool_type='avg',
                                   name='nasnet_backbone_output_pooling')

    if dense_drop_ratio>0:
        backbone_data = mx.sym.Dropout(data=backbone_data,
                                       p=dense_drop_ratio)
    if is_training and use_aux_head:
        return aux_head_data, backbone_data
    elif is_training and not use_aux_head:
        return None, backbone_data
    else:
        return backbone_data


def get_symbol(model_type='renasnet-ImageNet',
               classes=1000,
               use_aux_head=True,
               is_training=True):
    data = mx.sym.Variable(name='data')
    # nasnet A
    normal_code = [[0, 1], [7, 1], [0, 0], [1, 1], [3, 1], [7, 0], [3, 0], [3, 0], [1, 0], [0, 0]]
    reduct_code = [[2, 0], [1, 1], [4, 1], [2, 0], [3, 1], [1, 0], [4, 1], [0, 2], [3, 2], [7, 3]]

    renasnetD = [[1, 0], [0, 1], [7, 1], [0, 1], [0, 0], [3, 0], [1, 1], [0, 0], [7, 2], [0, 2]] # 5.366M 580M  B

    normal_code = normal_code
    reduct_code = reduct_code

    if model_type=='renasnet-CIFAR10':
        classes = 10
        if is_training:
            aux_head_data, backbone_data=renasnet_backbone(normal_code=normal_code,
                                                           reduct_code=reduct_code,
                                                           net_type='CIFAR-10',
                                                           input_data=data,
                                                           stem_filters=96,
                                                           num_cells=18,
                                                           num_conv_filters=32,
                                                           use_aux_head=use_aux_head,
                                                           filter_scale=2,
                                                           num_reduction_layers=2,
                                                           is_training=is_training,
                                                           dense_drop_ratio=0.5,
                                                           skip_reduction_layer_input=0)
        else:
            backbone_data=renasnet_backbone(normal_code=normal_code,
                                            reduct_code=reduct_code,
                                            net_type='CIFAR-10',
                                            input_data=data,
                                            stem_filters=96,
                                            num_cells=18,
                                            num_conv_filters=32,
                                            use_aux_head=use_aux_head,
                                            filter_scale=2,
                                            num_reduction_layers=2,
                                            is_training=is_training,
                                            dense_drop_ratio=0.5,
                                            skip_reduction_layer_input=0)
    elif model_type=='renasnet-ImageNet':
        classes = 1000
        if is_training:
            aux_head_data, backbone_data = renasnet_backbone(normal_code=normal_code,
                                                             reduct_code=reduct_code,
                                                             net_type='ImageNet',
                                                             input_data=data,
                                                             stem_filters=32,
                                                             num_cells=12,
                                                             num_conv_filters=44,
                                                             use_aux_head=use_aux_head,
                                                             filter_scale=2,
                                                             num_reduction_layers=2,
                                                             is_training=is_training,
                                                             dense_drop_ratio=0.2,
                                                             skip_reduction_layer_input=0)
        else:
            backbone_data = renasnet_backbone(normal_code=normal_code,
                                              reduct_code=reduct_code,
                                              net_type='ImageNet',
                                              input_data=data,
                                              stem_filters=32,
                                              num_cells=12,
                                              num_conv_filters=44,
                                              use_aux_head=use_aux_head,
                                              filter_scale=2,
                                              num_reduction_layers=2,
                                              is_training=is_training,
                                              dense_drop_ratio=0.2,
                                              skip_reduction_layer_input=0)
    elif model_type=='renasnet-ImageNet-Large':
        classes = 1000
        if is_training:
            aux_head_data, backbone_data = renasnet_backbone(normal_code=normal_code,
                                                             reduct_code=reduct_code,
                                                             net_type='ImageNet',
                                                             input_data=data,
                                                             stem_filters=96,
                                                             num_cells=18,
                                                             num_conv_filters=168,
                                                             use_aux_head=use_aux_head,
                                                             filter_scale=2,
                                                             num_reduction_layers=2,
                                                             is_training=is_training,
                                                             dense_drop_ratio=0.2,
                                                             skip_reduction_layer_input=0)
        else:
            backbone_data = renasnet_backbone(normal_code=normal_code,
                                              reduct_code=reduct_code,
                                              net_type='ImageNet',
                                              input_data=data,
                                              stem_filters=96,
                                              num_cells=18,
                                              num_conv_filters=168,
                                              use_aux_head=use_aux_head,
                                              filter_scale=2,
                                              num_reduction_layers=2,
                                              is_training=is_training,
                                              dense_drop_ratio=0.2,
                                              skip_reduction_layer_input=0)
    else:
        raise ValueError('no support model_type:{}'.format(model_type))

    if is_training and use_aux_head:
        label = mx.symbol.var('softmax_label')
        smooth_label = mx.symbol.one_hot(label,
                                         depth=classes,
                                         on_value=0.9 if model_type=='renasnet-ImageNet' else 1.0,
                                         off_value=0.1 / (classes - 1) if model_type=='renasnet-ImageNet' else 0.0)
        smooth_label_aux = mx.symbol.one_hot(label,
                                             depth=classes,
                                             on_value=1.0,
                                             off_value=0)
        # aux_head
        aux_head_loss_weight = 0.4
        flatten_aux_head = mx.sym.Flatten(data=aux_head_data,
                                          name="flatten_aux_head")
        fc_aux_head = mx.symbol.FullyConnected(data=flatten_aux_head,
                                               num_hidden=classes,
                                               name='fc_aux_head')
        smooth_loss_aux_head = -mx.symbol.sum(mx.symbol.log_softmax(fc_aux_head) * smooth_label_aux)
        smooth_loss_aux_head = mx.symbol.MakeLoss(smooth_loss_aux_head,
                                                  grad_scale=aux_head_loss_weight)
        softmax_aux_head = mx.symbol.SoftmaxActivation(fc_aux_head, name='softmax_out_aux_head')
        # softmax_aux_head = mx.symbol.SoftmaxOutput(data=fc_aux_head,
        #                                           label=smooth_label,
        #                                           grad_scale=0.4,
        #                                           name='softmax_out_aux_head')

        # backbone
        flatten_backbone = mx.sym.Flatten(data=backbone_data,
                                          name="flatten_backbone")
        fc_backbone = mx.symbol.FullyConnected(data=flatten_backbone,
                                               num_hidden=classes,
                                               name='fc_backbone')

        smooth_loss_backbone = -mx.symbol.sum(mx.symbol.log_softmax(fc_backbone) * smooth_label)
        smooth_loss_backbone = mx.symbol.MakeLoss(smooth_loss_backbone)
        softmax_backbone = mx.symbol.SoftmaxActivation(fc_backbone, name='softmax_out_backbone')

        return mx.symbol.Group([mx.symbol.BlockGrad(softmax_aux_head),
                                smooth_loss_aux_head,
                                mx.symbol.BlockGrad(softmax_backbone),
                                smooth_loss_backbone])
        # softmax_backbone = mx.symbol.SoftmaxOutput(data=fc_backbone,
        #                                           label=smooth_label,
        #                                           grad_scale=1,
        #                                           name='softmax_out_backbone')
        # out= mx.symbol.Group([softmax_aux_head,softmax_backbone])

        # arg_shapes, out_shapes,aux_shapes = out.infer_shape(**{"data":(256,3,224,224)})
        # print arg_shapes
        # print out_shapes
        # return out
    elif is_training and not use_aux_head:

        label = mx.symbol.var('softmax_label')
        smooth_label = mx.symbol.one_hot(label,
                                         depth=classes,
                                         on_value=0.9 if model_type=='renasnet-ImageNet' else 1.0,
                                         off_value=0.1 / (classes - 1) if model_type=='renasnet-ImageNet' else 0.0)

        # backbone
        flatten_backbone = mx.sym.Flatten(data=backbone_data,
                                          name="flatten_backbone")
        fc_backbone = mx.symbol.FullyConnected(data=flatten_backbone,
                                               num_hidden=classes,
                                               name='fc_backbone')

        # softmax_backbone = mx.symbol.SoftmaxOutput(data=fc_backbone,
        #                                           label=smooth_label,
        #                                           grad_scale=1,
        #                                           name='softmax_out_backbone')
        # return softmax_backbone
        smooth_loss_backbone = -mx.symbol.sum(mx.symbol.log_softmax(fc_backbone) * smooth_label)
        smooth_loss_backbone = mx.symbol.MakeLoss(smooth_loss_backbone)
        softmax_backbone = mx.symbol.SoftmaxActivation(fc_backbone, name='softmax_out_backbone')

        return mx.symbol.Group([mx.symbol.BlockGrad(softmax_backbone),
                                smooth_loss_backbone])
    else:
        flatten_backbone = mx.sym.Flatten(data=backbone_data,
                                          name="flatten_backbone")
        fc_backbone = mx.symbol.FullyConnected(data=flatten_backbone,
                                               num_hidden=classes,
                                               name='fc_backbone')
        out = mx.symbol.SoftmaxActivation(data=fc_backbone,
                                          name='softmax_out')
        return out


'''
use_aux_head = False
is_training  = False
model_type = 'renasnet-ImageNet' # 'renasnet-CIFAR10' #
net = get_symbol(model_type=model_type,
                 use_aux_head=use_aux_head,
                 is_training=is_training)


#net.simple_bind(ctx=mx.cpu(0),data=(3, 224, 224))
if model_type=='renasnet-CIFAR10':
    data_shape = (1, 3, 32, 32)
elif model_type=='renasnet-ImageNet':
    data_shape = (1, 3, 224, 224)
else:
    data_shape = (1, 3, 331, 331)

_, _, millified_ops_total, millified_param_total = get_ops_params(net,data_shape)

print millified_ops_total, millified_param_total
'''
