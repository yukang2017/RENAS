import mxnet as mx
fix_gamma = False
eps = 2e-5
bn_mom = 0.9
use_global_stats=False

input_shape=(256,3,224,224)
def get_inshape(variable_shape, input_shape):
    arg_shapes, out_shapes, aux_shapes = \
        variable_shape.infer_shape(**{"data": input_shape})
    return out_shapes


def Separable_conv2d(data,
                     in_channels,
                     out_channels,
                     kernel, 
                     pad, 
                     stride=(1,1),
                     bias=False,
                     bn_out=False,
                     act_out=False,
                     name=None,
                     workspace=512):
	#depthwise
    dw_out = mx.sym.Convolution(data=data,
                                num_filter=in_channels, 
                                kernel=kernel, 
                                pad=pad,
                                stride=stride,
                                no_bias=False if bias else True,
                                num_group=in_channels,
                                workspace=workspace, 
                                name=name +'_conv2d_depthwise')
    if bn_out:
        dw_out = mx.sym.BatchNorm(data=dw_out,
                                  fix_gamma=fix_gamma, 
                                  eps=eps,
                                  momentum=bn_mom,
                                  use_global_stats=use_global_stats,
                                  name=name+'_conv2d_depthwise_bn')
    if act_out:
        dw_out = mx.sym.Activation(data=dw_out, 
                                   act_type='relu', 
                                   name=name+'_conv2d_depthwise_relu')
    #pointwise
    pw_out = mx.sym.Convolution(data=dw_out, 
                                num_filter=out_channels,
                                kernel=(1, 1), 
                                stride=(1, 1),
                                pad=(0, 0),
                                num_group=1,
                                no_bias=False if bias else True, 
                                workspace=workspace, 
                                name=name+'_conv2d_pointwise')
    return pw_out


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


def Stack_separabels(data,
                     in_channels,
                     out_channels,
                     kernel,
                     pad,
                     stride,
                     bias=False,
                     name=None):

    sep1 = mx.sym.Activation(data=data,
                             act_type='relu',
                             name=name+'_sep1_relu')

    sep1 = Separable_conv2d(data=sep1,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel=kernel,
                            pad=pad,
                            stride=stride,
                            bias=bias,
                            bn_out=False,
                            act_out=False,
                            name=name +'_sep1_conv')

    sep1 = mx.sym.BatchNorm(data=sep1,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            name=name+'_sep1_bn')

    sep2 = mx.sym.Activation(data=sep1,
                            act_type='relu',
                            name=name+'_sep2_relu')
    sep2 = Separable_conv2d(data=sep2,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel=kernel,
                            pad=pad,
                            stride=(1,1),
                            bias=bias,
                            bn_out=False,
                            act_out=False,
                            name=name +'_sep2_conv')

    sep2 = mx.sym.BatchNorm(data=sep2,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            name=name+'_sep2_bn')

    return sep2
def Squeeze_channels(data,
                     output_channels,
                     bias=False,
                     name=None,
                     workspace=512):

    squeeze_data = mx.sym.Activation(data=data,
                                     act_type='relu',
                                     name=name+'_sequeeze_channels_relu')
    

    squeeze_data = mx.sym.Convolution(data=squeeze_data,
                                      num_filter=output_channels,
                                      kernel=(1, 1),
                                      stride=(1, 1),
                                      pad=(0, 0),
                                      num_group=1,
                                      no_bias=False if bias else True,
                                      workspace=workspace,
                                      name=name+'_sequeeze_channels_conv')

    squeeze_data = mx.sym.BatchNorm(data=squeeze_data,
                                    fix_gamma=fix_gamma,
                                    eps=eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    name=name+'_sequeeze_channels_bn')
    return squeeze_data


def Factorized_reduction(data,
                         output_filters, 
                         stride,
                         bias=False,
                         name=None,
                         workspace=512):
    assert stride[0]==stride[1]
    if stride[0]==1:
        return Squeeze_channels(data=data,
                                output_channels=output_filters,
                                bias=bias,
                                name=name,
                                workspace=workspace)
    elif stride[0]==2:
        assert output_filters%2 == 0

        factorized_data = mx.sym.Activation(data=data,
                                            act_type='relu',
                                            name=name+'_factorized_data_relu')

        # path1
        factorized_data_p1 = mx.symbol.Pooling(data = factorized_data,
                                               kernel=(1, 1),
                                               stride=stride,
                                               pool_type="avg",
                                               name=name +'_factorized_data_p1_pooling')
        factorized_data_p1 = mx.sym.Convolution(data=factorized_data_p1,
                                                num_filter=int(output_filters/2),
                                                kernel=(1, 1),
                                                stride=(1, 1),
                                                pad=(0, 0),
                                                num_group=1,
                                                no_bias=False if bias else True, 
                                                workspace=workspace, 
                                                name=name+'_factorized_data_p1_conv')
        #path2
        # pad

        factorized_data_p2 = mx.sym.pad(data=factorized_data,
                                        mode='constant',
                                        constant_value=0,
                                        pad_width=(0,0,0,0,0,1,0,1),
                                        name=name +'_factorized_data_p2_pad')


        factorized_data_p2 = mx.sym.slice(data=factorized_data_p2,
                                          begin=(None, None, 1, 1),
                                          end=(None, None, None, None),
                                          #step=(None, None, 1, 1),
                                          name=name+'_factorized_data_p2_crop')
        #arg_shapes, out_shapes,aux_shapes = factorized_data_p2.infer_shape(**{"data":(256,3,224,224)})
        #print arg_shapes
        #print out_shapes

        factorized_data_p2 = mx.sym.Pooling(data=factorized_data_p2,
                                            kernel=(1, 1),
                                            stride=stride,
                                            pool_type="avg",
                                            name=name +'_factorized_data_p2_pooling')

        factorized_data_p2 = mx.sym.Convolution(data=factorized_data_p2,
                                                num_filter=int(output_filters/2),
                                                kernel=(1, 1), 
                                                stride=(1, 1),
                                                pad=(0, 0),
                                                num_group=1,
                                                no_bias=False if bias else True, 
                                                workspace=workspace, 
                                                name=name+'_factorized_data_p2_conv')
        factorized_data_out = mx.sym.concat(factorized_data_p1, 
                                            factorized_data_p2, 
                                            dim=1,
                                            name=name+'_factorized_data_concat')

        factorized_data_out = mx.sym.BatchNorm(data=factorized_data_out,
                                               fix_gamma=fix_gamma,
                                               eps=eps,
                                               momentum=bn_mom,
                                               use_global_stats=use_global_stats,
                                               name=name+'_factorized_data_out_bn')
        return factorized_data_out

    else:
        raise ValueError("no support fatorized_type: {}")

def Pre_layer_reduction(data_pre,
                        data_cur,
                        output_filters,
                        stride,
                        bias=False,
                        name=None,
                        workspace=512):
    '''
    if data_pre is None:
        return mx.sym.identity(data_cur)
    '''

    if  data_pre is None:
        h1_shape = get_inshape(data_cur, input_shape)
        h1_out_channels = int(h1_shape[0][1])
        if h1_out_channels==output_filters:
            return data_cur
        else:
            return Squeeze_channels(data=data_cur,
                           output_channels=output_filters,
                           bias=False,
                           name=name+'_data_pre')
    else:
        return Factorized_reduction(data=data_pre,
                                    output_filters=output_filters,
                                    stride=stride,
                                    bias=bias,
                                    name=name,
                                    workspace=workspace)

def NormalCell(data_pre,
               data_cur,
               out_filters,
               dim_match=True,
               bias=False,
               name=None):


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
    # block2
    x_b1_left = Stack_separabels(data=h2,
                                 in_channels=out_filters,
                                 out_channels=out_filters,
                                 kernel=(5, 5),
                                 pad=(2, 2),
                                 stride=(1, 1),
                                 bias=bias,
                                 name=name + '_block1_right')
    x_b1_right = Stack_separabels(data=h1,
                                  in_channels=out_filters,
                                  out_channels=out_filters,
                                  kernel=(3, 3),
                                  pad=(1, 1),
                                  stride=(1, 1),
                                  bias=bias,
                                  name=name + '_block1_left')

    x_b1 = x_b1_left + x_b1_right
    # block2
    x_b2_left = Stack_separabels(data=h1,
                                 in_channels=out_filters,
                                 out_channels=out_filters,
                                 kernel=(5, 5),
                                 pad=(2, 2),
                                 stride=(1, 1),
                                 bias=bias,
                                 name=name + '_block2_left')
    x_b2_right = Stack_separabels(data=h1,
                                  in_channels=out_filters,
                                  out_channels=out_filters,
                                  kernel=(3, 3),
                                  pad=(1, 1),
                                  stride=(1, 1),
                                  bias=bias,
                                  name=name + '_block2_right')
    x_b2 = x_b2_left + x_b2_right
    # block3
    x_b3_left = mx.sym.Pooling(data=h2,
                               kernel=(3, 3),
                               stride=(1, 1),
                               pad=(1, 1),
                               pool_type="avg",
                               name=name + '_block3_left_pooling')
    x_b3_right = h1
    x_b3 = x_b3_left + x_b3_right

    # block4
    x_b4_left = mx.sym.Pooling(data=h1,
                               kernel=(3, 3),
                               stride=(1, 1),
                               pad=(1, 1),
                               pool_type="avg",
                               name=name + '_block4_left_pooling')
    x_b4_right = mx.sym.Pooling(data=h1,
                                kernel=(3, 3),
                                stride=(1, 1),
                                pad=(1, 1),
                                pool_type="avg",
                                name=name + '_block4_right_pooling')
    x_b4 = x_b4_left + x_b4_right

    #block5
    x_b5_left=Stack_separabels(data=h2,
                               in_channels=out_filters,
                               out_channels=out_filters,
                               kernel=(3,3),
                               pad=(1,1),
                               stride=(1,1),
                               bias=bias,
                               name=name+'_block5_left')
    x_b5_right=h2
    x_b5=x_b5_left+x_b5_right
    out_data=mx.sym.concat(*[h1,x_b1,x_b2,x_b3,x_b4,x_b5],name=name+'_out_concat')
    #out_data=mx.sym.concat(*[x_b5,x_b1,x_b3,x_b4,x_b2,h1],name=name+'_out_concat')
    return out_data, data_cur

def ReductionCell(data_pre,
                  data_cur,
                  out_filters,
                  stride=(2,2),
                  dim_match=False,
                  bias=False,
                  name=None):


        h1 = Pre_layer_reduction(data_pre=data_pre,
                             data_cur=data_cur,
                             output_filters=out_filters,
                             stride=(1,1) if dim_match else (2,2),
                             bias=bias,
                             name=name + '_data_pre')
        if data_pre is None:
            h1_shape=get_inshape(h1,input_shape)
            h1_out_channels=int(h1_shape[0][1])
        else:
            h1_out_channels=out_filters
        h2 = Squeeze_channels(data=data_cur,
                          output_channels=out_filters,
                          bias=False,
                          name=name + '_data_cur')


    #block1
        x_b1_left = Stack_separabels(data=h2,
                                     in_channels=out_filters,
                                     out_channels=out_filters,
                                     kernel=(5, 5),
                                     stride=stride,
                                     pad=(2, 2),
                                     bias=bias,
                                     name=name + '_block1_left')

        x_b1_right=Stack_separabels(data=h1,
                                    in_channels=h1_out_channels,
                                    out_channels=out_filters,
                                    kernel=(7,7),
                                    stride=stride,
                                    pad=(3,3),
                                    bias=bias,
                                    name=name+'_block1_right')

        x_b1=x_b1_left+x_b1_right
    #block2
        x_b2_left = mx.sym.Pooling(data=h2,
                                   kernel=(3, 3),
                                   stride=stride,
                                   pad=(1, 1),
                                   pool_type="max",
                                   name=name + '_block2_left_pooling')
        x_b2_right = Stack_separabels(data=h1,
                                      in_channels=h1_out_channels,
                                      out_channels=out_filters,
                                      kernel=(7, 7),
                                      stride=stride,
                                      pad=(3, 3),
                                      bias=bias,
                                      name=name + '_block2_right')
        x_b2 = x_b2_left + x_b2_right
    #block3
        x_b3_left = mx.sym.Pooling(data=h2,
                                   kernel=(3, 3),
                                   stride=stride,
                                   pad=(1, 1),
                                   pool_type="avg",
                                   name=name + '_block3_left_pooling')
        x_b3_right = Stack_separabels(data=h1,
                                      in_channels=h1_out_channels,
                                      out_channels=out_filters,
                                      kernel=(5, 5),
                                      stride=stride,
                                      pad=(2, 2),
                                      bias=bias,
                                      name=name + '_block3_right')
        x_b3=x_b3_left+x_b3_right

    #block4
        x_b4_left = mx.sym.Pooling(data=x_b1,
                                   kernel=(3, 3),
                                   stride=(1,1),
                                   pad=(1, 1),
                                   pool_type="avg",
                                   name=name + '_block4_left_pooling')
        x_b4_right=x_b2
        x_b4 = x_b4_left + x_b4_right

        # block4
        x_b5_left = Stack_separabels(data=x_b1,
                                     in_channels=out_filters,
                                     out_channels=out_filters,
                                     kernel=(3, 3),
                                     stride=(1, 1),
                                     pad=(1, 1),
                                     bias=bias,
                                     name=name + '_block5_right')


        x_b5_right = mx.sym.Pooling(data=h2,
                                    kernel=(3, 3),
                                    stride=stride,
                                    pad=(1, 1),
                                    pool_type="max",
                                    name=name + '_block5_left_pooling')

        x_b5 = x_b5_left + x_b5_right
        out_data = mx.sym.concat(*[x_b2, x_b3, x_b4, x_b5], name=name+'_out_concat')
        #out_data = mx.sym.concat(*[x_b2, x_b3, x_b5, x_b4], name=name+'_out_concat')
        return out_data, data_cur

def Auxiliary_head(data,
                    out_channels=768,
                    bias=False,
                    name=None):

        au_head_data = mx.sym.Activation(data=data,
                                         act_type='relu',
                                         name=name + '_au_head_data_relu')

        au_head_data=mx.sym.Pooling(data=au_head_data,
                                   kernel=(5, 5),
                                   stride=(3,3),
                                   pool_type="avg",
                                   name=name + '_au_head_pooling')

        au_head_data = mx.sym.Convolution(data=au_head_data,
                                    num_filter=128,
                                    kernel=(1,1),
                                    stride=(1,1),
                                    no_bias=False if bias else True,
                                    num_group=1,
                                    workspace=512,
                                    name=name + '_au_head_conv1x1')

        au_head_data = mx.sym.BatchNorm(data=au_head_data,
                                        fix_gamma=fix_gamma,
                                        eps=eps,
                                        momentum=bn_mom,
                                        use_global_stats=use_global_stats,
                                        name=name + '_au_head_conv1x1_bn')

        au_head_data = mx.sym.Activation(data=au_head_data,
                                         act_type='relu',
                                         name=name + '_au_head_conv1x1_relu')
        au_head_data_shape=get_inshape(au_head_data,input_shape)

        conv_shape=(int(au_head_data_shape[0][2]),
                    int(au_head_data_shape[0][3]))

        au_head_data = mx.sym.Convolution(data=au_head_data,
                                          num_filter=out_channels,
                                          kernel=conv_shape,
                                          stride=(1, 1),
                                          no_bias=False if bias else True,
                                          num_group=1,
                                          workspace=512,
                                          name=name + '_au_head_conv_output')

        au_head_data = mx.sym.BatchNorm(data=au_head_data,
                                        fix_gamma=fix_gamma,
                                        eps=eps,
                                        momentum=bn_mom,
                                        use_global_stats=use_global_stats,
                                        name=name + '_au_head_conv_output_bn')

        au_head_data = mx.sym.Activation(data=au_head_data,
                                         act_type='relu',
                                         name=name + '_au_head_conv_output_relu')
        return au_head_data


def Auxiliary_head(data,
                   out_channels=768,
                   bias=False,
                   name=None):
    au_head_data = mx.sym.Activation(data=data,
                                     act_type='relu',
                                     name=name + '_au_head_data_relu')
    au_head_data = mx.sym.Pooling(data=au_head_data,
                                  kernel=(5, 5),
                                  stride=(3, 3),
                                  pool_type="avg",
                                  name=name + '_au_head_pooling')
    au_head_data = mx.sym.Convolution(data=au_head_data,
                                      num_filter=128,
                                      kernel=(1, 1),
                                      stride=(1, 1),
                                      no_bias=False if bias else True,
                                      num_group=1,
                                      workspace=512,
                                      name=name + '_au_head_conv1x1')
    au_head_data = mx.sym.BatchNorm(data=au_head_data,
                                    fix_gamma=fix_gamma,
                                    eps=eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    name=name + '_au_head_conv1x1_bn')

    au_head_data = mx.sym.Activation(data=au_head_data,
                                     act_type='relu',
                                     name=name + '_au_head_conv1x1_relu')
    au_head_data_shape = get_inshape(au_head_data, input_shape)
    print au_head_data_shape

    conv_shape = (int(au_head_data_shape[0][2]),
                  int(au_head_data_shape[0][3]))
    au_head_data = mx.sym.Convolution(data=au_head_data,
                                      num_filter=out_channels,
                                      kernel=conv_shape,
                                      stride=(1, 1),
                                      no_bias=False if bias else True,
                                      num_group=1,
                                      workspace=512,
                                      name=name + '_au_head_conv_output')
    au_head_data = mx.sym.BatchNorm(data=au_head_data,
                                    fix_gamma=fix_gamma,
                                    eps=eps,
                                    momentum=bn_mom,
                                    use_global_stats=use_global_stats,
                                    name=name + '_au_head_conv_output_bn')

    au_head_data = mx.sym.Activation(data=au_head_data,
                                     act_type='relu',
                                     name=name + '_au_head_conv_output_relu')
    return au_head_data

