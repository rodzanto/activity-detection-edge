# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from __future__ import absolute_import

import subprocess
import io
import os
import time
import json
import uuid
import mxnet as mx
import numpy as np
from mxnet import gluon,nd
from io import BytesIO
from datetime import datetime
import gluoncv
from gluoncv.data.transforms import video
from gluoncv.data import VideoClsCustom
from gluoncv.utils.filesystem import try_import_decord

import logging
import platform
import sys
from threading import Timer

# Setup logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Retrieving platform information to send from Greengrass Core
#my_platform = platform.platform()

### I3D MXNET MODEL CODE HERE:
#ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()
ctx = mx.cpu()
#HMDB51 classes
# ------------------------------------------------------------ #
classes = ['fall', 'no_fall']
dict_classes = dict(zip(range(len(classes)), classes))
# ------------------------------------------------------------ #
#Local temp folder Greengrass video:
model_path = '/home/pi/model'
inference_path = '/tmp/video.avi'

# Hosting methods                                              #
my_counter = 0

def model_fn(model_dir):
    print('Loading model...')
    print(ctx)
    symbol = mx.sym.load('%s/model-symbol.json' % model_dir)
    outputs = mx.symbol.softmax(data=symbol, name='softmax_label')
    inputs = mx.sym.var('data')
    net = gluon.SymbolBlock(outputs, inputs)
    net.load_parameters('%s/model-0000.params' % model_dir, ctx=ctx)
    return net

#transform function that uses json (s3 path) as input and output
#def transform_fn(net, data, input_content_type, output_content_type):
def transform_fn(net, inference_path):
    print('Running inference...')
    start = time.time()
    #data = json.loads(data)
    #video_data = read_video_data(data['S3_VIDEO_PATH'])
    video_data = read_video_data(inference_path)
    #print(time.time())
    video_input = video_data.as_in_context(ctx)
    probs = net(video_input.astype('float32', copy=False))
    #print(time.time())
    predicted = mx.nd.argmax(probs, axis=1).asnumpy().tolist()[0]
    probability = mx.nd.max(probs, axis=1).asnumpy().tolist()[0]

    probability = '{:.4f}'.format(probability)
    predicted_name = dict_classes[int(predicted)]
    total_prediction = time.time()-start
    total_prediction = '{:.4f}'.format(total_prediction)

    now = datetime.utcnow()
    time_format = '%Y-%m-%d %H:%M:%S %Z%z'
    now = now.strftime(time_format)
    output = now + " - Inference " + predicted_name + " prob. " + probability + " inf-time " + total_prediction
    print(output)

def read_video_data(s3_video_path, num_frames=32):
    """Read and preprocess video data from the S3 bucket."""
    #print('read and preprocess video data here ')
    download_path = s3_video_path
    video_list_path = '/tmp/video_list.txt'

    #Dummy duration and label with each video path
    video_list = '{} {} {}'.format(download_path, 10, 1)
    with open(video_list_path, 'w') as fopen:
        fopen.write(video_list)

    #Constants
    data_dir = '/tmp/'
    num_segments = 1
    new_length = num_frames
    new_step =1
    use_decord = True
    video_loader = True
    slowfast = False
    #Preprocessing params
    #The transformation function does three things: center crop the image to 224x224 in size, transpose it to num_channels,num_frames,height*width, and normalize with mean and standard deviation calculated across all ImageNet images.
    #Use the general gluoncv dataloader VideoClsCustom to load the data with num_frames = 32 as the length.
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    transform = video.VideoGroupValTransform(size=input_size, mean=mean, std=std)
    video_utils = VideoClsCustom(root=data_dir,
                                 setting=video_list_path,
                                 num_segments=num_segments,
                                 new_length=new_length,
                                 new_step=new_step,
                                 video_loader=video_loader,
                                 use_decord=use_decord,
                                 slowfast=slowfast)

    #Read for the video list
    video_name = video_list.split()[0]

    decord = try_import_decord()
    decord_vr = decord.VideoReader(video_name)
    duration = len(decord_vr)

    skip_length = new_length * new_step
    segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

    if video_loader:
        if slowfast:
            clip_input = video_utils._video_TSN_decord_slowfast_loader(video_name, decord_vr,
                                                                       duration, segment_indices, skip_offsets)
        else:
            clip_input = video_utils._video_TSN_decord_batch_loader(video_name, decord_vr,
                                                                    duration, segment_indices, skip_offsets)
    else:
        raise RuntimeError('We only support video-based inference.')

    clip_input = transform(clip_input)

    if slowfast:
        sparse_sampels = len(clip_input) // (num_segments * num_crop)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, input_size, input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (new_length, 3, input_size, input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if new_length == 1:
        clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case

    clip_input = nd.array(clip_input)

    #Cleanup temp files
    #os.remove(download_path)
    #os.remove(video_list_path)
    #os.system('rm {}'.format(download_path))
    #os.system('rm {}'.format(video_list_path))

    return clip_input

def greengrass_i3d():
    global my_counter
    global net

    if my_counter == 0:
        ###Load the MXNet model
        net = model_fn(model_path)
        my_counter = my_counter + 1
    else:
        ###Run the inference on the video
        transform_fn(net, inference_path)

    Timer(5, greengrass_i3d).start()
    return

greengrass_i3d()

def function_handler(event, context):
    return