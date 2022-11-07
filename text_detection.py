import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "CRAFT-pytorch"))
#sys.path.append(os.path.join(os.path.dirname(__file__), "Automated-objects-removal-inpainter"))
sys.path.append(os.path.join(os.path.dirname(__file__), "deep-text-recognition-benchmark"))
#sys.path.append('/content/CRAFT-pytorch')
#sys.path.append('/content/Automated-objects-removal-inpainter')


#########################################################################################
#########################################################################################
#########################################################################################
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from shapely.geometry import Point, Polygon

import ipyplot

#### Craft imports
# import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict


####### Craft Dataset

import re
import six
import math
import lmdb
import torch
import string

from natsort import natsorted
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


############ Object remover
#import glob
#import random
#import torchvision.transforms.functional as F
#from torch.utils.data import DataLoader
#from imageio import imread
#from skimage.feature import canny
#from skimage.color import rgb2gray, gray2rgb
#from src.utils import create_mask
#from src.segmentor_fcn import segmentor,fill_gaps





######## Edge Connect (Дорисовываем картинку из штрихов)
#from src.dataset import Dataset
#from src.models import EdgeModel, InpaintingModel
#from src.utils import Progbar, create_dir, stitch_images, imsave
#from torchvision import transforms


######## Object remover MAIN
#from shutil import copyfile
#from src.config import Config
#########################################################################################
#########################################################################################
#########################################################################################

#### Craft imports

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def create_craft_args(refine):

# --trained_model: pretrained model
# --text_threshold: text confidence threshold
# --low_text: text low-bound score
# --link_threshold: link confidence threshold
# --cuda: use cuda for inference (default:True)
# --canvas_size: max image size for inference
# --mag_ratio: image magnification ratio
# --poly: enable polygon type result
# --show_time: show processing time
# --test_folder: folder path to input images
# --refine: use link refiner for sentense-level dataset
# --refiner_model: pretrained refiner model

    args = argparse.Namespace()
    args.trained_model = 'weights/craft_mlt_25k.pth'
    # args.text_threshold = 0.6 # 0.7
    # args.low_text = 0.35 # 0.4
    # args.link_threshold = 0.6 # 0.4
    args.text_threshold = 0.6 # 0.7
    args.low_text = 0.35 # 0.4
    args.link_threshold = 0.7 # 0.4
    if torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    args.canvas_size = 1280
    args.mag_ratio = 1.5
    args.poly = False
    args.show_time = False
    args.test_folder = './data'
    args.refine = refine
    args.refiner_model = 'weights/craft_refiner_CTW1500.pth'
    return args

""" For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)

# result_folder = './result/'
# if not os.path.isdir(result_folder):
#     os.mkdir(result_folder)

def test_net(args, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


#def create_text_mask(args, image_array, debug=False):
    # # load net
    # net = CRAFT()     # initialize

    # if debug:
        # print('Loading weights from checkpoint (' + args.trained_model + ')')

    # if args.cuda:
        # net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    # else:
        # net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    # if args.cuda:
        # net = net.cuda()
        # net = torch.nn.DataParallel(net)
        # cudnn.benchmark = False

    # net.eval()

    # # LinkRefiner
    # refine_net = None
    # if args.refine:
        # from refinenet import RefineNet
        # refine_net = RefineNet()
        
        # if debug:
            # print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')

        # if args.cuda:
            # refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            # refine_net = refine_net.cuda()
            # refine_net = torch.nn.DataParallel(refine_net)
        # else:
            # refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        # refine_net.eval()
        # args.poly = True

    # t = time.time()

    # image = image_array

    # bboxes, polys, score_text = test_net(args, net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

    # final_bboxes = np.around(bboxes).astype(int)

    # final_polys = [np.around(x).astype(int).reshape(-1).tolist() for x in polys]
    # # final_polys = [np.around(x).astype(int) for x in polys]

    # # return final_bboxes, final_polys
    # return bboxes, polys, score_text

def create_text_mask(args, net, refine_net, image_array, debug=False):
    # # load net
    # net = CRAFT()     # initialize

    # if debug:
    #     print('Loading weights from checkpoint (' + args.trained_model + ')')

    # if args.cuda:
    #     net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    # else:
    #     net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    # if args.cuda:
    #     net = net.cuda()
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = False

    # net.eval()

    # # LinkRefiner
    # refine_net = None
    # if args.refine:
    #     from refinenet import RefineNet
    #     refine_net = RefineNet()
        
    #     if debug:
    #         print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')

    #     if args.cuda:
    #         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
    #         refine_net = refine_net.cuda()
    #         refine_net = torch.nn.DataParallel(refine_net)
    #     else:
    #         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    #     refine_net.eval()
    #     args.poly = True

    #args, net, refine_net = init_craft_networks(refiner=refine, debug=debug)

    t = time.time()

    image = image_array

    bboxes, polys, score_text = test_net(args, net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

    final_bboxes = np.around(bboxes).astype(int)

    final_polys = [np.around(x).astype(int).reshape(-1).tolist() for x in polys]
    # final_polys = [np.around(x).astype(int) for x in polys]

    # return final_bboxes, final_polys
    return bboxes, polys, score_text

##################################################################################################################
##################################################################################################################
##################################################################################################################


def display_image_boxes(image_array, boxes):
    import PIL.ImageDraw as ImageDraw
    import PIL.Image as Image
    mask = Image.new("L", (image_array.shape[1], image_array.shape[0]))
    draw = ImageDraw.Draw(mask)
    for i in range(len(boxes)):
        draw.polygon(boxes[i], outline=256, fill=256)
    display(mask)


def transform_bboxes_to_rectangles(bboxes):
    rectangles = [[[x.min(axis=0)[0], x.min(axis=0)[1]],
                    [x.max(axis=0)[0], x.min(axis=0)[1]],
                    [x.max(axis=0)[0], x.max(axis=0)[1]],
                    [x.min(axis=0)[0], x.max(axis=0)[1]]] for x in bboxes]
    return np.array(rectangles)

def create_cutted_images_list(image_array, rectangles):
    list_ = []
    print("rectangles=", rectangles)
    for i in range(len(rectangles)):
        # display(Image.fromarray(np.array(mask)))
        x = rectangles[i].astype(int)
        list_.append(image_array[
        x[1][1]:x[3][1], 
        x[0][0]:x[2][0], 
        :
        ])
        
    return list_

def get_image_mask_from_boxes(image_array, boxes):
    import PIL.ImageDraw as ImageDraw
    import PIL.Image as Image
    mask = Image.new("L", (image_array.shape[1], image_array.shape[0]))
    draw = ImageDraw.Draw(mask)
    for i in range(len(boxes)):
        draw.polygon(boxes[i], outline=256, fill=256)
    return np.array(mask)

def create_word_2_sentence_index(word_bboxes, sentence_bboxes):

    result = {}

    for w_idx in range(len(word_bboxes)):
        arr = np.array([])
        for s_idx in range(len(sentence_bboxes)):
            word_polygon = Polygon(word_bboxes[w_idx])
            sent_polygon = Polygon(sentence_bboxes[s_idx])
            share_intersection = word_polygon.intersection(sent_polygon).area/word_polygon.area
            arr = np.append(arr, share_intersection)
        
        result[w_idx] = arr.argmax()
        # print(arr.tolist())
    
    return list(result.items())

def create_word_2_sentence_index_sorted(w2s_idx, word_rectangles):
    new_idx = sorted(w2s_idx, key=lambda x: (x[1], word_rectangles[:, 0, 0][x[0]]))

    return new_idx

def firts_work_upper_case(final_recognition_array, case_sensivity_recognition_array):
    for index in range(len(final_recognition_array)):
        if case_sensivity_recognition_array[index][0][0].isupper() and not final_recognition_array[index][0][0].isupper():
            final_recognition_array[index] = (final_recognition_array[index][0][0].upper() + final_recognition_array[index][0][1:], final_recognition_array[index][1])  
    
    return final_recognition_array


##################################################################################################################
##################################################################################################################
##################################################################################################################


############################################
########## ХЗ может это и не нужно##########
############################################
class CustomsDataset(Dataset):
    def __init__(self, array_of_cutted_images, opt):
        self.opt = opt
        self.array_of_cutted_images = array_of_cutted_images
    def __len__(self):
        return len(self.array_of_cutted_images)
    def __getitem__(self, index):
        # (images, index_images) index_images = string/int - identifyer
        #return (self.array_of_cutted_images[index], index)
        if self.opt.rgb:
            img = Image.fromarray(self.array_of_cutted_images[index]).convert('RGB')  # for color image
        else:
            img = Image.fromarray(self.array_of_cutted_images[index]).convert('L')
        # print(f'Getted image for index {index}')
        return (img, str(index))

def recognition_pipeline(opt, word_cutted_images_list, debug=False):
    import string
    import argparse

    import torch
    import torch.backends.cudnn as cudnn
    import torch.utils.data
    import torch.nn.functional as F

    from utils import CTCLabelConverter, AttnLabelConverter
    from dataset import RawDataset, AlignCollate
    from model import Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    if debug:
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    if debug:
        print('Device %s' % device)

    # load model
    if debug:
        print('loading pretrained model from %s' % opt.saved_model)

    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    #demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_data = CustomsDataset(word_cutted_images_list, opt=opt)
    if debug:
        print('CustomsDataset init')
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    # print('demo_loader create')
    # predict
    model.eval()

    result = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            # print(image_path_list)
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            # print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                # print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
                result.append((pred, confidence_score.item()))
            log.close()
            return result

def create_word_recongnition_args(is_use_second_model=True, is_sensitive=False):
    opt = argparse.Namespace()
    opt.trained_model = 'weights/craft_mlt_25k.pth'
    opt.image_folder = "demo_image/" # NOT USED, DUMP
    opt.workers = 4
    opt.batch_size = 192
    opt.character = '0123456789abcdefghijklmnopqrstuvwxyz'
    # if is_use_second_model:
    #     opt.sensitive = True
    # else:
    #     opt.sensitive = False

    opt.sensitive = is_sensitive

    if is_use_second_model:
        if opt.sensitive:
            opt.saved_model = "weights/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth" # !!!
        else:
            opt.saved_model = "weights/TPS-ResNet-BiLSTM-Attn.pth" # !!!
    else:
        opt.saved_model = "weights/TPS-ResNet-BiLSTM-CTC.pth" # !!!
        # """ Data processing """
    opt.batch_max_length = 25
    opt.imgH = 32 
    opt.imgW = 100
    opt.rgb = False 
    opt.PAD = True
    # """ Model Architecture """

    opt.Transformation = 'TPS'
    opt.FeatureExtraction = 'ResNet'
    opt.SequenceModeling = 'BiLSTM'
    if is_use_second_model:
        opt.Prediction = "Attn"
    else:
        opt.Prediction = "CTC"
    opt.num_fiducial = 20
    opt.input_channel = 1
    opt.output_channel = 512
    opt.hidden_size = 256
    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    return opt

def create_final_recognition_result(recognition_result_first, recognition_result_second):
    final_result_list = []
    for i in range(len(recognition_result_first)):
        if recognition_result_first[i][1] > recognition_result_second[i][1]:
            final_result_list.append(recognition_result_first[i])
        else:
            final_result_list.append(recognition_result_second[i])
        #print(recognition_result_first[i][0], recognition_result_second[i][0])
    #print(final_result_list)
    return final_result_list

##################################################################################################################
##################################################################################################################
##################################################################################################################

######## Font size detection
#from transformers import FSMTTokenizer, FSMTForConditionalGeneration
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
import PIL.ImageDraw as ImageDraw

###################################################################
#####Перевод и текста и рисование текста###########################
###################################################################


def search_font_size(text, bbox, debug=False):
    width, height = bbox[2]-bbox[0]
    font_size = 1
    font = ImageFont.truetype("fonts/arial.ttf", font_size)
    font_width, font_height = font.getsize(text)
    if debug:
        print(width, height, font_width, font_height, font_size)
    while font_width < width and font_height < height:
        font_size += 1
        font = ImageFont.truetype("fonts/arial.ttf", font_size)
        font_width, font_height = font.getsize(text)
        if debug:
            print(width, height, font_width, font_height, font_size)
    font_size -= 1
    if debug:
        print(font_size)
        print(width, height, font_width, font_height, font_size)
    return ImageFont.truetype("fonts/arial.ttf", font_size)

# def compile_image(image, sentence_dict, sentence_bboxes):

#     image = image.copy()
#     draw = ImageDraw.Draw(image)

#     for idx, sentence in sentence_dict.items():
#         draw.text((sentence_bboxes[idx][0][0],sentence_bboxes[idx][0][1]), 
#                 sentence, 
#                 font=search_font_size(sentence, sentence_bboxes[idx]), 
#                 fill=(0, 0, 0), 
#                 stroke_width=2, 
#                 stroke_fill=(255,255,255))
#     return image
        

def create_sentence_dict(w2s_idx_sorted, recognized_word_list, sentence_bboxes):
    result = {}

    for el in w2s_idx_sorted:
        if el[1] in result.keys():
            result[el[1]]['list'].append((recognized_word_list[el[0]][0], recognized_word_list[el[0]][1], el[0]))
        else:
            result[el[1]] = {
                'list': [],
                'bbox': sentence_bboxes[el[1]]
            }
            result[el[1]]['list'].append((recognized_word_list[el[0]][0], recognized_word_list[el[0]][1], el[0]))
            #result[el[1]]['index_list'].append(el[0])

    return result

def translate_sentence(string, translator_model, translator_tokenizer):
    input_ids = translator_tokenizer.encode(string, return_tensors="pt")
    outputs = translator_model.generate(input_ids)
    decoded = translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

def translate_sentence_dict(dict_, translator_model, translator_tokenizer, debug=False):

    result = {}

    for key, value in dict_.items():
        if debug:
            print(key, value)
        result[key] = translate_sentence(' '.join(value), translator_model, translator_tokenizer)
    return result

def bounds_to_bbox(x):
    return np.array([[x[0], x[1]], [x[2], x[1]], [x[2], x[3]], [x[0], x[3]]], dtype='float32')

def display_image_boxes(image_array, boxes):
    import PIL.ImageDraw as ImageDraw
    import PIL.Image as Image
    mask = Image.new("L", (image_array.shape[1], image_array.shape[0]))
    draw = ImageDraw.Draw(mask)
    font = ImageFont.truetype("fonts/arial.ttf", 12)
    for i in range(len(boxes)):
        draw.polygon(boxes[i], outline=256, fill=256)
        draw.text((boxes[i][0][0],boxes[i][0][1]), 
            text=str(i), 
            font=font)
    display(mask)

# TODO: РЕФАКТОРИНГ
# Тут скорее всего нужен рефакторинг т.к. не для всех будет работать правило
# Объединение по минимальной дистанции 10, скорее всего нужно будет смотреть на "размер ширкфат"
def create_paragraph_bboxes(sentence_bboxes
                            ):
    bboxes = sorted(sentence_bboxes,key=lambda x: x[0][0])
    paragraph_bboxes = []
    counter = 0
    # print('counter in ', counter)
    indexes = {x: 0 for x in list(range(len(sentence_bboxes)))}
    for num_master, box_master in enumerate(bboxes):
        for num_slave, box_slave in enumerate(bboxes):
            if indexes[num_master] == 0 and indexes[num_slave] == 0 and num_master != num_slave:
                # print('master ', num_master, 'slave ', num_slave)
                poly_master = Polygon(box_master)
                poly_slave = Polygon(box_slave)
                bounds_master = poly_master.bounds
                bounds_slave = poly_slave.bounds
                min_distance = poly_master.distance(poly_slave)

                # TODO: Сделать зависимость от размера шрифта
                if bounds_slave[0] < bounds_master[3] and min_distance <=10:
                    # print('append')
                    paragraph_bboxes.append(bounds_to_bbox(cascaded_union([poly_master, poly_slave]).bounds))
                    counter += 1
                    indexes[num_master] = 1
                    indexes[num_slave] = 1
        if indexes[num_master] == 0:
            paragraph_bboxes.append(box_master)
    # print('counter out ', counter)
    if counter == 0:
        return sentence_bboxes
    else:
        return create_paragraph_bboxes(paragraph_bboxes)

def create_word_2_sentence_index(word_bboxes, sentence_bboxes):

    result = {}

    for w_idx in range(len(word_bboxes)):
        arr = np.array([])
        for s_idx in range(len(sentence_bboxes)):
            word_polygon = Polygon(word_bboxes[w_idx])
            sent_polygon = Polygon(sentence_bboxes[s_idx])
            share_intersection = word_polygon.intersection(sent_polygon).area/word_polygon.area
            arr = np.append(arr, share_intersection)
        
        result[w_idx] = arr.argmax()
        # print(arr.tolist())
    
    return list(result.items())

def create_sentence_2_paragraph_index_sorted(sent2para_idx, sentence_bboxes):
    new_idx = sorted(sent2para_idx, key=lambda x: (x[1], sentence_bboxes[:, 0, 1][x[0]]))

    return new_idx

def create_paragraph_dict(sent2para_index_sorted, sentence_dict, paragraph_bboxes):

    result = {}

    for el in sent2para_index_sorted:
        if el[1] in result.keys():
            result[el[1]]['list'] = result[el[1]]['list'] + [("\n", 1.0, -1)] + sentence_dict[el[0]]['list']
            result[el[1]]['splits'] += 1
        else:
            result[el[1]] = {
                'list': sentence_dict[el[0]]['list'],
                'splits': 1,
                'bbox': paragraph_bboxes[el[1]].tolist()
            }
    return result

# def translate_sentence(string):
#     input_ids = tokenizer.encode(string, return_tensors="pt")
#     outputs = model.generate(input_ids)
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return decoded

# def translate_paragraph_dict(dict_, translator_model, translator_tokenizer, debug=False):

    # result = {}

    # for key in dict_.keys():
        # result[key] = translate_sentence(' '.join(dict_[key]['list']), translator_model, translator_tokenizer)
        # paragraph_array = result[key].split(' ')
        # step = 0 if dict_[key]['splits']==0 else math.ceil(len(paragraph_array)/dict_[key]['splits'])
        # if debug:
            # print('step ', step)
        # shuffle = 0
        # if step > 0:
            # for i in range(step, len(paragraph_array)+len(result[key])%step, step):
                # if debug:
                    # print(i)
                # paragraph_array.insert(i+shuffle, '\n')
                # shuffle += 1
        # result[key] = ' '.join(paragraph_array)
    # return result

def search_font_size(image, text, bbox, debug=False):
    if debug:
        print(bbox)
    width, height = bbox[2]-bbox[0]
    font_size = 1
    font = ImageFont.truetype("fonts/arial.ttf", font_size)
    # font_width, font_height = font.getsize(text)
    font_width, font_height = ImageDraw.ImageDraw(image).multiline_textsize(text, font=font)
    if debug:
        print('width {}, height {}, font_width {}, font_height {}, font_size {}'.format(width, height, font_width, font_height, font_size))
    while font_width < width :# and font_height < height:
        font_size += 1
        font = ImageFont.truetype("fonts/arial.ttf", font_size)
        # font_width, font_height = font.getsize(text)
        font_width, font_height = ImageDraw.ImageDraw(image).multiline_textsize(text, font=font)
        if debug:
            print('planned font size ', font.getsize(text))
            print('width {}, height {}, font_width {}, font_height {}, font_size {}'.format(width, height, font_width, font_height, font_size))
    font_size -= 1
    if debug:
        print('out font size ', font_size)
        print('width {}, height {}, font_width {}, font_height {}, font_size {}'.format(width, height, font_width, font_height, font_size))
    return ImageFont.truetype("fonts/arial.ttf", font_size)

def compile_image(image, sentence_dict, sentence_bboxes, debug):

    image = image.copy()
    draw = ImageDraw.Draw(image)

    for idx, sentence in sentence_dict.items():
        # draw.polygon(sentence_bboxes[idx], outline=256, fill=256)
        draw.text((sentence_bboxes[idx][0][0],sentence_bboxes[idx][0][1]), 
                sentence, 
                font=search_font_size(image, sentence, sentence_bboxes[idx], debug), 
                fill=(0, 0, 0), 
                stroke_width=2, 
                stroke_fill=(255,255,255))
    return image


##################################################################################################################
##################################################################################################################
##################################################################################################################

def init_craft_networks(refiner=False, debug=False):
    args = create_craft_args(refine=refiner)

    # load net
    net = CRAFT()  # initialize

    if debug:
        print('Loading weights from checkpoint (' + args.trained_model + ')')

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()

        if debug:
            print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')

        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    return args, net, refine_net

##################################################################################################################
##################################################################################################################
##################################################################################################################

#def pipeline(image_link, model_isr, model_translator, tokenizer_translator, font, debug=True):

def pipeline(
            image_link,
            text_detection_craft_args_refine, 
            text_detection_craft_net_refine, 
            text_detection_refiner_craft_net_refine,
            text_detection_craft_args_not_refine, 
            text_detection_craft_net_not_refine, 
            text_detection_refiner_craft_net_not_refine,
            debug=True
            ):

    image_path = image_link
    image_file_name = os.path.basename(image_path)
    image_pil = Image.open(BytesIO(requests.get(image_path).content))
    source_image_for_output = image_pil.copy() # Исходная картинка которую мы подадим на выход для сравнения
    if debug:
        display('downloaded image', image_pil)
    image_array = np.array(image_pil)

    image_width = image_array.shape[1]
    image_height = image_array.shape[0]
    if debug:
        print("Image width = ", image_width, "Image hight = ", image_height)

    #args = create_craft_args(refine=False)
    #word_bboxes, word_polys, word_score_text = create_text_mask(args, image_array)

    #args = create_craft_args(refine=True)
    #sentence_bboxes, sentence_polys, sentence_score_text = create_text_mask(args, image_array)

    word_bboxes, word_polys, word_score_text = create_text_mask(
        text_detection_craft_args_not_refine, 
        text_detection_craft_net_not_refine, 
        text_detection_refiner_craft_net_not_refine,
        image_array
        )
    sentence_bboxes, sentence_polys, sentence_score_text = create_text_mask(
        text_detection_craft_args_refine, 
        text_detection_craft_net_refine, 
        text_detection_refiner_craft_net_refine,
        image_array
        )

    mask_array_from_words = get_image_mask_from_boxes(image_array, word_bboxes)

    word_rectangles = transform_bboxes_to_rectangles(word_bboxes)
    word_cutted_images_list = create_cutted_images_list(image_array, word_rectangles)

    if debug:
        print('word_cutted_images_list')
        for i in range(len(word_cutted_images_list)):
            display(Image.fromarray(word_cutted_images_list[i]))

        sent_rectangles = transform_bboxes_to_rectangles(sentence_bboxes)
        sent_cutted_images_list = create_cutted_images_list(image_array, sent_rectangles)
        print('sent_cutted_images_list')
        for i in range(len(sent_cutted_images_list)):
            display(Image.fromarray(sent_cutted_images_list[i]))


    image_data_dictionary = {
        "image_path": image_path,
        "image_file_name": image_file_name,
        "image_array": image_array.copy(),
        "image_width": image_width,
        "image_height": image_height, 
        "word_bboxes": word_bboxes,
        "word_rectangles": word_rectangles,
        "mask_array_from_words": mask_array_from_words
    }
    image_data_list = [image_data_dictionary]

    w2s_idx = create_word_2_sentence_index(word_bboxes, sentence_bboxes)
    if debug:
        print("w2s_idx=", w2s_idx)
        print("word_rectangles=", word_rectangles)

    w2s_idx_sorted = create_word_2_sentence_index_sorted(w2s_idx, word_rectangles)
    if debug:
        print("w2s_idx_sorted=", w2s_idx_sorted)

    opt = create_word_recongnition_args(is_use_second_model=False)
    recognition_result_1 = recognition_pipeline(opt, word_cutted_images_list, debug)

    if debug:
        print('recognition_result_1', recognition_result_1)
    
    opt = create_word_recongnition_args(is_use_second_model=True, is_sensitive=False)
    recognition_result_2 = recognition_pipeline(opt, word_cutted_images_list, debug)
    # Нужно для этой модели снизить вероятность у распознования
    recognition_result_2 = [(item[0], item[1]*0.7) for item in recognition_result_2]

    if debug:
        print('recognition_result_2', recognition_result_2)


    opt = create_word_recongnition_args(is_use_second_model=True, is_sensitive=True)
    recognition_result_sensitive = recognition_pipeline(opt, word_cutted_images_list, debug)

    if debug:
        print('recognition_result_sensitive', recognition_result_sensitive)

    if debug:
        print('recognition_result_1', recognition_result_1)
        print('recognition_result_2', recognition_result_2)
        print('recognition_result_sensitive', recognition_result_sensitive)

    # Из трех моделей распознования слов выбираем с максимальными шансами
    final_recognition_result = create_final_recognition_result(recognition_result_1, recognition_result_sensitive)
    final_recognition_result = create_final_recognition_result(final_recognition_result, recognition_result_2)
    final_recognition_result = firts_work_upper_case(final_recognition_result, recognition_result_sensitive)

    if debug:
        print('final_recognition_result', final_recognition_result)

    recognized_word_list = [x[0] for x in final_recognition_result]

    #sentence_dict = create_sentence_dict(w2s_idx_sorted, [x[0] for x in final_recognition_result])
    sentence_dict = create_sentence_dict(w2s_idx_sorted, final_recognition_result, sentence_bboxes)

    if debug:
        print('sentence_dict', sentence_dict)

    # # Удаление текста, mode это алгоритм который удаляет, второй это комбинированный из двух алгоритмов
    # main(image_data_list, mode=2)

    # image_super_resolution_processing(model_isr, image_data_list)

    # image_with_deleted_text = Image.fromarray(image_data_list[0]["image_with_deleted_text"])

    paragraph_bboxes = create_paragraph_bboxes(sentence_bboxes)
    if debug:
        print('paragraph_bboxes=', paragraph_bboxes)

    sent2para_index = create_word_2_sentence_index(sentence_bboxes, paragraph_bboxes)

    sent2para_index_sorted = create_sentence_2_paragraph_index_sorted(sent2para_index, sentence_bboxes)

    paragraph_dict = create_paragraph_dict(sent2para_index_sorted, sentence_dict, paragraph_bboxes)

    if debug:
        print('paragraph_dict', paragraph_dict)

    if debug:
        print("word_bboxes", word_bboxes)
        print("word_rectangles", word_rectangles)

    #paragraph_dict_translated = translate_paragraph_dict(paragraph_dict, model_translator, tokenizer_translator)

    #if debug:
    #    print('paragraph_dict_translated', paragraph_dict_translated)

    #compiled_image = compile_image(image_with_deleted_text, paragraph_dict_translated, paragraph_bboxes, debug)

    #return source_image_for_output, compiled_image
    return paragraph_dict

##################################################################################################################
##################################################################################################################
##################################################################################################################


# def test_remover_func():
    # #from google.colab import output

    # input_image_url = 'https://img-9gag-fun.9cache.com/photo/axMNd31_460s.jpg' #@param {type:"string"}

    # image_path = input_image_url
    # image_file_name = os.path.basename(image_path)

    # ##
    # #output = self.postprocess(outputs_merged)[0]
    # #path = os.path.join(self.results_path, name)
    # #if self.debug:
        # #print(index, name)

    # #imsave(output, path)
    # #os.path.join(self.results_path, fname + '_edge.' + fext)


    # if not os.path.exists('./results_images'):
        # os.makedirs('./results_images')

    # if input_image_url is not None and input_image_url !='':
        # # source_image, output_image = pipeline(input_image_url, model_isr, model_translator, tokenizer_translator, font, debug=False)

        # # Init CraftNets
        # craft_args, craft_net, refiner_craft_net = init_craft_networks(refiner=False, debug=False)
        # edge_connect_model = init_edge_connect_model(mode=3)

        # source_image, output_image = pipeline(
            # input_image_url,
            # craft_args, # Args create with craft nets, and use for text polygons detection
            # craft_net,
            # refiner_craft_net, # refiner for more text detection accuracy, == none in this project
            # edge_connect_model, # Inpaint EdgeConnect model, "restore" image
            # debug=False
        # )

        # if source_image is None or output_image is None:
            # return

        # #output.clear()
        # ipyplot.plot_images([source_image, output_image], max_images=2, img_width=output_image.width)

        # # Save output image
        # output_image_path = os.path.join("./results_images", image_file_name)
        # output_image.save(output_image_path)
        # print("Safe out image - ", output_image_path)
    # else:
        # print('Provide an image url and try again.')

#########################################################################################
#########################################################################################
#########################################################################################

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi_text_detection(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__text_detection__':
    print_hi_text_detection('PyCharm')
    #test_remover_func()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
