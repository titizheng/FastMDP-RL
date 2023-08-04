import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import openslide as ol
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image
import torch
import argparse
import torch.utils.data as data
import re

def seg_slide(slide, seg_level, patch_size):
    """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
       
    """

    def _filter_contours(contours, hierarchy, filter_params):  
        """
            Filter contours by: area.
            hierarchy
        """
        filtered = []

        # find foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)

        for cont_idx in hierarchy_1:
            cont = contours[cont_idx]
            a = cv2.contourArea(cont)
            if a > filter_params['a_t']:
                filtered.append(cont_idx)

        all_holes = []
        for parent in filtered:
            all_holes.append(np.flatnonzero(hierarchy[:, 1] == parent))

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []

        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
            filtered_holes = []

            for hole in unfilered_holes:
                if cv2.contourArea(hole) > filter_params['a_h']:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

    level_dim = slide.level_dimensions
    mthresh = 7
    sthresh = 120
    sthresh_up = 255
    close = 4
    ref_patch_size = 512
    filter_params = {'a_t': 1, 'a_h': 1, 'max_n_holes': 1}

    img = np.array(slide.read_region((0, 0), seg_level, level_dim[seg_level])
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
    img_med = cv2.medianBlur(img_gray, mthresh)  # Apply median blurring，
    # Thresholding
    _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY_INV)#
    # plt.imshow(img_otsu)
    # plt.show()

    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, kernel)
  


    scale = _assertLevelDownsamples(slide)[seg_level]#
    scaled_patch_area = int(ref_patch_size ** 2 / (scale[0] * scale[1]))
    filter_params['a_t'] = filter_params['a_t'] * scaled_patch_area
    filter_params['a_h'] = filter_params['a_h'] * scaled_patch_area

    # Find and filter contours
    contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
   
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]#
    if filter_params:
        foreground_contours, hole_contours = _filter_contours(contours, hierarchy,filter_params)  # Necessary for filtering out artifacts，

    
    contours_tissue = scaleContourDim(foreground_contours, scale)#
    holes_tissue = scaleHolesDim(hole_contours, scale)
    return contours_tissue, holes_tissue

def vis_slide(slide, contours_tissue, holes_tissue, vis_level):
    """
   
    :param slide:
    :param contours_tissue:
    :param holes_tissue:
    :param vis_level:
    :return:
    """
    level_dim = slide.level_dimensions
    line_thickness = 250
    color = (0, 255, 0)
    hole_color = (0, 0, 255)
    annot_color = (255, 0, 0)
    crop_window = None
    max_size = None

    img = np.array(slide.read_region((0, 0), vis_level, level_dim[vis_level]).convert("RGB"))
    downsample = _assertLevelDownsamples(slide)[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level
    line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
    if contours_tissue is not None:
        cv2.drawContours(img, scaleContourDim(contours_tissue, scale),
                         -1, color, line_thickness, lineType=cv2.LINE_8)

        for holes in holes_tissue:
            cv2.drawContours(img, scaleContourDim(holes, scale),
                             -1, hole_color, line_thickness, lineType=cv2.LINE_8)

    img = Image.fromarray(img)

    if crop_window is not None:
        top, left, bot, right = crop_window
        left = int(left * scale[0])
        right = int(right * scale[0])
        top = int(top * scale[1])
        bot = int(bot * scale[1])
        crop_window = (top, left, bot, right)
        img = img.crop(crop_window)
    w, h = img.size
    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))
    return img


def tiling_slide(slide, contours_tissue, holes_tissue, patch_level, patch_size, patch_step, name):
    """
   
    :param slide:
    :param contours_tissue:
    :param holes_tissue:
    :param patch_level:
    :param patch_size:
    :param patch_step:
    :param save_path:
    :param name:
    :return:
    """
    grid = []
    print("Creating patches for: ", name, "...", )
    num = 0
    for idx, cont in enumerate(contours_tissue):
       
        patch_gen = _getPatchGenerator(slide, cont, idx, patch_level, holes_tissue, patch_size, patch_step)
        for coord in patch_gen:
            x, y = coord["x"], coord["y"]
            grid.append((x, y))
            num += 1
    
    return grid

def isInContourV1(cont, pt, patch_size=None):
    return 1 if cv2.pointPolygonTest(cont, pt, False) >= 0 else 0



def isInContourV2(cont, pt, patch_size=256):
    return 1 if cv2.pointPolygonTest(cont, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False) >= 0 else 0


def isInContourV3(cont, pt, patch_size=256):
    center = (pt[0] + patch_size // 2, pt[1] + patch_size // 2)
    all_points = [(center[0] - patch_size // 4, center[1] - patch_size // 4),
                  (center[0] + patch_size // 4, center[1] + patch_size // 4),
                  (center[0] + patch_size // 4, center[1] - patch_size // 4),
                  (center[0] - patch_size // 4, center[1] + patch_size // 4)
                  ]
    for points in all_points:
        if cv2.pointPolygonTest(cont, points, False) >= 0:
            return 1

    return 0


def isInContours(cont_check_fn, contour, pt, holes=None, patch_size=256):
    
    if cont_check_fn(contour, pt, patch_size):
        if holes is not None:
            return not isInHoles(holes, pt, patch_size)
        else:
            return 1
    return 0


def isInHoles(holes, pt, patch_size):
    for hole in holes:
        if cv2.pointPolygonTest(hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False) > 0:
            return 1

    return 0


def isWhitePatch(patch, satThresh=5):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:, :, 1]) < satThresh else False


def isBlackPatch(patch, rgbThresh=40):
    return True if np.all(np.mean(patch, axis=(0, 1)) < rgbThresh) else False

def _getPatchGenerator(slide, cont, cont_idx, patch_level, holes_tissue, patch_size=256, step_size=256, use_padding = True):
    """
    get patch
    :param slide: openslide.OpenSlide
    :param cont: contours
    :param cont_idx:
    :param patch_level:
    :param save_path:
    :param holes_tissue:
    :param patch_size:
    :param step_size:
    :param name:
    :return:
    """
    white_black = True
    white_thresh = 5
    black_thresh = 40
    contour_fn = 'four_pt'

    level_dim = slide.level_dimensions

    start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (
        0, 0, level_dim[patch_level][0], level_dim[patch_level][1])#cv2.boundingRect
    print("Bounding Box:", start_x, start_y, w, h)
    print("Contour Area:", cv2.contourArea(cont))#cv2.contourArea


    # the downsample corresponding to the patch_level
    patch_downsample = (int(_assertLevelDownsamples(slide)[patch_level][0]),
                        int(_assertLevelDownsamples(slide)[patch_level][1]))#
    # size of patch at level 0 (reference size)
    ref_patch_size = (patch_size * patch_downsample[0], patch_size * patch_downsample[1])#

    # step sizes to take at levl 0
    step_size_x = step_size * patch_downsample[0]*1
    step_size_y = step_size * patch_downsample[1]*1   

    if contour_fn == 'four_pt':
        cont_check_fn = isInContourV3
    elif contour_fn == 'center':
        cont_check_fn = isInContourV2
    elif contour_fn == 'basic':
        cont_check_fn = isInContourV1
    else:
        raise NotImplementedError

    img_w, img_h = level_dim[0]#

    if use_padding:
        stop_y = start_y + h
        stop_x = start_x + w
    else:
        stop_y = min(start_y + h, img_h - ref_patch_size[1])  # 
        stop_x = min(start_x + w, img_w - ref_patch_size[0])

    # stop_y = min(start_y + h, img_h - ref_patch_size[1])
    # stop_x = min(start_x + w, img_w - ref_patch_size[0])

    count = 0
    for y in range(start_y, stop_y, step_size_y):  #
        for x in range(start_x, stop_x, step_size_x):

            if not isInContours(cont_check_fn, cont, (x, y), holes_tissue[cont_idx],ref_patch_size[0]):  # point not inside contour and its associated holes
                continue #

            count += 1
            # patch_PIL = slide.read_region((x, y), 0, (ref_patch_size[0], ref_patch_size[1])).convert('RGB')
            patch_PIL = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert('RGB')

            if white_black:
                if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(
                        np.array(patch_PIL), satThresh=white_thresh):
                    continue

            # x, y coordinates become the coordinates in the downsample, and no long correspond to level 0 of WSI
            
            patch_info = {'x': x // (patch_downsample[0]),
                          'y': y // (patch_downsample[1])}

            yield patch_info  

            # patch_info = {'x': x // (patch_downsample[0]),
            #               'y': y // (patch_downsample[1] ), 'cont_idx': cont_idx,
            #               'patch_level': patch_level,
            #               'downsample': slide.level_downsamples[patch_level],
            #               'downsampled_level_dim': tuple(np.array(slide.level_dim[patch_level]) ),
            #               'level_dim': slide.level_dim[patch_level],
            #               'patch_PIL': patch_PIL, 'name': slide.name, 'save_path': save_path}# 
            #
            # yield patch_info

            
            # patch_level 
            # level_downsamples [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0), (16.0, 16.0), (32.0, 32.0), (64.0, 64.0), (128.0, 128.0),(256.0, 256.0), (512.0, 512.0)]
            # level_dim ((31744, 37888), (15872, 18944), (7936, 9472), (3968, 4736), (1984, 2368), (992, 1184), (496, 592),(248, 296), (124, 148), (62, 74))
            # name '70533_0-Tumor'

    print("patches extracted: {}".format(count))

def _assertLevelDownsamples(slide):
    """
    
    <class 'list'>: [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0), (16.0, 16.0), (32.0, 32.0), (64.0, 64.0), (128.0, 128.0)]
    :param slide:
    :return:
    """
    level_downsamples = []
    dim_0 = slide.level_dimensions[0]
   
    for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):#
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples.append((downsample, downsample))

    return level_downsamples

def scaleContourDim(contours, scale):
    return [np.array(cont * scale, dtype='int32') for cont in contours]

def scaleHolesDim(contours, scale):
    return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]


def StitchPatches(grid, slide,downscale=4, draw_grid=False, bg_color=(0, 0, 0), alpha=-1):
    coordls = []
    for gi in grid:
        gi_list = []
        gi_list.append(gi[0])
        gi_list.append(gi[1])
        coordls.append(gi_list)
    source_coords = np.array(coordls)
    # coords = list(grid ) #ndarray[[2424 3176],[3176 384],[3176 768],...]
    #((35712, 26368), (17856, 13184), (8928, 6592), (4464, 3296), (2232, 1648), (1116, 824), (558, 412), (279, 206))
    w, h =  slide.level_dimensions[args.level]

    print('original size: {} x {}'.format(w, h))
    w = w // downscale
    h = h // downscale
    resize_coords = (source_coords / downscale).astype(np.int32)  #
    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(grid)))
    downscaled_shape = (args.patch_size // downscale, args.patch_size // downscale)

    heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap,source_coords, resize_coords, downscaled_shape, indices=None, draw_grid=draw_grid)  # 

    return heatmap


def DrawMap(canvas, source_coords, resize_coords, patch_size, indices=None, verbose=1, draw_grid=True):
    if indices is None:
        indices = np.arange(len(resize_coords)
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        # print('start stitching {}'.format(patch_dset.attrs['wsi_name']))

    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))

        patch_id = indices[idx]
        x_coord,y_coord = source_coords[patch_id]
        sour_coord = (x_coord* 2** args.level, y_coord *2** args.level)
        patch = np.array(slide.read_region(sour_coord, args.level, (256, 256)).convert("RGB"))

        # patch = patch_dset[patch_id]
        patch = cv2.resize(patch, patch_size)
        coord = resize_coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch[:canvas_crop_shape[0],
                                                                                           :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def DrawGrid(img, coord, shape, thickness=2, color=(0,0,0,255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img


def stitching(grid, downscale=64):
    start = time.time()
    heatmap = StitchPatches(grid, slide,downscale=downscale, bg_color=(0, 0, 0), alpha=-1,
                                draw_grid=False)  # 这个保存patch的图片的时候，里面有一个downscal把文件进行了缩小
    total_time = time.time() - start

    return heatmap, total_time


'''--------------------------------------------------------------'''

if __name__ == '__main__':
    parse = argparse.ArgumentParser("whole slide image preprocessing!")
    parse.add_argument("--slide_path", default="", type=str, help="slide path")
    parse.add_argument("--level", default=1, type=int, help="level of slide")
    parse.add_argument("--patch_size", default=256, type=int, help="patch size")
    parse.add_argument("--step_size", default=256, type=int, help="step")
    parse.add_argument("--save_pth", default=r"", type=str, help="Store the extracted coordinate points in the PTH file")
    parse.add_argument("--save_vis", default=r"",type=str, help="Store the extracted tissue profile image")
    parse.add_argument("--save_sti",default=r"", type=str,help="Store the stitch tissue profile image")

    # parse.add_argument("--save_patch", default=r"", type=str, help="The path where the patch is stored")
    args = parse.parse_args()
    slide_num, target = 0, 0
    
    file_name = []
    grids = []
    targets = []


    slides_mel = sorted(os.listdir(os.path.join(args.slide_path, "Tumor")))  
    slides_nev = sorted(os.listdir(os.path.join(args.slide_path, "Normal")))  
    slides = []  
 

    for i in slides_mel:  
        slides.append('Tumor/' + i)  # melanoma

    for j in slides_nev:  
        slides.append('Normal/' + j)  # Normal


   
    slides = [slide for slide in slides if os.path.isfile(os.path.join(args.slide_path, slide))]

    for name in slides:
        disease_name,slide_name =  name.split("/") #
        sn, _ = os.path.splitext(slide_name)

        pthnames = sn + ".pth"

        if os.path.isfile(os.path.join(args.save_pth, pthnames)) is False: #
            slide_path = os.path.join(os.path.join(args.slide_path, disease_name), slide_name)
            slide = ol.OpenSlide(slide_path)
            contours_tissue, holes_tissue = seg_slide(slide=slide, seg_level=args.level, patch_size=args.patch_size)
            grid = tiling_slide(slide=slide, contours_tissue=contours_tissue, holes_tissue=holes_tissue,
                                patch_level=args.level,
                                patch_size=args.patch_size, patch_step=args.step_size, name=slide_name)
            if re.search("Tumor", disease_name) is not None:
                target = 1
            if re.search("Normal", disease_name) is not None:  # Normal
                target = 0

            torch.save({
                "slides": slide_path,
                "grid": grid,
                "targets": target,
                "level": args.level
            }, os.path.join(args.save_pth, "{}.pth".format(sn)))  # 
            file_name.append(slide_path)
            grids.append(grid)
            targets.append(target)
            slide_num += 1




           
            vis_img = vis_slide(slide, contours_tissue, holes_tissue, vis_level=args.level) #image类型
            mask_path = os.path.join(args.save_vis, sn + '.png')
            vis_img.save(mask_path)
            # plt.imshow(vis_img)
            # plt.show()
            #
           
            heatmap, stitch_time_elapsed = stitching(grid, downscale=8) 
            stitch_path = os.path.join(args.save_sti,sn + '.png')
            heatmap.save(stitch_path)




    torch.save({
        "slides": file_name,
        "grid": grids,
        "targets": targets,
        "level": args.level,
        "mult": "1",
    }, os.path.join(args.save_pth, "level_{}.pth".format(args.level)))


  
