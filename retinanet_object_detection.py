import cv2
import numpy as np
import os
import time


def retinanet_preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean. The input image is in BGR format

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def retinanet_compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = float(min_side) / float(smallest_side)

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = float(max_side) / float(largest_side)

    return scale


def retinanet_resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = retinanet_compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)
    # scale = round(scale, 4)
    # print("scale: ", scale)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def object_too_close_to_boundary(b, img_width, img_height, boundary_filter_size):
    """ Check if the detected bounding box is too close to the image boundary

    Args
        b:                      The detected object bounding box
        img_width:              Source image width
        img_height:             Source image height
        boundary_filter_size:   The minimum distance to the boundary

    Returns
        True or False
    """
    if b[0] < boundary_filter_size or b[1] < boundary_filter_size or b[2] > img_width - boundary_filter_size or b[3] > img_height - boundary_filter_size:
        return True
    else:
        return False


def retinanet_object_detection(image_org,
                                object_detection_graph, object_detection_session, 
                                object_detection_threshold=0.5,
                                object_count_threshold=1,
                                backbone='resnet',
                                boundary_filter_size=0,                                
                                output_detection_image=False,
                                output_dir=None,
                                img_name=None,
                                patch_idx=-1):
    """ Object detection based on retinanet

    Args
        image_org:                  The original input image
        object_detection_graph:     The Tensorflow Graph
        object_detection_session:   The Tensorflow Session
        object_detection_threshold: The detection threhshold for object detection algorithm
        object_count_threshold:     The minimum number of objects have to detected, otherwise returns error
        backbone:                   The backbone network of the detection graph. The backbone will decide the image preprocessing function
        boundary_filter_size:       The minimum distance to the boundary        

        The following 4 parameters are for debugging purpose only:

        output_detection_image:     True to output detecton result image, False not
        output_dir:                 The output dir to store the detection result image
        img_name:                   The full input image name, reuse the name to save the detection results, can be None
        patch_idx:                  The original image may has been splitted into patches for object detection.
                                    This patch idex is purely for output detection image debugging purpose

    Returns
        ret:                    0 successful
                                100 Not enough objects detected                                
        objects:                An array of bounding box center positions, excluding those bounding boxes too close to boundaries
        average_object_width:   Average object bounding box width 
        average_object_height:  Average object bounding box height
    """

    ret = 0

    img_height, img_width = image_org.shape[:2]

    if img_name != None:
        img_base_name = os.path.splitext(os.path.basename(img_name))[0]
        if patch_idx >= 0:
            img_base_name = img_base_name + '_' + str(patch_idx)

    # copy to draw on
    draw = image_org.copy()

    # preprocess each image for network
    if backbone == 'mobilenet' or backbone == 'densenet':
        img = retinanet_preprocess_image(image_org, mode='tf')
    else:
        img = retinanet_preprocess_image(image_org, mode='caffe')

    img, scale = retinanet_resize_image(img)

    #print(scale)

    # process image
    start = time.time()
    image_tensor = object_detection_graph.get_tensor_by_name('input_1:0')
    output_tensor_0 = object_detection_graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0')
    output_tensor_1 = object_detection_graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0')
    output_tensor_2 = object_detection_graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0')
    boxes, scores, labels = object_detection_session.run([output_tensor_0, output_tensor_1, output_tensor_2], feed_dict={image_tensor: np.expand_dims(img, axis=0)})
    #print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    #print(scores[0])
    #print(labels[0])

    # visualize detections
    detected_bboxes = []    
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < object_detection_threshold:
            break

        # print(score, label)
        b = box.astype(int)
        
        detected_bboxes.append(b)
        
    if len(detected_bboxes) < object_count_threshold:
        ret = 100    

    #print(len(detected_bboxes))

    # Using the bounding box centers as object locations and filter out those objects too close to boundaries
    objects = []
    scores_filtered = []
    labels_filtered = []
    object_width_sum = 0
    object_height_sum = 0
    for i in range(len(detected_bboxes)):
        b = detected_bboxes[i]
        if object_too_close_to_boundary(b, img_width=img_width, img_height=img_height, boundary_filter_size=boundary_filter_size):
            continue          
        objects.append(b)
        scores_filtered.append(scores[0][i])
        labels_filtered.append(labels[0][i])

        object_width_sum += (b[2] - b[0])
        object_height_sum += (b[3] - b[1])

    if len(objects) > 0:
        average_object_width = int(object_width_sum / len(objects))
        average_object_height = int(object_height_sum / len(objects))
    else:
        average_object_width = 0
        average_object_height = 0

    #print(average_object_width)
    #print(average_object_height)    

    # Save the detection images
    if output_detection_image:
        if img_width < 1000 or img_height < 1000:
            thickness = 1
        elif img_width < 2000 or img_height < 2000:
            thickness = 2
        else:
            thickness = 3

        # draw detection boxes
        for i in range(len(detected_bboxes)):
            b = detected_bboxes[i]
            if object_too_close_to_boundary(b, img_width=img_width, img_height=img_height, boundary_filter_size=boundary_filter_size):
                draw_box(draw, b, color=(0, 0, 255), thickness=thickness)            
            elif labels[0][i] >= 1:
                draw_box(draw, b, color=(255, 255, 0), thickness=thickness)            
            else:
                draw_box(draw, b, color=(255, 0, 0), thickness=thickness)
        img_output_filename = img_base_name + '_d.jpg'
        cv2.imwrite(os.path.join(output_dir, img_output_filename), draw)    

    return ret, objects, scores_filtered, labels_filtered, average_object_width, average_object_height

