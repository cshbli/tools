import os
import argparse
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import cv2
import sys

# source and credits:
# https://raw.githubusercontent.com/datitran/raccoon_dataset/master/xml_to_csv.py

def parse_args():
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Converting XML annotations to Object Detection CSV format')   

    parser.add_argument('--image_dir',              help='Image path', type=str, default='images')
    parser.add_argument('--annotation_dir',         help='Annotation XML path', type=str, default='annotations')

    parser.add_argument('csv_file_name',            help='Output CSV file name', type=str)
    
    return parser.parse_args()     


def xml_to_csv(image_dir, annotation_dir):
    xml_list = []
    for xml_file in glob.glob(annotation_dir + '/*.xml'):
        print('processing ' + xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)        
        img_file_name = os.path.basename(root.find('path').text)
        im_src = cv2.imread(os.path.join(image_dir, img_file_name), cv2.IMREAD_IGNORE_ORIENTATION)
        img_height, img_width = im_src.shape[:2]        
        if height != img_height or width != img_width:
            print(img_file_name + ' image size not match!')            
        for member in root.findall('object'):
            bbox = member.find('bndbox')
            x1 = max(0, int(bbox.find('xmin').text))
            y1 = max(0, int(bbox.find('ymin').text))
            x2 = min(width, int(bbox.find('xmax').text))
            y2 = min(height, int(bbox.find('ymax').text))                        
            area = (x2-x1) * (y2-y1)
            if x2 < x1 or y2 < y1 or area < 20 :
                print('skip') 
            else :
                value = (os.path.join(image_dir, img_file_name),                                          
                     x1,
                     y1,
                     x2,
                     y2,
                     member[0].text,
                     )
                xml_list.append(value)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def train():    
    xml_df = xml_to_csv(args.image_dir, args.annotation_dir)    
    xml_df.to_csv(args.csv_file_name, header=False, index=None)
    print('> Successfully converted xml to csv.')

def val():
    image_path = os.path.join(os.getcwd(), 'data', 'tf_wider_val', 'annotations','xmls')
    xml_df = xml_to_csv(image_path)
    labels_path = os.path.join(os.getcwd(), 'data', 'tf_wider_val', 'val.csv')
    xml_df.to_csv(labels_path, index=None)
    print('> tf_wider_val -  Successfully converted xml to csv.')


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.image_dir) and not os.path.isdir(args.image_dir):
        print('Image dir does not exist or it is not a directory')          
        sys.exit()

    if not os.path.exists(args.annotation_dir) and not os.path.isdir(args.annotation_dir):
        print('Annotation dir does not exist or it is not a directory')          
        sys.exit()

    train()
    #val()
