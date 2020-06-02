import io
import os
import sys
import math
import logging
import base64
import argparse
import random
import json
import glob
import PIL.Image
import PIL.ImageDraw
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from pycocotools import mask as mask_util
ROOT_PATH = os.path.abspath("../")
sys.path.append(ROOT_PATH)
from datasets.protos import string_int_label_map_pb2

"""
期望的数据集的目录结构如下
-/path/to(name)/datasets(year)
    |-train (type)
        |-train_image_0.jpg
        |-train_image_1.jpg
        |-...
    |-val
        |-val_image_0.jpg
        |-val_image_1.jpg
        |-...
    |-test
        |-test_image_0.jpg
        |-test_image_1.jpg
        |-...
    |-annotations
         |-train - 各自的具体标注
            |-annotation_train_image_0.json
            |-annotation_train_image_1.json
            |-...
         |-val (minival / valminusminival)
            |-annotation_val_image_0.json
            |-annotation_val_image_1.json
            |-...
         |-test
            |-annotation_test_image_0.json
            |-annotation_test_image_1.json
            |-...
         |-instances_train2019.json - 所有训练的标注整合
         |-instances_val2019.json   - 所有验证的标注整合
         |-instances_test2019.json  - 所有测试的标注整合
    |-labelmap.pbtxt    - 类标签的proto文本文件
    |-saliencymap.pbtxt - 显著性标签的proto文本文件
"""


##########################################
#  类索引构造方法
##########################################


def _validate_label_map(label_map):
    """
    检查生成的label_map是否有效
    :param label_map: 用于检查的StringIntLabelMap数据
    :raise
        ValueError: 如果label_map无效
    """
    for item in label_map.item:
        if item.id < 0:
            raise ValueError("标签文件中的ID应≥0")
        if item.id == 0:
            if item.name.lower() != "background":
                raise ValueError("标签ID为0的应该是背景类标签")
            if item.HasField("display_name") and item.display_name.lower() != "background":
                raise ValueError("标签ID为0的应该是背景类标签(显示标签也应该是)")


def load_labelmap(label_map_path):
    """
    读取标签文本文件
    :param label_map_path:
    :return:
    """
    with tf.io.gfile.GFile(label_map_path, "r") as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()  # 初始化label_map对象
        try:
            text_format.Merge(label_map_string, label_map)  # 类型转换
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)  # 验证转换是否正确
    return label_map


def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
    """
    给定标签文本proto数据，然后返回一个与评估兼容的类的列表
    注：如果有多个类同ID的情况下，我们只保留第一个出现的类
    :param label_map: StringIntLabelMapProto或者是None，如果是None，则根据max_num_classes参数构造类
    :param max_num_classes: 最大包括的标签索引的数目(连续的)
    :param use_display_name: boolean，是否选择文本中的display_name部分作为类名，如果为False或者display_name部分不存在，则采用name部分作为类名
    :return:
        一个类索引字典，代表了数据集中所有可能的类
    """
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({"id": class_id + label_id_offset,
                               "name": "category_{}".format(class_id + label_id_offset)})
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:  # 忽略没有介于(0, max_num_classes]的ID
            logging.info("忽略元素%d，因为它超出了要求的标签范围", item.id)
            continue
        if use_display_name and item.HasField("display_name"):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({"id": item.id,
                               "name": name})
    return categories


def create_category_index(categories):
    """
    构造一个与COCO数据集兼容的字典，由其类ID作为索引
    :param categories: 字典的list，必须含有下面两种键："id"，"name"
    :return:
        一个由每个类自己的ID作为键的索引的字典
    """
    category_index = {}
    for category in categories:
        category_index[category["id"]] = category
    return category_index


def create_category_from_labelmap(label_map_path, use_display_name=True):
    """
    读取标签文本文件，然后返回一个与评估兼容的类的列表
    该方法将标签原型文件进行转换并输出一个字典的list，每一个字典元素都包含了下面的键
    "id"：整形的ID，与类唯一对应
    "name"：String的名字，代表了类的真实名称，例如，"cat"，"dog" ...
    :param label_map_path: 以"StringIntLabelMap"为原型的文本文件(.pbtxt)路径
    :param use_display_name: boolean，是否选择文本中的display_name部分作为类名，如果为False或者display_name部分不存在，则采用name部分作为类名
    :return:
        一个类索引字典，代表了数据集中所有可能的类
    """
    label_map = load_labelmap(label_map_path)  # 读取标签文本文件
    max_num_classes = max(item.id for item in label_map.item)  # 最多可能的类数目
    return convert_label_map_to_categories(label_map, max_num_classes, use_display_name)


def create_category_index_from_labelmap(label_map_path, use_display_name=True):
    """
    读取一个标签文本文件，然后返回一个类索引的dict
    :param label_map_path: 以"StringIntLabelMap"为原型的文本文件(.pbtxt)路径
    :param use_display_name: boolean，是否选择文本中的display_name部分作为类名，如果为False或者display_name部分不存在，则采用name部分作为类名
    :return:
        一个类索引字典，将整数的ids与一个包括了类的字典映射起来，例如：
        {1: {"id": 1, "name": person}, 2: {"id": 2, "name": ""bicycle}, ...}
    """
    categories = create_category_from_labelmap(label_map_path, use_display_name)
    return create_category_index(categories)


##########################################
#  labelme生成文件转换类coco数据集方法
##########################################


def img_b64_to_arr(img_b64):
    # 该方法用于解码labelme生成的JSON图像数据，labelme以base64编码，这里进行解码，然后返回图像数据
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    # 该方法为根据图片大小和边框的点坐标，还原出mask
    # 最后输出的mask为二值的图像矩阵(具体值为True False表示是否为mask区域)
    # 输出的矩阵大小为(height, width)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


class Labelme2coco(object):

    def __init__(self,
                 jsonfile_list=None,
                 image_path_prefix=None,
                 save_path=None,
                 category_index=None,
                 saliency_index=None,
                 image_id_start_num=1,
                 ann_id_start_num=1,
                 dataset_type=None,
                 dataset_year=2019):
        self.jsonfile_list = jsonfile_list  # 所有要转化的JSON文件名列表(文件位置列表)
        self.image_path_prefix = image_path_prefix  # 所有图像的路径的前缀(请参考推荐的目录结构)
        self.save_path = save_path  # 最后输出类COCO数据集JSON文件的保存路径
        self.category_index = category_index  # 类ID与具体类的字典的索引
        self.saliency_index = saliency_index  # 显著性ID与显著性字典的索引
        if dataset_type is not None:
            self.dataset_type = dataset_type
            if dataset_type.lower() == 'train':
                self.save_name = os.path.join(save_path, 'instances_train{}.json'.format(dataset_year))
                print('处理训练集数据...')
            elif dataset_type.lower() == 'val' or dataset_type.lower() == "minival" or dataset_type.lower() == "valminusminival":
                self.save_name = os.path.join(save_path, 'instances_val{}.json'.format(dataset_year))
                print('处理验证集数据...')
            elif dataset_type.lower() == 'test':
                self.save_name = os.path.join(save_path, 'instances_test{}.json'.format(dataset_year))
                print('处理测试集数据...')
        else:
            self.dataset_type = 'unknown'
            self.save_name = os.path.join(save_path, 'instances_{}{}.json'.format(dataset_type, dataset_year))
            print('处理数据集数据(未分训练/验证/测试)...')
        self.coco_like = dict()  # 最后输出的JSON数据的总内容
        self.images = []  # 类COCO的Images部分数据
        self.annotations = []  # 类COCO的Annotations的部分数据
        self.categories = []  # 类COCO的Categories的部分数据
        self.saliencies = [{"id": 1, "name": "N"},
                           {"id": 2, "name": "S"},
                           {"id": 3, "name": "L"},
                           {"id": 4, "name": "C"},
                           {"id": 5, "name": "LS"},
                           {"id": 6, "name": "CS"},
                           {"id": 7, "name": "CL"},
                           {"id": 8, "name": "CLS"}]  # 代表不同的显著性分级，级数固定，这里由低到高
        self.label_ids = []  # 所有已有的标签ID，如[2, 5, 1]
        self.labels = []  # 保存所有已有的标签名，如["toothbrush", "comb", "cup"]
        self.imgID = image_id_start_num  # 为了保证每个图像的ID都是独特的，这里对于train/val的数据集进行了记录
        self.annID = ann_id_start_num  # 为了保证每个标注的ID都是独特的，这里对于train/val的数据集进行了记录
        self.height = 0
        self.width = 0

    def data_transfer(self):
        for file_num, json_file in enumerate(self.jsonfile_list):
            # file_num 即是读取的第几个文件，该file_num与images[]中对应某个image[id]对应
            # 同时，与annotations[]中该image的标注annotation[image_id]对应一致
            if file_num % 100 == 0:
                print('正在处理第 %d / %d 幅图像...' % (file_num + 1, len(self.jsonfile_list)))
            with open(json_file, 'r') as fid:
                json_data = json.load(fid)  # 加载某个json文件数据
                self.images.append(json_data["images"][0])
                for ann in json_data['annotations']:
                    class_id = ann["category_id"]
                    if class_id not in self.label_ids:
                        self.label_ids.append(class_id)
                    self.annotations.append(ann)
                    # self.annID += 1
            # self.imgID += 1
        self.label_ids = sorted(self.label_ids)
        self.category_data_transfer(self.label_ids)
        if self.dataset_type.lower() == 'train':
            print('训练集数据格式转换完毕...')
        elif self.dataset_type.lower() == 'val':
            print('验证集数据格式转换完毕...')
        elif self.dataset_type.lower() == 'test':
            print('测试集数据格式转换完毕...')
        else:
            print('数据集格式转换为COCO格式完毕...')
        self.coco_like = dict()
        self.coco_like["images"] = self.images
        self.coco_like["annotations"] = self.annotations
        self.coco_like["categories"] = self.categories
        self.coco_like["saliencies"] = self.saliencies

    def image_data_transfer(self, json_data):
        # 某张图片的数据，其格式对应为
        # {
        # license: 许可证id 这里我们不需要
        # file_name: 文件对应的文件名字 如：xxx.jpg √
        # file_path: 文件对应的本地文件路径 如：D:\\Project\\Python\\Datasets\\xxx.jpg * 新增 *
        # coco_url: coco数据集中的图片地址 这里不需要
        # height: 图片高度 √
        # width: 图片宽度 √
        # date_capture: 图片拍摄时间 这里不需要
        # flickr_url: 图片地址 这里不需要
        # id: 图片的id，这里与读取的文件的序数保持一致 √
        # }
        image_data = dict()
        self.height = json_data["height"]
        self.width = json_data["width"]
        image_data["height"] = self.height
        image_data["width"] = self.width
        image_data["id"] = self.imgID
        image_data["file_name"] = json_data["file_name"]  # os.path还有splittext方法，用于将文件名和扩展名分开
        image_data["file_path"] = json_data["file_path"]  # todo 使用已有的还是获取这些图像对应的本地图像位置
        # image_data["file_path"] = os.path.join(self.image_path_prefix, image_data["file_name"])

        return image_data

    def category_data_transfer(self, label_ids):
        # 某个类的数据，其格式对应为
        # {
        # 'supercategory': 父类的名字，按照命名格式为label分割后的第一个 √
        # 'id': 该类的id，即与annotation中的category_id值对应 √
        # 'name': 类名称，即具体的类名，按照命名格式为label分割后的第二个 √
        # }
        for label in label_ids:
            category_data = dict()
            category_data['supercategory'] = ''
            category_data.update(self.category_index[label])
            self.categories.append(category_data)
        return self.categories

    def annotation_data_transfer(self, points, label, segmentation, file_name, save_mask=False, keep_decimal=True):
        # 某个标注的annotation的格式对应为
        # {
        # 'segmentation': [[mask的各个点，labelme生成为[N, 2]，这里需展平为 x1. y1. x2. y2排列]]}], 这里可选择保留两位小数 √
        # 'area': mask像素面积, √ 由cocoAPI去进行计算
        # 'iscrowd': 目前默认为0 √ 这里都是多边形，没有REL格式，所以不用1
        # 'image_id': 与该图片的id即image中的id对应，即读取的文件序数 √
        # 'bbox': 根据mask直接计算 注意：这里的bbox不是[ymin xmin ymax xmax]格式 而是 [xmin ymin bwidth bheight]格式 √ 由cocoAPI计算
        # 'category_id': 子类对应的具体类的id √
        # 'saliency_id': 子类对应的具体显著性级别的id * 新增 *
        # 'id': 这一份annotation的id，每写一个这个值就加1 √
        # }
        annotation_data = dict()
        if keep_decimal:  # 是否保留两位小数
            points_np = np.round(np.asarray(points), 2)
        else:
            points_np = np.asarray(points)
        annotation_data['segmentation'] = [points_np.flatten().tolist()]
        annotation_data['iscrowd'] = 0
        annotation_data['image_id'] = self.imgID
        annotation_data['bbox'] = mask_util.toBbox(segmentation).tolist()
        annotation_data['area'] = float(mask_util.area(segmentation))
        if len(label) == 1:
            for label_info in self.category_index.values():
                if label_info['name'] == label[0]:
                    annotation_data['category_id'] = label_info['id']  # 与category对应
        else:
            for label_info in self.category_index.values():
                if label_info['name'] == label[1]:
                    annotation_data['category_id'] = label_info['id']  # 与category对应
            for saliency_info in self.saliency_index.values():
                if saliency_info["name"] == label[-1]:
                    annotation_data["saliency_id"] = saliency_info["id"]  # 与saliency对应
        annotation_data['id'] = self.annID
        if save_mask:
            encoding = mask_util.frPyObjects(annotation_data['segmentation'], self.height, self.width)
            binary_mask = mask_util.decode(encoding)
            binary_mask = np.amax(binary_mask, axis=2)  # amax是在指定维度上求最大值，这里其实和squeeze效果等价 shape会从(height, width, 1) 变成 (height, width)
            binary_mask = binary_mask * 255
            pil_image = PIL.Image.fromarray(binary_mask)  # 这里只是保存每个mask，具体到tfrecord数据是不要转化为0~255图像，而是保持0,1的mask，只是方便观察
            pil_image.save(file_name.split('.')[0] + '_' + str(label[1]) + '_' + str(label[2]) + '.PNG')
        return annotation_data

    def saliency_data_transfer(self, label):
        # 某个显著性级别的数据，其格式对应为
        # {
        # 'level': 显著性级别的名字，共分9级(0代表背景类，即非显著性，标注的ID只有1-8，均为标注为显著性的部分)
        # 显著性分级按照color、location、shape贡献排列，其中颜色对比度高的>位置靠中心的>面积占比大的
        # 具体格式为'CLS'>'CL'>'CS'>'C'>'LS'>'L'>'S'>'N' 对应的ID为8 > 7 > 6 > 5 > 4 > 3 > 2 > 1
        # CLS含义为：与附近颜色对比度高且位于图像中心区域且形状较大
        # CL含义为： 与附近颜色对比度高且位于中心区域(形状不大)
        # CS含义为： 与附近颜色对比度高且形状较大(不位于中心区域)
        # C含义为：  与附近颜色对比度高(不位于中心区域，形状也不大)
        # LS含义为： 位于中心区域且形状较大(颜色对比度不高)
        # L含义为：  位于中心区域(颜色对比度不高且形状不大)
        # S含义为：  形状较大
        # N含义为：  上面均不具备，但是是检测的目标物体
        # 'id': 显著性级别的数字id，即上面的1-8
        # }
        saliency_data = dict()

    def save_json(self):
        self.data_transfer()
        json.dump(self.coco_like, open(self.save_name, 'w'), indent=4)
        # image_name_list = [os.path.splitext(os.path.split(json_file)[-1])[0] + '.jpg' for json_file in self.jsonfile_list]
        # json.dump(image_name_list, open(os.path.join(self.save_path, str(self.dataset_type) + '_image_name.json'), 'w'), indent=4)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(usage="Python utils"
                                          " --path (/path/to/json-dir)"
                                          " --label (/path/to/labelmap)"
                                          " --saliency (/path/to/saliencymap)"
                                          " --out (/path/to/output)",
                                    description="用于将Labelme生成的每个图片的标注转换为类COCO数据集的格式(增加了显著性的标注)，"
                                                "如果你的标注文件有显著性的标注，请同时给上显著性的真值标签表",
                                    epilog="版本：1.0.2，维护者：Jhc")
    parse.add_argument("--path",metavar="/path/to/json-dir", required=True, type=str,
                       help="标注的JSON文件所在的文件夹路径")
    parse.add_argument("--label", metavar="/path/to/labelmap", required=True, type=str,
                       help="标注的所有类的具体的类ID和类名，应为.pbtxt文件格式")
    parse.add_argument("--saliency", metavar="/path/to/saliencymap", required=False, type=str, default="",
                       help="标注的所有的显著性的类ID和类名，应为.pbtxt文件格式")
    parse.add_argument("--out", metavar="/path/to/output", required=True, type=str,
                       help="输出的整合了标注的类COCO数据集格式的文件的路径或文件名(应为.json文件格式)")
    args = parse.parse_args()
    print(args.path)
    category_index = create_category_index_from_labelmap(args.label)
    if args.saliency:
        saliency_index = create_category_index_from_labelmap(args.saliency)
    else:
        saliency_index = None
    load_path = os.path.abspath(args.path)  # 标注文件读取路径
    save_path = os.path.abspath(args.out)  # 输出的类COCO数据集标注路径
    if os.path.isdir(load_path):
        if os.name == "nt":  # Windows下，路径为\表示
            dataset_year, _, dataset_type = load_path.split("\\")[-3:]  # 自动根据文件夹名字判断为训练集/验证集/测试集，如果无法判断则先不考虑数据集类型
            if not dataset_type.lower() in ["train", "val", "minival", "valminusminival", "test"]:
                dataset_type = None
            try:
                dataset_year = int(dataset_year)
            except ValueError:
                print("无法将{}转换为整形数，请参考期望的目录结构，数据年限将采用默认值(2019)".format(dataset_year))
                dataset_year = 2019
        else:
            dataset_year, _, dataset_type = load_path.split("/")[-3:]  # 自动根据文件夹名字判断为训练集/验证集/测试集，如果无法判断则先不考虑数据集类型
            if not dataset_type.lower() in ["train", "val", "minival", "valminusminival", "test"]:
                dataset_type = None
            try:
                dataset_year = int(dataset_year)
            except ValueError:
                print("无法将{}转换为整形数，请参考期望的目录结构，数据年限将采用默认值(2019)".format(dataset_year))
                dataset_year = 2019
        jsonfile_list = glob.glob(os.path.join(load_path, "*.json"))  # 获取当前目录下所有匹配的JSON标注文件
        image_path_prefix = os.path.split(jsonfile_list[0])[0].replace("annotations\\train", "train")  # 用于后续
        assert os.path.exists(image_path_prefix), "该目录{}应该存在".format(image_path_prefix)
        assert os.path.isdir(image_path_prefix), "该目录{}应该为文件夹".format(image_path_prefix)
        if not jsonfile_list:  # 如果当前文件夹下没有JSON文件
            raise FileNotFoundError("请确认--path参数下的文件夹是否包含了JSON文件")
        transfertool = Labelme2coco(jsonfile_list=jsonfile_list,
                                    image_path_prefix=image_path_prefix,
                                    save_path=save_path,
                                    category_index=category_index,
                                    saliency_index=saliency_index,
                                    dataset_type=dataset_type,
                                    dataset_year=dataset_year)
        transfertool.save_json()
    else:
        raise ValueError("请确认--path参数的值，是否为包含了标注的JSON文件夹(非JSON文件)且存在")
