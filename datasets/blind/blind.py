import os
import sys
import json
import numpy as np
from collections import defaultdict
from pycocotools import mask as maskUtils
# 获取根目录
ROOT_DIR = os.path.abspath("../../")

# 引入msrcnn库包，增加系统路径以便于能找到本地库
from msrcnn.config import Config
from msrcnn.utils import Dataset
from datasets import create_dataset


# 用于保存日志和模型检查点的路径，如果没有提供，可以通过命令行的--log参数提供
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2019"


##########################################
#  Blind数据集的配置类
##########################################

class BlindConfig(Config):
    """
    用于训练本地blind数据集的配置对象，继承自Config基类并重写了一些属性
    """
    # 该数据集的识别名
    NAME = "blind"

    # 多GPU请根据实际配置GPU数目
    GPU_COUNT = 1

    # 我们的显卡(2080TI)只能支持一幅图
    IMAGES_PER_GPU = 1

    STEPS_PER_EPOCH = 1000

    # 数据集总共的类数(包括BackGround)
    NUM_CLASSES = 1 + 22

    # 数据集总共的显著性级别数(比如，此时对应S，L，C)
    NUM_SALIENCY_LEVELS = 3

    MIN_ROI_LEVEL = {"class": 2,
                     "saliency": 3,
                     "mask": 2}

    USE_FINE_GRAINED = {"class": False,
                        "saliency": True,
                        "mask": False}

    # 使用每个区域坐标信息的跳跃连接
    USE_ROI_SHORTCUT = True

    # 是否使用sigmoid激活各标签，否则使用softmax激活
    USE_SIGMOID = True

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_saliency_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }


class BlindInferenceConfig(BlindConfig):
    """
    用于推断过程中使用的配置类
    """
    # 推断过程中，Batch_size置为1，以便于评估
    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.8

    DETECTION_MIN_SALIENCY_CONFIDENCE = 0.5

##########################################
#  Blind数据集类
##########################################


class BlindDataset(Dataset):
    """
    继承于Dataset类的本地blind数据集类，用于数据集操作，blind数据集增加了显著性标注，要重写相关方法
    """

    def load_blind(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, labelmap_path=None, saliencymap_path=None):
        category_index = create_dataset.create_category_index_from_labelmap(labelmap_path)
        saliency_index = create_dataset.create_category_index_from_labelmap(saliencymap_path)
        # 构造一个图像ID与所有包含的标注的索引，例如图像ID为1，其包含两个标注，因此其值应为[{"segmentation":[[]],},{"segmentation":[[]],}]
        # 整个字典为{1: [{}, {}], ...}，键为图像ID，值为该图像对应所有标注(注：一幅图像一定要么全有显著性标注，要么全没有)
        annotation_index = defaultdict(list)
        class_ids = []
        saliency_ids = []
        jsonfile_path = os.path.abspath("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        with open(jsonfile_path, "r") as fid:
            json_data = json.load(fid)
            images = json_data["images"]  # 读取所有图像的数据
            annotations = json_data["annotations"]  # 读取所有标注的数据(一幅图像可能有多个标注数据)
            categories = json_data["categories"]  # 读取所有的类数据
            for annotation in annotations:
                annotation_index[annotation["image_id"]].append(annotation)  # 为所有标注给定其图像ID索引
            for category in categories:
                class_ids.append(category["id"])  # 标注中所有类的ID(原文件中的ID顺序不重要)
            assert len(class_ids) > 0, "未查找到类别信息!请检查标注文件的categories部分"
            class_ids = sorted(class_ids)  # 所有由小到大的类的IDs
            for class_id in class_ids:
                # 添加各个类别的信息
                self.add_class("blind", class_id, category_index[class_id]["name"])
            for image in images:
                # 添加各个图像的信息
                assert annotation_index[image["id"]] != [], "请确认你的每一幅图都有标注,{}".format(image["id"])
                self.add_image("blind", image["id"], path=image["file_path"],
                               height=image["height"], width=image["width"],
                               annotations=annotation_index[image["id"]])
            if "saliencies" in json_data:  # 如果有显著性标签
                saliencies = json_data["saliencies"]  # 读取所有的显著性类数据(可能有)
                # 所有的显著性信息将以
                # [{"id": , "name": , "source": },
                # {}]
                # 的列表来呈现且一定包括一个完全不显著的类ID(0)
                for saliency in saliencies:
                    saliency_ids.append(saliency["id"])
                assert len(saliency_ids) > 0, "请检查标注文件的saliency部分"
                saliency_ids = sorted(saliency_ids)
                for saliency_id in saliency_ids:
                    self.add_saliency("blind", saliency_id, saliency_index[saliency_id]["name"])

    def load_mask(self, image_id):
        """
        读取指定图像ID的所有标注实例的masks，包括类别ids，以及显著性ids(如果有)
        不同的数据集采用不同的存储mask的方式，该方法将不同的mask数据格式转换为位图的数据格式[height, width, instances]
        :param image_id: 读取图像的ID(内部ID，可能与图像本身的ID不同)
        :return:
            masks: 一个bool的array，其shape为[height, width, instances count]，即一个实例对应一个mask
            class_ids: 一维的array表示了每个实例对应的类ID
        """
        # 如果不是blind数据集图像，则调用父类的方法
        image_info = self.image_info[image_id]  # 获取对应内部ID的具体信息
        if image_info["source"] != "blind":
            return super().load_mask(image_id)

        instance_masks = [] # 该图像所有标注的实例mask
        class_ids = []  # 该图像对应的所有的类ID
        saliency_ids = []  # 该图像对应的所有多标签显著性类的ID(如果存在)，否则最终会返回全[0, 0, 0]的list
        annotations = image_info["annotations"]  # 该图像对应所有标注的数据
        if "saliency_id" in annotations[0]:
            has_saliency = True  # keys中有该键，说明该图是有显著性级别标注
        else:
            has_saliency = False

        # 构造一个[height, width, instance count]的mask
        for annotation in annotations:
            # 遍历当前图像中的每个标注(有Saliency标注的图像则其所有标注都是有Saliency标注的)
            class_id = self.map_source_class_id("blind.{}".format(annotation["category_id"]))  # 获取该标注实例的类ID
            if class_id:  # 该类不能是BG类
                m = self.annToMask(annotation, image_info["height"], image_info["width"])  # 将标注转换为Mask
                # 如果物体太小以至于转换后的Mask全为0(像素面积)
                if m.max() < 1:  # 如果Mask最大值没有1，则说明像素面积为0
                    continue
                #  对于crowd类的标注，赋予负类标签，且注意返回的Mask
                if annotation["iscrowd"]:
                    class_id *= -1
                    # 对于crowd标注的Mask，annToMask()方法有时间会返回比给定维度小的mask，如果是，则resize
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=np.bool)
                instance_masks.append(m)
                class_ids.append(class_id)
                if has_saliency:
                    saliency_id = self.map_source_saliency_id("blind.{}".format(annotation["saliency_id"]))  # 获取该标注实例的显著性ID(为list)
                    saliency_ids.append(saliency_id)  # 添加对应多标签
                else:
                    if self.use_sigmoid:
                        saliency_ids.append([0, 0, 0])
                    else:
                        saliency_ids.append([0])

        # 最后将该图像的所有实例Mask打包为一个Array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            saliency = np.array([has_saliency], dtype=np.bool)
            if self.use_sigmoid:
                saliency_ids = np.array(saliency_ids, dtype=np.float32)
            else:
                saliency_ids = np.array(saliency_ids, dtype=np.int32)
            return mask, class_ids, saliency, saliency_ids
        else:  # 如果该图像没有类，则返回空的数据
            return super().load_mask(image_id)

    def annToRLE(self, ann, height, width):
        """
        将标注(可能是多边形/未压缩RLE/RLE)压缩为RLE格式
        """
        segm = ann["segmentation"]
        if isinstance(segm, list):
            # 为多边形标注格式，而且一个目标可能会有多个标注区域，我们要把所有的部分混合在一个RLE编码中
            rles = maskUtils.frPyObjects(segm, height, width)  # 获取该标注的数据并转换为RLE
            rle = maskUtils.merge(rles)  # 将同一实例的所有RLE混合在一起
        elif isinstance(segm["counts"], list):
            # 为非压缩RLE格式
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # 本身就是RLE格式
            rle = segm
        return rle

    def annToMask(self, ann, height, width):
        """
        将标注(可能是多边形/未压缩RLE/RLE)转换为二值的Mask
        """
        rle = self.annToRLE(ann, height, width)  # 首先标注统一转换为RLE格式
        m = maskUtils.decode(rle)  # 将RLE格式解码获取二值的Mask
        return m

    def image_reference(self, image_id):
        """
        返回一个对应图片ID的链接
        注：此ID为图片内部ID而非真实ID，根据内部ID我们可以找到真实ID
        返回的链接可能最后不是有效的
        """
        info = self.image_info[image_id]
        if info["source"] == "blind":
            if info.__contains__("coco_url"):
                return info["coco_url"]
            elif info.__contains__("flickr_url"):
                return info["flickr_url"]
            else:
                return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super().image_reference(image_id)