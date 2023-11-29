# 计算 VOC格式GT和预测结果 的 AP 及 P-R Curve

import numpy as np
import os
import pickle
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_rec(filename):
    """
    解析PASCAL VOC xml文件
    return：dict list [{'name': xxx, 'bbox': [xmin, ymin, xmax, ymax]},{},....]
    """

    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {
            'name': obj.find('name').text,
            'pose': obj.find('pose').text,
            'truncated': int(obj.find('truncated').text),
            'difficult': int(obj.find('difficult').text)
        }
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(_recall, _precision, use_07_metric=False):
    """
    给定recall和precision之后计算返回AP，其中recall是从小到大排序，precision每一个元素是对应排序后的recall的值
    _recall: np.array
    _precision: np.array
    use_07_metric:如果为True则采用07年的方式计算AP
    """

    if use_07_metric:  # VOC在2010之后换了评价方法，所以决定是否用07年的
        _ap = 0.
        for t in np.arange(0., 1.1, 0.1):  # 07年的采用11个点平分recall来计算
            if np.sum(_recall >= t) == 0:
                p = 0
            else:
                p = np.max(_precision[_recall >= t])  # 取一个recall阈值t之后最大的precision
            _ap = _ap + p / 11.  # 将11个precision加和平均
    else:
        # 这里是使用VOC2010年后的方法求mAP，计算光滑后PR曲线的面积，不再是固定的11个点
        # 在recall的首和尾添加值来完成更好完成
        # 在precision的尾部添加0为了更好得到“光滑”后的precision的值
        mrec = np.concatenate(([0.], _recall, [1.]))  # recall和precision前后分别加了一个值，因为recall最后是1，所以
        mpre = np.concatenate(([0.], _precision, [0.]))  # 右边加了1，precision加的是0

        # 调整mpre，从后往前取最大值，保证precision单调不增。
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # 计算PR曲线下的面积
        # X轴为R（recall的值）
        i = np.where(mrec[1:] != mrec[:-1])[0]  # 返回了所有改变了recall的点的位置
        # 求每个矩形的面积和
        # 具体理解见前文解释
        _ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return _ap


# 计算每个类别对应的AP，mAP是所有类别AP的平均值
# 主要是处理得到rec, prec数组
def voc_eval(det_path, anno_path, image_set_file, classname, cache_dir, ov_thresh=0.5, use_07_metric=False):
    """

    Top level function that does the PASCAL VOC evaluation.
    det_path:
            检测结果的文件路径。检测结果文件的每一行应该是：img_ID，置信度，xmin, ymin, xmax, ymax
            detpath应该是这样的字符串'./results/comp4_det_test_{}.txt'
            detpath.format(classname) should produce the detection results file.
    anno_path:
            Path to annotations
            annopath应该是这样的字符串"dataset/voc/VOC2007/Annotations/{}.xml"
            annopath.format(imagename) should be the xml annotations file.
    image_set_file:
            储存了图片名的text，每一行是一张图片的名。'dataset/voc/VOC2007/ImageSets/Main/test.txt'
    classname:
            类别名
    cache_dir:
            用于存储注解(annotations)的路径，生成一个pickle_file
    ov_thresh:
            IOU_threshold (default = 0.5)
    use_07_metric:
            是否使用07年的计算ap的方式(default False)
    return：rec, prec, ap
    """

    # 第一步获得各图片的GT
    # 如果不存在注释路径的文件夹，则先创建文件夹
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    cachefile = os.path.join(cache_dir, 'annoys.pkl')

    # 读取图片名，并储存在列表中
    with open(image_set_file, 'r') as f:
        lines = f.readlines()
    image_names = [x.strip() for x in lines]

    recs = dict()
    print("Reading annotation:")
    for i, image_name in enumerate(tqdm(image_names)):
        # 获取图片中对应的GT的解析
        recs[image_name] = parse_rec(anno_path.format(image_name))

    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
        pickle.dump(recs, f)

    # 从上面的recs中提取出我们要判断的那类目标的标注信息(GT)
    class_recs = dict()
    npos = 0
    for image_name in image_names:
        # [obj，obj，....] 每个obj={'name': xxx, 'bbox': [xmin, ymin, xmax, ymax]}
        R = [obj for obj in recs[image_name] if obj['name'] == classname]
        # 二维数组，(number_obj,4),该张图片有number_obj个类别为classname的目标框
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
        # 该图片中该类别对应的所有bbox的是否已被匹配的标志位
        det = [False] * len(R)
        # 累计所有图片中的该类别目标的GT总数，不算diffcult
        npos = npos + sum(~difficult)

        class_recs[image_name] = {
            'bbox': bbox,
            'difficult': difficult,
            'det': det
        }

    # 第二步读取模型识别的结果
    detfile = det_path.format(classname)
    # 读取相应类别的检测结果文件，每一行对应一个检测目标
    with open(detfile, 'r') as f:
        lines = f.readlines()  # 读取所有行

    splitlines = [x.strip().split(' ') for x in lines]  # 处理为[[image_id, 置信度, xmin, ymin, xmax, ymax ],[]...]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])  # 一维数组
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])  # 二维数组，（number_bbox,4）

    # sort by confidence 按置信度由大到小排序
    sorted_ind = np.argsort(-confidence)  # 获得Indx
    # sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]  # 对BB重排序
    image_ids = [image_ids[x] for x in sorted_ind]  # 对image_ids重排序

    # 记下dets并对每个image打上标注是TP还是FP
    nd = len(image_ids)  # 检测结果文件的行数
    tp = np.zeros(nd)  # 用于标记每个检测结果是tp还是fp
    fp = np.zeros(nd)
    for d in range(nd):
        # 取出该条检测结果所属图片中的所有ground truth
        # 其实image_id就是image_name，R={'bbox': bbox(二维数组),'difficult': difficult,'det': [bool]}
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)  # bb一维数组
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)  # 二维数组

        if BBGT.size > 0:
            # compute overlaps  计算与该图片中所有ground truth的最大重叠度(IOU)
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])  # 一维
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih  # 一维

            # 重叠部分面积一维
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            # 计算得到检测结果的这个框与该张图片的所有该类比的GT的IOU，一维
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        # 这里就是具体的分配TP和FP的规则了
        # 如果最大的重叠度大于一定的阈值（IOU_threshold）
        if ovmax > ov_thresh:
            # 如果最大重叠度对应的ground truth为difficult就忽略，因为上面npos就没算
            if not R['difficult'][jmax]:
                # 如果对应的最大重叠度的ground truth以前没被匹配过则匹配成功，即tp
                if not R['det'][jmax]:
                    tp[d] = 1.
                    # 表示框被匹配过了
                    R['det'][jmax] = 1
                else:
                    # 若之前有置信度更高的检测结果匹配过这个ground truth，则此次检测结果为fp
                    fp[d] = 1.
        else:
            # 该图片中没有对应类别的目标ground truth或者与所有ground truth重叠度都小于阈值
            fp[d] = 1.

    # 计算 precision recall
    # 累加函数np.cumsum([1, 2, 3, 4]) -> [1, 3, 6, 10]
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # tp/GT,也就得到了voc_ap函数所需要的rec了
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # 避免除以零
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def extract_selected_classes_info(_predict_path, _groundtruth_path, filename, candidate_class):
    """
    解析PASCAL VOC xml文件 提取候选类
    return：[{'name': xxx, 'bbox': [xmin, ymin, xmax, ymax]},{},....]
    """

    gt_path = os.path.join(_groundtruth_path.format(item[:-4]))
    gt_tree = ET.parse(gt_path)
    gt_width = gt_tree.find('size').find('width').text
    gt_height = gt_tree.find('size').find('height').text

    item_path = os.path.join(_predict_path.format(item[:-4]))
    tree = ET.parse(item_path)
    objects = []
    pred_width = tree.find('size').find('width').text
    pred_height = tree.find('size').find('height').text

    width_proportion = int(gt_width) / int(pred_width)
    height_proportion = int(gt_height) / int(pred_height)
    if gt_width != pred_width or gt_height != pred_height:
        print(f"GT_img: {gt_width}*{gt_height} \t"
              f"Pred_img: {pred_width}*{pred_height} \t "
              f"proportion: {str(width_proportion)[:6]}")

    for obj in tree.findall('object'):
        if obj.find('name').text != candidate_class:
            continue
        obj_struct = {
            'class': obj.find('name').text,
            'truncated': int(obj.find('truncated').text),
            'difficult': int(obj.find('difficult').text),
            'confidence': float(obj.find('confidence').text)
        }
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(bbox.find('xmin').text) * width_proportion,
            int(bbox.find('ymin').text) * height_proportion,
            int(bbox.find('xmax').text) * width_proportion,
            int(bbox.find('ymax').text) * height_proportion
        ]
        obj_struct['file_name'] = filename[:-4]
        objects.append(obj_struct)

    return objects


if __name__ == "__main__":
    classname = 'person'
    dataset_path = f"D:\\CVlm-Show\\dataset\\ehs_200"
    predict_path = f"{dataset_path}\\img_cvlm_predict\\{{}}.xml"
    det_path = f"{dataset_path}\\det_rst_{{}}.txt"
    anno_path = f"{dataset_path}\\img_ground_truth\\{{}}.xml"
    image_set_file = f"{dataset_path}\\test.txt"
    cache_dir = dataset_path

    predict_list_xml = os.listdir(f"{dataset_path}\\img_cvlm_predict")
    candidate_class_list = list()
    for item in predict_list_xml:
        temp_list = extract_selected_classes_info(predict_path, anno_path, item, classname)
        for itm in temp_list:
            candidate_class_list.append((
                itm['file_name'],
                itm['confidence'],
                itm['bbox'][0],
                itm['bbox'][1],
                itm['bbox'][2],
                itm['bbox'][3]
            ))
    with open(det_path.format(classname), 'w') as f:
        for item in candidate_class_list:
            f.write(f"{item[0]} {item[1]} {item[2]} {item[3]} {item[4]} {item[5]}\n")
    f.close()

    recalls, precisions, ap = voc_eval(det_path, anno_path, image_set_file, classname, cache_dir)
    print(f"Class {classname} AP: {str(ap)[:8]}")

    # P-R Curve
    plt.title('P-R Curve')  # 标题
    plt.xlabel('Recalls', fontsize=14)  # x轴标签
    plt.ylabel('Precisions', fontsize=14)  # y轴标签
    plt.plot(recalls, precisions)  # 画线
    plt.show()
