# AI-script

AP_and_PR_calc.py:
    classname = 'person' # 待检测路径
    dataset_path = f"D:\\dataset" # 数据集路径
    predict_path = f"{dataset_path}\\img_cvlm_predict\\{{}}.xml" # 预测结果 .xml文件集合路径（VOC格式）
    det_path = f"{dataset_path}\\det_rst_{{}}.txt" # 预测结果集合路径（ (image_name, confidence, xmin, ymin, xmax, ymax)((image_name, confidence, xmin, ymin, xmax, ymax))格式 ）
    anno_path = f"{dataset_path}\\img_ground_truth\\{{}}.xml" # GT .xml文件集合路径（VOC格式）
    image_name_set_path = f"{dataset_path}\\test.txt" # 图片名集合路径
    cache_path = dataset_path # 缓存路径
