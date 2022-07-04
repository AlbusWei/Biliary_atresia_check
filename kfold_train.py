import paddle
import os
from ResNet import ResNet50, ResNet152
from SwinT import swin_tiny, SwinTransformer_base_patch4_window7_224
from CvT import cvt_21_224
from CSwin import CSWinTransformer_tiny_224, CSWinTransformer_base_224
from ViT import ViT_small_patch16_224, ViT_base_patch16_224
from DeiT import DeiT_tiny_patch16_224
from dataPretreatment import kfold_data_generate, kfold_dataset, generate_dataset
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from senet import SE_ResNeXt50_vd_32x4d, SENet154_vd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import classification_report
import seaborn as sns

model_dict = {
    "ResNet152": {
        "function": ResNet152,
        "save_dir": './checkpoint/ResNet152',
        "pretrain": 'pretrain/ResNet152_pretrained.pdparams'
    },
    "ViT": {
        "function": ViT_base_patch16_224,
        "save_dir": './checkpoint/ViT',
        "pretrain": 'pretrain/ViT_base_patch16_224_pretrained.pdparams'
    },
    "SwinT": {
        "function": SwinTransformer_base_patch4_window7_224,
        "save_dir": './checkpoint/SwinT',
        "pretrain": 'pretrain/SwinTransformer_base_patch4_window7_224_pretrained.pdparams'
    },
    "CSwinT": {
        "function": CSWinTransformer_base_224,
        "save_dir": './checkpoint/CSwin',
        "pretrain": 'pretrain/CSWinTransformer_base_224_pretrained.pdparams'
    },
    "CvT": {
        "function": cvt_21_224,
        "save_dir": './checkpoint/CvT',
        "pretrain": 'pretrain/cvt_21_224_pt.pdparams'
    },
    "SeNet": {
        "function": SE_ResNeXt50_vd_32x4d,
        "save_dir": 'checkpoint/SeNet',
        "pretrain": 'pretrain/SE_ResNeXt50_vd_32x4d_pretrained.pdparams',
        "final": "checkpoint/SeNet/final.pdparams"
    },
    "Se154": {
        "function": SENet154_vd,
        "save_dir": 'checkpoint/Se154',
        "pretrain": 'pretrain/SENet154_vd_pretrained.pdparams',
        "final": "checkpoint/Se154/final.pdparams"
    }
}


def train_with_model(model_name, train_loader, valid_loader, save_dir=None, resume_train=False, lr=0.01, BATCH_SIZE=16,
                     EPOCHS=100, callback=None):
    model_method = model_dict[model_name]["function"]
    if save_dir is None:
        save_dir = model_dict[model_name]["save_dir"]

    model = paddle.Model(model_method(class_num=2))
    optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())

    # model.summary((1, 3, 224, 224))

    if resume_train:
        model.load(model_dict[model_name]["final"], skip_mismatch=True)
    else:
        model.load(model_dict[model_name]["pretrain"], skip_mismatch=True)

    model.prepare(optimizer, CrossEntropyLoss(), Accuracy())
    # 启动训练
    model.fit(train_loader,
              valid_loader,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              eval_freq=5,  # 多少epoch 进行验证
              save_freq=5,  # 多少epoch 进行模型保存
              log_freq=100,  # 多少steps 打印训练信息
              save_dir=save_dir,
              callbacks=callback)
    return model


def modeloutput(model, images, mode="paddleClas"):
    if mode == "paddleClas":
        out = model.predict_batch(images)
        # print(out)
        out = paddle.to_tensor(out)
        out = paddle.nn.functional.softmax(out)[0][0]

    else:
        out = model.predict_batch(images)
        out = paddle.to_tensor(out)
        out = paddle.nn.functional.softmax(out)[0]

    y = out.tolist()
    # out = np.argmax(out)
    return y

def main():
    # 所有文件路径,看情况修改
    train_path_list = ["work/Original_train_dataset", "work/MobilePhone_train_dataset/doctor_A",
                       "work/MobilePhone_train_dataset/doctor_C", "work/MobilePhone_train_dataset/doctor_D",
                       "work/MobilePhone_train_dataset/doctor_E", "work/MobilePhone_train_dataset/doctor_F"]
    test_path_list = ["work/Original_test_dataset", "work/MobilePhone_test_dataset/doctorA",
                      "work/MobilePhone_test_dataset/doctorB", "work/MobilePhone_test_dataset/doctorC",
                      "work/MobilePhone_test_dataset/doctorD", "work/MobilePhone_test_dataset/doctorE",
                      "work/MobilePhone_test_dataset/doctorF", "work/MobilePhone_test_dataset/doctorG"]
    for ep in range(5):
        for i in train_path_list:
            kfold_data_generate(i, "train", ep=ep)

    for i in test_path_list:
        generate_dataset(i, "test")

    train_loaders, val_loaders, test_loader = kfold_dataset()
    callback = paddle.callbacks.VisualDL(log_dir='log/')

    model_list = []
    result = []

    for i in range(len(train_loaders)):
        train_loader = train_loaders[i]
        valid_loader = val_loaders[i]
        sd = os.path.join("checkpoint/kfold", str(i))
        model = train_with_model("CSwinT", train_loader, valid_loader, save_dir=sd, EPOCHS=210, callback=callback)

        # 测试
        result.append(model.evaluate(test_loader, log_freq=30, verbose=2))
        model_list.append(model)

    dataiter = iter(test_loader)
    # 融合测试
    y_score = list()
    pre_label = list()
    true_label = list()
    for images, labels in dataiter:
        _y = [0, 0]
        for m in model_list:
            y = modeloutput(m, images, mode="paddleClas")
            _y[0] += y[0]
            _y[1] += y[1]
        # y = modeloutput(model3, images,mode="PASSL")
        # _y[0] += y[0]
        # _y[1] += y[1]
        # y[0]/=6
        # y[1]/=6
        y[0] /= 3
        y[1] /= 3
        out = np.argmax(_y)
        y_score.append(_y)
        pre_label.append(out)
        true_label.append(labels.numpy()[0])

    confusion = confusion_matrix(true_label, pre_label)  # 计算混淆矩阵
    plt.figure(figsize=(7, 7))
    sns.heatmap(confusion, cmap='Blues_r', annot=True, fmt='.20g', annot_kws={'size': 20, 'weight': 'bold', })  # 绘制混淆矩阵
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.show()
    print("混淆矩阵为：\n{}".format(confusion))
    print("\n计算各项指标：")
    print(classification_report(true_label, pre_label, digits=4))

    # 绘制ROC曲线

    plt.figure(figsize=(8, 8))
    kind = {"non_BA": 0, 'BA': 1}
    y_score = np.array(y_score)
    fpr, tpr, threshold = roc_curve(true_label, y_score[:, kind['non_BA']], pos_label=kind['non_BA'])
    roc_auc = auc(fpr, tpr)  ###计算auc的
    fpr1, tpr1, threshold = roc_curve(true_label, y_score[:, kind['BA']], pos_label=kind['BA'])
    roc_auc1 = auc(fpr1, tpr1)  ###计算auc的
    plt.plot(fpr, tpr, marker='o', markersize=5, label='non_BA')
    plt.plot(fpr1, tpr1, marker='*', markersize=5, label='BA')
    plt.title("non_BA AUC:{:.4f}, BA AUC:{:.4f}".format(
        roc_auc, roc_auc1))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc=4)
    plt.show()

if __name__ == "__main__":
    main()
