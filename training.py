import paddle
from ResNet import ResNet50, ResNet152
from SwinT import swin_tiny, SwinTransformer_base_patch4_window7_224
from CvT import cvt_21_224
from CSwin import CSWinTransformer_tiny_224, CSWinTransformer_base_224
from ViT import ViT_small_patch16_224, ViT_base_patch16_224
from DeiT import DeiT_tiny_patch16_224
from dataPretreatment import generate_dataloader
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy

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
    "DeiT": {
        "function": DeiT_base_distilled_patch16_224,
        "save_dir": './checkpoint/DeiT',
        "pretrain": 'pretrain/DeiT_base_distilled_patch16_224_pretrained.pdparams',
        "final": "checkpoint/DeiT/final.pdparams"
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
    }
}


def train_with_model(model_name, train_loader, valid_loader, resume_train=False,BATCH_SIZE=16, EPOCHS=100, callback=None):
    model_method = model_dict[model_name]["function"]
    save_dir = model_dict[model_name]["save_dir"]

    model = paddle.Model(model_method(class_num=2))
    optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

    model.summary((1, 3, 224, 224))

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



def resnet_train(train_loader, valid_loader, save_dir='./checkpoint/ResNet152', callback=None):
    BATCH_SIZE = 16
    EPOCHS = 50  # 训练次数

    # model = paddle.Model(ResNet50(num_classes=2))
    model = paddle.Model(ResNet152(class_num=2))
    # beta1 = paddle.to_tensor([0.9], dtype="float32")
    # beta2 = paddle.to_tensor([0.99], dtype="float32")

    # optimizer = paddle.optimizer.AdamW(learning_rate=0.01,
    #                                    parameters=model.parameters(),
    #                                    beta1=beta1,
    #                                    beta2=beta2,
    #                                    weight_decay=0.01)
    optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())

    # model.summary((1, 3, 224, 224))

    # model.load('pretrain/ResNet152_pretrained.pdparams',skip_mismatch=True)
    model.load('checkpoint/ResNet152/10.pdparams', skip_mismatch=True)

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


def swinT_train(train_loader, valid_loader, save_dir='./checkpoint/SwinT', callback=None):
    BATCH_SIZE = 256
    EPOCHS = 200  # 训练次数

    model = paddle.Model(swin_tiny(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    model.load('pretrain/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams', skip_mismatch=True)

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


def ViT_train(train_loader, valid_loader, save_dir='./checkpoint/ViT', callback=None):
    BATCH_SIZE = 512
    EPOCHS = 200  # 训练次数

    model = paddle.Model(ViT_small_patch16_224(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    # 加载预训练权重
    model.load('pretrain/ViT_small_patch16_224_pretrained.pdparams', skip_mismatch=True)

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


def ViT_base_train(train_loader, valid_loader, save_dir='./checkpoint/ViT_base', callback=None):
    BATCH_SIZE = 16
    EPOCHS = 200  # 训练次数

    model = paddle.Model(ViT_base_patch16_224(class_num=2))
    # beta1 = paddle.to_tensor([0.9], dtype="float32")
    # beta2 = paddle.to_tensor([0.99], dtype="float32")

    # optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
    #                                    parameters=model.parameters(),
    #                                    beta1=beta1,
    #                                    beta2=beta2,
    #                                    weight_decay=0.01)

    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

    # model.summary((1, 3, 224, 224))

    # 加载预训练权重
    model.load('pretrain/ViT_base_patch16_224_pretrained.pdparams', skip_mismatch=True)

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


def CvT_train(train_loader, valid_loader, save_dir='./checkpoint/CvT', callback=None):
    BATCH_SIZE = 16
    EPOCHS = 200  # 训练次数

    model = paddle.Model(cvt_21_224(class_num=2))
    # beta1 = paddle.to_tensor([0.9], dtype="float32")
    # beta2 = paddle.to_tensor([0.99], dtype="float32")

    # optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
    #                                    parameters=model.parameters(),
    #                                    beta1=beta1,
    #                                    beta2=beta2,
    #                                    weight_decay=0.01)

    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

    # model.summary((1, 3, 224, 224))

    # 加载预训练权重
    model.load('pretrain/cvt_21_224_pt.pdparams', skip_mismatch=True)

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


def CSwin_train(train_loader, valid_loader, save_dir='./checkpoint/CSwin', callback=None):
    BATCH_SIZE = 512
    EPOCHS = 200  # 训练次数

    model = paddle.Model(CSWinTransformer_tiny_224(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    # 加载预训练权重
    model.load('pretrain/CSWinTransformer_tiny_224_pretrained.pdparams', skip_mismatch=True)

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


def DeiT_train(train_loader, valid_loader, save_dir='./checkpoint/DeiT', callback=None):
    BATCH_SIZE = 512
    EPOCHS = 200  # 训练次数

    model = paddle.Model(DeiT_tiny_patch16_224(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    # 加载预训练权重
    model.load('pretrain/DeiT_tiny_patch16_224_pretrained.pdparams', skip_mismatch=True)

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


def main():
    # 文件地址
    train_txt = "work/train_list.txt"
    test_txt = "work/test_list.txt"
    val_txt = "work/val_list.txt"

    callback = paddle.callbacks.VisualDL(log_dir='log/')

    train_loader, valid_loader, test_loader = generate_dataloader()
    # trn_dateset,val_dateset,test_dateset = dataset_without_batch()
    # preview(train_loader)

    # model = resnet_train(train_loader, valid_loader,callback=callback)
    # model = resnet_train(trn_dateset, val_dateset,callback=callback)
    model = CvT_train(train_loader, valid_loader, callback=callback)
    # model = ViT_base_train(train_loader, valid_loader,callback=callback)
    # model = swinT_train(train_loader, valid_loader,callback)
    # model = CSwin_train(train_loader, valid_loader,callback=callback)
    # model = DeiT_train(train_loader, valid_loader,callback=callback)

    # 测试
    model.evaluate(test_loader, log_freq=30, verbose=2)
    # model.evaluate(test_dateset, log_freq=30, verbose=2)


if __name__ == "__main__":
    main()
