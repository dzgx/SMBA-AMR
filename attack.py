import os
import warnings
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
Victim_Label = np.array([0, 1, 2, 3, 4, 5]) 
Target_Label = 6
Poisoning_Rate = 0.01


import h5py
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import GRUModel
# import LSTMModel
import MCNET
import ResNet
# import MCLDNN
from tools import show_history
from get_classes import get_classes

# 检查文件夹是否存在，不存在则创建
if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./figure'):
    os.makedirs('./figure')

def mix_signals(X_train):
    N = X_train.shape[0]  # 样本数量
    L = X_train.shape[2]  # 信号长度
    mixed_X = np.zeros_like(X_train)
    noise_data = np.zeros_like(X_train)
    
    for i in range(N):
        I = X_train[i, 0, :]
        Q = X_train[i, 1, :]
        
        # 计算信号幅度
        signal_amplitude = np.max(np.abs(I + 1j*Q))
        
        # 生成高斯噪声
        np.random.seed(2016)
        noise_I = np.random.normal(0, 1, int(L/16)) * signal_amplitude
        noise_Q = np.random.normal(0, 1, int(L/16)) * signal_amplitude

        # 组合复数信号
        original_complex = I + 1j*Q
        noise_complex = noise_I + 1j*noise_Q
        
        # 计算FFT
        original_fft = np.fft.fft(original_complex)
        amplitude_spectrum = np.abs(original_fft)
        phase_spectrum = np.angle(original_fft)
        
        noise_fft = np.fft.fft(noise_complex)
        noise_amplitude = np.abs(noise_fft)
        
        # 混合幅度谱 (信号和噪声各占50%)
        # 随机选择插入位置
        max_start = len(amplitude_spectrum) - len(noise_amplitude)
        start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
        end_idx = start_idx + len(noise_amplitude)
        
        # 混合幅度谱 (在随机位置插入噪声)
        mixed_amplitude = amplitude_spectrum.copy()
        mixed_amplitude[start_idx:end_idx]  = mixed_amplitude[start_idx:end_idx] + noise_amplitude

        # 重建混合信号
        mixed_spectrum = mixed_amplitude * np.exp(1j * phase_spectrum)
        mixed_signal = np.fft.ifft(mixed_spectrum)
        
        # 存储结果
        mixed_X[i, 0, :] = np.real(mixed_signal)
        mixed_X[i, 1, :] = np.imag(mixed_signal)
    
    return mixed_X

def main():

    n_classes = len(get_classes(from_file="./classes_rml20.txt"))
    # 加载数据集
    train_data = h5py.File('./rml20_train_8_10_data.hdf5', 'r')
    val_data = h5py.File('./rml20_val_1_10_data.hdf5', 'r')

    X_train = train_data['X_train'][:, :, :]
    Y_train = train_data['Y_train'][:].astype(np.int32)
    X_val = val_data['X_val'][:, :, :]
    Y_val = val_data['Y_val'][:].astype(np.int32)

    X_train = X_train.swapaxes(2, 1)
    X_val = X_val.swapaxes(2, 1)

    for i in range(Victim_Label.shape[0]):
        victim_class_index = np.where(Y_train == i)[0]  # 找到训练集中目标类别的索引
        #构造投毒训练集
        poison_num = int(victim_class_index.shape[0] * Poisoning_Rate)  # 计算投毒样本的数量
        print("poison num:", poison_num)  # 打印投毒样本的数量
        poison_index = np.random.choice(victim_class_index, poison_num, replace=False)  # 随机选择投毒样本的索引
        print(poison_index.shape)
        X_train[poison_index, :, :] = mix_signals(X_train[poison_index, :, :])  # 部署触发器到训练集中目标类别的样本
        Y_train[poison_index] = Target_Label

    ######################################################
    # MCNET ResNet
    X_train=np.expand_dims(X_train,axis=3)
    X_val=np.expand_dims(X_val,axis=3)
    ######################################################
    # GRU
    # X_train = X_train.swapaxes(2, 1)
    # X_val = X_val.swapaxes(2, 1)
    ######################################################
    # MCLDNN
    # X1_train=np.expand_dims(X_train[:,:,0], axis=2)
    # X1_val=np.expand_dims(X_val[:,:,0],axis=2)
    # X2_train=np.expand_dims(X_train[:,:,1], axis=2)
    # X2_val=np.expand_dims(X_val[:,:,1],axis=2)
    # X_train = X_train.swapaxes(2, 1)
    # X_val = X_val.swapaxes(2, 1)
    # X_train=np.expand_dims(X_train,axis=3)
    # X_val=np.expand_dims(X_val,axis=3)
    ######################################################
    
    # 数据预处理
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=n_classes)
    Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=n_classes)

    # 关闭数据文件
    train_data.close()
    val_data.close()

    # 定义模型
    # model = GRUModel.GRUModel(input_shape=(1000, 2), classes=n_classes)
    # model = ResNet.ResNet(input_shape=(2, 1000, 1), classes=n_classes)
    model = MCNET.MCNET(input_shape=(2, 1000, 1), classes=n_classes)
    # model = LSTMModel.LSTMModel(input_shape=(128, 2), classes=n_classes)
    # model = MCLDNN.MCLDNN(classes=n_classes)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    # 绘制模型结构
    # plot_model(model, to_file='./figure/model.png', show_shapes=True)
    model.summary()

    # 定义回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./weights/weights_epoch_{epoch:03d}-acc_{accuracy:.4f}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='auto'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True,
        )
    ]

    ########################################################
    # mcnet Transformer
    # 训练模型
    history = model.fit(
        X_train,
        Y_train,
        batch_size=400,
        epochs=1000,
        verbose=1,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )
    ########################################################

    ########################################################
    # MCLDNN
    # 训练模型
    # history = model.fit(
    #     [X_train,X1_train,X2_train],
    #     Y_train,
    #     batch_size=400,
    #     epochs=1000,
    #     verbose=1,
    #     validation_data=([X_val,X1_val,X2_val], Y_val),
    #     callbacks=callbacks
    # )
    ########################################################

    # 绘制训练历史
    show_history(history)

if __name__ == "__main__":
    main()