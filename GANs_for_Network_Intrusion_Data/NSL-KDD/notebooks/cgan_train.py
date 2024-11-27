import sys
sys.path.append('../utils')


import numpy as np
import pandas as pd
from collections import defaultdict
import pickle, os, itertools
from tqdm import tqdm  # tqdm 用于显示进度条
import time
import utils , preprocessing  

import tensorflow as tf
from keras.layers import Dense, Input, Dropout,concatenate
from keras.models import Model
from keras import backend as K
from scipy.stats import norm
import inspect


import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(12343)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置 TensorFlow 仅输出 ERROR 级别的信息
tf.get_logger().setLevel('ERROR')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def plot_summary(d_l, g_l,acc_r,acc_g, m =''):
    n = np.arange(len(d_l))
    title = 'Loss and Accuracy plot'+'\n'+ m
    title = title.replace('.pickle','')
    fig, axs = plt.subplots(2,figsize=(19.20,10.80))

    axs[0].set_title(title,fontsize=20.0,fontweight="bold")
    axs[0].plot(n, g_l,label='Generator loss',linewidth=4)
    axs[0].plot(n, d_l,label='Discriminator loss',linewidth=4)
    axs[0].legend(loc=0, prop={'size': 20})
    axs[0].set_ylabel('Loss',fontsize=20.0,fontweight="bold")
    axs[0].tick_params(labelsize=20)
    axs[0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False,labelsize=20)

    # axs[1].plot(n, acc,'r',label='Discriminator accuracy',linewidth=4)
    axs[1].plot(n, acc_g,label='Accuracy on Generated',linewidth=4)
    axs[1].plot(n, acc_r,label='Accuracy on Real',linewidth=4)
    axs[1].legend(loc=0,prop={'size': 20})
    axs[1].set_ylabel('Accuracy',fontsize=20.0,fontweight="bold")
    axs[1].set_xlabel('Ephoc',fontsize=20.0,fontweight="bold")
    axs[1].tick_params(labelsize=20)

    plt.tight_layout()
    #plt.show()
    if not os.path.exists("imgs"):
        os.makedirs('imgs')
    plt.savefig(f'imgs/{m[:-7]}.png',dpi = 300)
    plt.close('all') #plt.close(fig)

class CGAN():
    """Conditinal Generative Adversarial Network class"""
    
    def __init__(self,arguments,X,y):
        [self.rand_noise_dim, self.tot_epochs, self.batch_size,self.D_epochs, \
         self.G_epochs,self.learning_rate, self.n_layers, self.activation,self.optimizer, self.min_num_neurones] = arguments

        self.X_train = X
        self.y_train = y

        self.label_dim = y.shape[1]
        self.x_data_dim = X.shape[1]

        self.g_losses = []
        self.d_losses, self.disc_loss_real, self.disc_loss_generated = [], [], []
        self.acc_history = []
        
        self.__define_models()
        self.gan_name = '_'.join(str(e) for e in arguments).replace(".","")
        
        self.terminated = False

    def build_generator(self,x,labels):
        """Create the generator model G(z,l) : z -> random noise , l -> label (condition)"""
        
        x = concatenate([x,labels])
        for i in range(1,self.n_layers+1):
            x = Dense(self.min_num_neurones*i, activation=self.activation)(x)
            
        x = Dense(self.x_data_dim)(x)
        x = concatenate([x,labels])

        return x

    def build_discriminator(self,x):
        """Create the discrimnator model D(G(z,l)) : z -> random noise , l -> label (condition)"""
        
        for n in reversed(range(1,self.n_layers+1)):
            x = Dense(self.min_num_neurones*n, activation=self.activation)(x)
        
        x = Dense(1, activation='sigmoid')(x)

        return x

    def __define_models(self):
        """Define Generator, Discriminator & combined model"""
        
        # Create & Compile generator
        generator_input = Input(shape=(self.rand_noise_dim,))
        labels_tensor = Input(shape=(self.label_dim,))
        generator_output = self.build_generator(generator_input, labels_tensor)

        self.generator = Model(inputs=[generator_input, labels_tensor], outputs=[generator_output], name='generator')
        self.generator.compile(loss='binary_crossentropy',optimizer=self.optimizer, metrics=['accuracy'])
        K.set_value(self.generator.optimizer.lr,self.learning_rate)
        

        # Create & Compile generator
        discriminator_model_input = Input(shape=(self.x_data_dim + self.label_dim,))
        discriminator_output = self.build_discriminator(discriminator_model_input)

        self.discriminator = Model(inputs=[discriminator_model_input],outputs=[discriminator_output],name='discriminator')
        self.discriminator.compile(loss='binary_crossentropy',optimizer=self.optimizer, metrics=['accuracy'])
        K.set_value(self.discriminator.optimizer.lr,self.learning_rate)

        # Build "frozen discriminator"
        frozen_discriminator = Model(inputs=[discriminator_model_input],outputs=[discriminator_output],name='frozen_discriminator')
        frozen_discriminator.trainable = False

        # Debug 1/3: discriminator weights
        n_disc_trainable = len(self.discriminator.trainable_weights)

        # Debug 2/3: generator weights
        n_gen_trainable = len(self.generator.trainable_weights)

        # Build & compile combined model from frozen weights discriminator
        combined_output = frozen_discriminator(generator_output)
        self.combined = Model(inputs = [generator_input, labels_tensor],outputs = [combined_output],name='adversarial_model')
        self.combined.compile(loss='binary_crossentropy',optimizer=self.optimizer, metrics=['accuracy'])
        K.set_value(self.combined.optimizer.lr,self.learning_rate)

        # Debug 3/3: compare if trainable weights correct
        assert(len(self.discriminator.trainable_weights) == n_disc_trainable)
        assert(len(self.combined.trainable_weights) == n_gen_trainable)
        
    def __get_batch_idx(self):
        """random selects batch_size samples indeces from training data"""
        
        batch_ix = np.random.choice(len(self.X_train), size=self.batch_size, replace=False)

        return batch_ix
    
    def dump_to_file(self,save_dir = "./logs"):
        """Dumps the training history and GAN config to pickle file """
        
        H = defaultdict(dict)
        H["acc_history"] = self.acc_history
        H["Generator_loss"] = self.g_losses
        H["disc_loss_real"] = self.disc_loss_real
        H["disc_loss_gen"] = self.disc_loss_generated
        H["discriminator_loss"] = self.d_losses
        H["rand_noise_dim"] , H["total_epochs"] = self.rand_noise_dim, self.tot_epochs
        H["batch_size"] , H["learning_rate"]  = self.batch_size, self.learning_rate
        H["n_layers"] , H["activation"]  = self.n_layers, self.activation
        H["optimizer"] , H["min_num_neurones"] = self.optimizer, self.min_num_neurones
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with open(f"{save_dir}/{self.gan_name}{'.pickle'}", "wb") as output_file:
            pickle.dump(H,output_file)

    def train(self):
        """Trains the CGAN model"""

        # Adversarial ground truths
        real_labels = np.ones((self.batch_size, 1))
        fake_labels = np.zeros((self.batch_size, 1))
        # Adversarial ground truths with noise
        #real_labels = np.random.uniform(low=0.999, high=1.0, size=(self.batch_size,1))
        #fake_labes = np.random.uniform(low=0, high=0.00001, size=(self.batch_size,1))

        for epoch in range(self.tot_epochs):
            #Train Discriminator
            for i in range(self.D_epochs):

                idx = self.__get_batch_idx()
                x, labels = self.X_train[idx], self.y_train[idx]

                #Sample noise as generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.rand_noise_dim))

                #Generate a half batch of new images
                generated_x = self.generator.predict([noise, labels])

                #Train the discriminator
                d_loss_fake = self.discriminator.train_on_batch(generated_x, fake_labels)
                d_loss_real = self.discriminator.train_on_batch(np.concatenate((x,labels),axis=1), real_labels)
                d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)

            self.disc_loss_real.append(d_loss_real[0])
            self.disc_loss_generated.append(d_loss_fake[0])
            self.d_losses.append(d_loss[0])
            self.acc_history.append([d_loss_fake[1],d_loss_real[1]])
            
            #NB: Gradients could be exploding or vanishing (cliping or changing activation function,learning rate could be a solution)
            # if np.isnan(d_loss_real) or np.isnan(d_loss_fake):
            #     self.terminated = True
            #     break
            #NB: Gradients could be exploding or vanishing (cliping or changing activation function, learning rate could be a solution)
            if np.isnan(d_loss_real).any() or np.isnan(d_loss_fake).any():
                self.terminated = True
                break

                
            #Train Generator (generator in combined model is trainable while discrimnator is frozen)
            for j in range(self.G_epochs):
                #Condition on labels
                # sampled_labels = np.random.randint(1, 5, self.batch_size).reshape(-1, 1)
                sampled_labels = np.random.choice([0,2,3,4],(self.batch_size,1), replace=True)

                #Train the generator
                g_loss = self.combined.train_on_batch([noise, sampled_labels], real_labels)
                self.g_losses.append(g_loss[0])

            #Print metrices
            # print ("Epoch : {:d} [D loss: {:.4f}, acc.: {:.4f}] [G loss: {:.4f}]".format(epoch, d_loss[0], 100*d_loss[1], g_loss[0]))


train,test, label_mapping = preprocessing.get_data(encoding="Label")
data_cols = list(train.columns[ train.columns != 'label' ])
x_train , x_test = preprocessing.preprocess(train,test,data_cols,"Robust",True)

y_train = x_train.label.values
y_test = x_test.label.values

data_cols = list(x_train.columns[ x_train.columns != 'label' ])

to_drop = preprocessing.get_contant_featues(x_train,data_cols)
x_train.drop(to_drop, axis=1,inplace=True)
x_test.drop(to_drop, axis=1,inplace=True)

data_cols = list(x_train.columns[ x_train.columns != 'label' ])

att_ind = np.where(x_train.label != label_mapping["normal"])[0]
x = x_train[data_cols].values[att_ind]
y = y_train[att_ind].reshape(-1,1)



if not os.path.exists('logs'):
    os.makedirs('logs')

base_n_count = [17,27,37]  # 调整基本神经元数量的范围，从3-41变为5-51，步长为5
ephocs = [2000]  # 调整周期范围，从100-5000变为500-5500，步长为500
batch_sizes = [128]  # 简化批量大小选择为128, 256, 512
learning_rates = [0.0005]  # 调整学习率范围，从0.01到0.00001，共10个值
num_layers = [8, 10]  # 调整隐藏层数量，范围从2到10

# optimizers = ["sgd", "RMSprop", "adam", "Adagrad", "Adamax","Nadam"]
optimizers = ["sgd","adam"]  # 精简优化器选项，去除一些效果可能较差的
activation_func = ["relu", "tanh","elu"]  # 精简激活函数选项

tot = list(itertools.product([32], ephocs, batch_sizes, [1], [1],
                             learning_rates, num_layers, activation_func, optimizers, base_n_count))



# 记录总开始时间
start_time = time.time()

# 遍历所有组合并训练模型
for idx, i in tqdm(enumerate(tot), total=len(tot), desc="Processing"):
    iteration_start_time = time.time()  # 记录当前组合的开始时间
    args = list(i)
    cgan = CGAN(args, x, y)  # 假设 CGAN 类已经定义好
    cgan.train()  # 开始训练
    if not cgan.terminated:  # 检查训练是否终止
        cgan.dump_to_file()  # 保存训练结果

    iteration_end_time = time.time()  # 记录当前组合的结束时间
    iteration_duration = iteration_end_time - iteration_start_time
    print(f"组合 {idx + 1}/{len(tot)} 训练时间: {iteration_duration:.2f} 秒")

# 记录总结束时间
end_time = time.time()
total_duration = end_time - start_time
print(f"总运行时间: {total_duration:.2f} 秒")