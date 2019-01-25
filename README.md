# 垃圾短信分类
## 介绍
    二类分类问题， 垃圾邮件认定为正类Positive， 正常则为负类Negative
## 涉及工具包
    pip install keras
    pip install sklearn
    pip install tensorflow
    pip install flask
## 步骤
1. 特征挖掘
    - 数据清理
        - 过滤标点符号
        - 过滤停用词
        - 小写转化
        - 过滤非英文单词
        - 过滤短词汇（长度为1）
    
    - TF-IDF
    
        将语料库文本数据转化为特征向量
2. 分类器选择
    - 朴素贝叶斯
    - 决策树
    - SVM
    - 神经网络
    
        通过hyperas调校神经网络模型参数，以得到表现最好的模型
        
            model = Sequential()
            model.add(Dense(np.power(2, 6), input_dim = len(self.X_trn[0])))
            model.add(LeakyReLU(.11162758792630445))
            model.add(Dropout(.64640024226702))
            model.add(Dense(np.power(2, 7)))
            model.add(LeakyReLU(.2613309282324268))
            model.add(Dropout(.6920257899742139))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy']) 
        
3. 评估指标
    - 测试集综合准确度
    
        TP + TN / Total.N
    - 精确值（Precision）
    
        TP / (TP + FP)
    - 召回率（Recall）
    
        TP / (TP + FN)
    - F1 Score
        （精确值和召回率的调和平均值）
    
        precision * recall * 2 / (precision + recall)

## 测试

    运行命令：
        set FLASK_APP=Ser.py
        flask run
    
    以post方式发送http请求http://localhost:5000/check，请求主体文本以JSON格式{'msg':'what's your msg?'}
    返回结果为True，代表为垃圾短信， False为正常
     


    
