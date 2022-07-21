# torch-template

pytorch训练神经网络的模板代码

运行代码：
```
#注意，运行第二次时，因为会自动加载第一次的checkpoint，所以会瞬间训练完。
sh scripts/train_remote.sh 
```

典型特征
+ 将模型和数据集相关的参数保存为两个配置文件，自动解析
+ 会自动按照每个模型配置文件的名字新建checkpoint文件夹，然后保存模型和日志
+ 自动加载checkpoint，包括模型，优化器
+ 标准的训练框架