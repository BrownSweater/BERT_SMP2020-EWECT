在SMP2020的微博情绪分类任务上，微调在中文预料上预训练的BERT模型，进行文本分类。



## 1、数据集简介

来源于[微博情绪分析评测(SMP2020-EWECT)](https://smp2020ewect.github.io/)，本届微博情绪分类评测任务一共包含两个数据集：

- 通用微博数据集，其中的微博是随机收集的包含各种话题的数据
- 疫情微博数据集，其中的微博数据均与本次疫情相关。

微博情绪分类任务旨在识别微博中蕴含的情绪，输入是一条微博，输出是该微博所蕴含的情绪类别。在本次评测中，我们将微博按照其蕴含的情绪分为以下六个类别之一：积极、愤怒、悲伤、恐惧、惊奇和无情绪。

|      情绪       | 通用微博数据集                                               | 疫情微博数据集                                               |
| :-------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
|  积极（happy）  | 哥，你猜猜看和喜欢的人一起做公益是什么感觉呢。我们的项目已经进入一个新阶段了，现在特别有成就感。加油加油。 | 愿大家平安、健康[心]#致敬疫情前线医护人员# 愿大家都健康平安  |
|   愤怒(angry)   | 每个月都有特别气愤的时候。，多少个瞬间想甩手不干了，杂七杂八，当我是什么。 | 整天歌颂医护人员伟大的自我牺牲精神，人家原本不用牺牲好吧！吃野味和隐瞒疫情的估计是同一波人，真的要死自己去死，别拉上无辜的人。 |
|    悲伤(sad)    | 回忆起老爸的点点滴滴，心痛…为什么.接受不了                   | 救救武汉吧，受不了了泪奔，一群孩子穿上大人衣服学着救人 请官方不要瞒报谎报耽误病情，求求武汉zf了[泪][泪][泪][泪] |
|   恐惧(fear)    | 明明是一篇言情小说，看完之后为什么会恐怖的睡不着呢，越想越害怕[吃驚] | 对着这个症状，没病的都害怕[允悲][允悲]                       |
| 惊奇(surprise)  | 我竟然不知道kkw是丑女无敌里的那个                            | 我特别震惊就是真的很多人上了厕所是不会洗手的。。。。         |
| 无情绪(neutral) | 我们做不到选择缘分，却可以珍惜缘分。                         | 辟谣，盐水漱口没用。                                         |



## 2、数据预处理

本次训练中只使用了**通用微博数据集**，针对数据集的规模、类别分布、数据质量进行分析、处理。



### 2.1、通用数据集大小

通用数据集训练集、验证集、测试集数据集大小：

|            | Train | Val  | Test |
| ---------- | ----- | ---- | ---- |
| 通用数据集 | 27768 | 2000 | 5000 |

> VAL 只有 Train的7%，在训练时应该提高val的大小，防止验证结果有偏。

### 2.2、通用数据集类别分布

>  usual train

![image-20211227025412895](/Users/wjj/Library/Application Support/typora-user-images/image-20211227025412895.png)



>  usual val

![image-20211227025646888](/Users/wjj/Library/Application Support/typora-user-images/image-20211227025646888.png)



> usual test

![image-20211227030140100](/Users/wjj/Library/Application Support/typora-user-images/image-20211227030140100.png)



训练集、验证集、测试集的类别比例基本一致，但是存在明显的==样本不均衡==，fear类仅有4%。

### 2.3、通用数据集清洗

进行了全角转半角、繁转简、英文大写转小写、去除url、去除email、去除@以及保留emoji等操作，下表展示了部分清洗数据，在模型处理中，我们限制数据的最大长度为140。

> 参考代码：https://github.com/thinkingmanyangyang/smp-ewect-code/blob/master/clean_data.py

| 清洗策略   | 清洗前                                                       | 清洗后                                                       |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 繁简体转换 | 昨個回髮廊洗頭結果設計師竟然想吹直我的頭髮！嘿小姐妳幫我用卷的竟敢吹直它妳可以對我沒印象但妳總該看一下吧！嗯嗯我不會再去了！ | 昨个回发廊洗头结果设计师竟然想吹直我的头发!嘿小姐你帮我用卷的竟敢吹直它你可以对我没印象但你总该看一下吧!嗯嗯我不会再去了! |
| 微博@标签  | 每次遇见这样，我都想下去手撕这样违法吗@陈震同学              | 每次遇见这样,我都想下去手撕这样违法吗                        |
| URL        | 对于结婚我可能真低调到了，生完娃仍然有许多人发来☞你什么时候结婚的？疑问。对于辣妈，我真得战战兢兢忐忐忑忑，辣容易，当妈好难。极强的动手能力与时间分配能力，杂物整理能力，好耐力等等的考验。说实话我前所未有的觉得自己需要莫大的支持和身边的帮助。可。哎。。。http://t.cn/R2Wx9I3 | 对于结婚我可能真低调到了,生完娃仍然有许多人发来☞你什么时候结婚的?疑问。对于辣妈,我真得战战兢兢忐忐忑忑,辣容易,当妈好难。极强的动手能力与时间分配能力,杂物整理能力,好耐力等等的考验。说实话我前所未有的觉得自己需要莫大的支持和身边的帮助。可。哎。。。 |



## 3、模型训练

使用Bert在中文数据集上的预训练后的模型，进行fine-tune。

>  参考仓库：https://github.com/huggingface/transformers
>
>  Transformers 提供了数以千计的预训练模型，支持 100 多种语言的文本分类、信息抽取、问答、摘要、翻译、文本生成。它的宗旨让最先进的 NLP 技术人人易用。



训练参数如下：

| 参数           | 设置               |
| -------------- | ------------------ |
| 显卡           | GTX1080Ti with 12G |
| 模型名称       | bert-base-chinese  |
| batch_size     | 64                 |
| 混合精度训练   | amp  True          |
| 学习率         | 1e-5               |
| Warm up        | 1 epoch            |
| 训练轮数       | 30                 |
| 优化器         | AdamW              |
| 序列的最大长度 | 140                |
|                |                    |



### 3.1、train

Tensorboard的监控，包括：

- accuracy
- f1 score(macro)
- train loss
- lr：在第一个epoch使用warm up 

![image-20211228002941558](/Users/wjj/Library/Application Support/typora-user-images/image-20211228002941558.png)

> 随着训练的进行，train的acc、f1_score都是稳步提升，loss稳步下降，收敛。（横坐标为迭代步数）

### 3.2、validation

Tensorboard的监控，包括：

- accuracy
- f1 score(macro)
- val loss

![image-20211228003223447](/Users/wjj/Library/Application Support/typora-user-images/image-20211228003223447.png)

> 每个epoch训练完成都在验证集上进行验证，acc、f1_score、loss和训练集上的趋势基本一致。（横坐标为训练轮数）

## 4、模型测试

在测试集上评测模型：

- ==accuracy：0.772==
- ==f1 score(macro)：0.739==

![image-20211228233712843](/Users/wjj/Library/Application Support/typora-user-images/image-20211228233712843.png)

> 以上是本次比赛最后的排名。在仅仅使用单个BERT模型，不使用：
>
> - 疫情数据集
> - RoBerta等改进的中文预训练模型
> - 多模型融合
>
> 也取得了有一定竞争力的效果。

# 三、总结

BERT采用了Transformer Encoder的模型来作为语言模型， 完全抛弃了RNN/CNN等结构，而完全采用Attention机制来进行input-output之间关系的计算，在无监督预料上进行超大规模的预训练，能够有效提升下游语言任务的准确性。本文在BERT的文本分类任务上进行实践，数据集来源于**SMP2020-EWECT微博情绪分类**的通用数据集，在没有使用任何trick的情况下，在中文预训练的BERT上进行微调，单模型取得了==77.2%==的准确率。



# 四、相关代码

`git clone https://github.com/BrownSweater/BERT_SMP2020-EWECT.git`

## 1、附件

> ├── clean_data.py     ##  数据预处理代码
> ├── **data/**                   ##  数据集
> │   ├── **clean/**          ##  原始数据集
> │   └── **raw/**           ##  清洗后数据集
> ├── events.out.tfevents.1640466123.wjj.9864.0  ## tensorboard结果文件
> ├── finetune.py       ##  训练代码
> ├── inference.py    ## 推理代码
> ├── requirements.txt   ## 依赖包
> ├── test.py              ##  测试代码
> ├── test.sh              ##  测试脚本
> └── train.sh           ##  训练脚本

> 模型文件：https://pan.baidu.com/s/1e6Cs9STYmcdCWv6j3bjKQg    2ddr



## 2、训练

- 安装依赖包：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package`
- 开始训练：`sh train.sh`，在workspace/wb/目录下会保存相关训练的文件



## 3、测试

- 下载提供的模型文件 or 自己训练得到的模型，放在workspace/wb/目录下
- 开始测试：`sh test.sh`

![image-20211228233712843](/Users/wjj/Library/Application Support/typora-user-images/image-20211228233712843.png)

> 以上是本次比赛最后的排名。在仅仅使用BERT模型，而不是使用



## 4、推理

- 下载提供的模型文件 or 自己训练得到的模型，放在workspace/wb/目录下

- 执行代码：`python inference.py --input '你是个什么东西，垃圾' --device cpu`

  > 执行结果：##################### result: angry #####################



## 5、Web部署

- 下载提供的模型文件 or 自己训练得到的模型，放在workspace/wb/目录下
- 执行代码：`python3  web_demo.py  --device cpu`