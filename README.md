论文上，心里有个底，这个路是对的

用大模型，应该蒸馏出来，知识迁移，<DINO SAM 能用> 看看开源了没

DINO 论文层面理解理解

跟一跟，想法上梳理梳理

扩散模型，做的是盲超分，blind multi degration

大模型做图像复原 ？

公众号，想法的新颖度，期刊把一个方法做全 

1. 代码跑完
2. 搜集论文，cover全
3. 根据代码，根据论文，梳理idea



代做事项
1. PSNR 计算
2. 数据集增添 Set14
3. Urban 100

实验结果记录
不带 Overlap
Model：'/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240204_022316/295000_model.pth'
            With Yadapt   Without Yadapt  (Average PSNR)
1. Set 5      32.03           31.89
2. Set 14     29.15           29.19*
3. Urban 100  26.35           26.62*
4. Manga 109  29.33           29.91*
5. BSDS100    28.38           29.00*

不带 Overlap
Model：'/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240204_022316/100000_model.pth'
            With Yadapt   Without Yadapt  (Average PSNR)
1. Set 5      30.80*          29.62
2. Set 14     28.45*          27.51
3. Urban 100   
4. Manga 109
5. BSDS100    27.81*          27.35


带 Overlap
Model：'/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240204_022316/295000_model.pth'
            With Yadapt   Without Yadapt  (Average PSNR)  SwinIR
1. Set 5      34.09           34.44*                       38.35
2. Set 14     30.28           30.42*
3. Urban 100  27.49           27.63*
4. Manga 109  33.17           33.50*
5. BSDS100    29.61           29.73*        

带 Overlap
Model：'/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240204_022316/100000_model.pth'
            With Yadapt   Without Yadapt  (Average PSNR)
1. Set 5                      32.45
2. Set 14                     29.09
3. Urban 100                  26.23
4. Manga 109                  30.71
5. BSDS100                    28.62

# 是不是完全对齐？

# 1.训练的不够
# 2.SwinIR 加载预训练模型 strict=False, 得有 20w step, 检查是不是加载上了
# 3.学习率的下降到 2e-5 到 2.5e-5
# 4.数据集，Flick2k，DIV2k，找一个 AIM2019 特殊退化，拿 bicubic 训练，用 AIM2019 测试；或者 AIM2019 
# 5.先用 SwinIR 测 AIM2019，然后用 SwinIR+Yadapt 测 AIM2019 | 或者我们的方法拿 AIM2019 重新训练
# 6.等于0和不等于0这样测是否有意义？

带 Overlap
Model：'/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240204_022316/100000_model.pth'
            With Yadapt   Without Yadapt  (Average PSNR)
1. Set 5        34.19              
2. Set 14                     
3. Urban 100                  
4. Manga 109    32.75                 
5. BSDS100                    

带 Overlap
Yadapt 置于 0 训练
Model: 'experiments/SwinIR_20240317_192139/60000_model.pth'

1. Set 5        36.45 // 38.11(预训练模型 predict_adapter.py)
1. Set 5        38.32(predict.py)



# ===============
# 带 Overlap， Yadapt 置于 0 训练
Config path: /home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml
Resume from checkpoint from experiments/SwinIR_20240319_012910/245000_model.pth
Avg PSNR: 36.48
Config path: /home/mayanze/PycharmProjects/SwinTF/config/Set14test.yaml
Resume from checkpoint from experiments/SwinIR_20240319_012910/245000_model.pth
Avg PSNR: 32.42
Config path: /home/mayanze/PycharmProjects/SwinTF/config/BSDS100.yaml
Resume from checkpoint from experiments/SwinIR_20240319_012910/245000_model.pth
Avg PSNR: 31.13
Config path: /home/mayanze/PycharmProjects/SwinTF/config/manga109test.yaml
Resume from checkpoint from experiments/SwinIR_20240319_012910/245000_model.pth
Avg PSNR: 35.25
Config path: /home/mayanze/PycharmProjects/SwinTF/config/urban100test.yaml
Resume from checkpoint from experiments/SwinIR_20240319_012910/245000_model.pth
Avg PSNR: 29.19




---

| Dataset  | Config Path                                                  | Average PSNR |
| -------- | ------------------------------------------------------------ | ------------ |
| Set 5    | `/home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml` | 36.44        |
| Set14    | `/home/mayanze/PycharmProjects/SwinTF/config/Set14test.yaml` | 32.36        |
| BSDS100  | `/home/mayanze/PycharmProjects/SwinTF/config/BSDS100.yaml`   | 31.09        |
| Manga109 | `/home/mayanze/PycharmProjects/SwinTF/config/manga109test.yaml` | 35.18        |
| Urban100 | `/home/mayanze/PycharmProjects/SwinTF/config/urban100test.yaml` | 29.11        |





yadapt 不为 0 7 万步

| Dataset       | Average PSNR |
| ------------- | ------------ |
| Example Set 5 | 36.44        |
| Set14         | 32.36        |
| BSDS100       | 31.09        |
| Manga109      | 35.18        |
| Urban100      | 29.11        |



逻辑上这里应该是相同万步比较？

yadapt不为0，2万步

Config path: /home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/20000_model.pth
Avg PSNR: 36.03
Config path: /home/mayanze/PycharmProjects/SwinTF/config/Set14test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/20000_model.pth
Avg PSNR: 32.17
Config path: /home/mayanze/PycharmProjects/SwinTF/config/BSDS100.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/20000_model.pth
Avg PSNR: 30.91
Config path: /home/mayanze/PycharmProjects/SwinTF/config/manga109test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/20000_model.pth
Avg PSNR: 34.51
Config path: /home/mayanze/PycharmProjects/SwinTF/config/urban100test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/20000_model.pth
Avg PSNR: 28.83

SwinIRAdapter 2万步
| Dataset   | Avg PSNR |
|-----------|----------|
| Set5      | 36.03    |
| Set14     | 32.17    |
| BSDS100   | 30.91    |
| Manga109  | 34.51    |
| Urban100  | 28.83    |

SwinIR Adapter 7万步

| Dataset       | Average PSNR |
| ------------- | ------------ |
| Example Set 5 | 36.46        |
| Set14         | 32.37        |
| BSDS100       | 31.10        |
| Manga109      | 35.19        |
| Urban100      | 29.11        |

SwinIR Adapter 16万步

| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 36.42    |
| Set14    | 32.37    |
| BSDS100  | 31.09    |
| Manga109 | 35.18    |
| Urban100 | 29.14    |


---


SwinIR，2万步

| Dataset       | Average PSNR |
| ------------- | ------------ |
| Example Set 5 | 37.58        |
| Set14         | 33.47        |
| BSDS100       | 31.95        |
| Manga109      | 37.90        |
| Urban100      | 31.76        |


SwinIR 6万步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 37.78    |
| Set14    | 33.70    |
| BSDS100  | 32.05    |
| Manga109 | 38.41    |
| Urban100 | 32.43    |



SwinIR 23万步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 37.89    |
| Set14    | 33.78    |
| BSDS100  | 32.15    |
| Manga109 | 38.65    |
| Urban100 | 32.60    |



SwinIR 29万步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 38.02    |
| Set14    | 33.98    |
| BSDS100  | 32.21    |
| Manga109 | 39.09    |
| Urban100 | 33.06    |

30w步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 38.03    |
| Set14    | 33.97    |
| BSDS100  | 32.21    |
| Manga109 | 39.22    |
| Urban100 | 33.06    |

33w 步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 38.01    |
| Set14    | 34.00    |
| BSDS100  | 32.20    |
| Manga109 | 39.05    |
| Urban100 | 33.06    |


SwinIR 34万步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 37.99    |
| Set14    | 33.98    |
| BSDS100  | 32.19    |
| Manga109 | 39.18    |
| Urban100 | 33.04    |

SwinIR 37万步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 38.04    |
| Set14    | 34.02    |
| BSDS100  | 32.21    |
| Manga109 | 39.13    |
| Urban100 | 33.09    |

SwinIR 40万步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 38.08    |
| Set14    | 33.99    |
| BSDS100  | 32.23    |
| Manga109 | 39.12    |
| Urban100 | 33.10    |



SwinIR 44万步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 38.06    |
| Set14    | 33.98    |
| BSDS100  | 32.21    |
| Manga109 | 39.17    |
| Urban100 | 33.09    |



---
# 新的网络

SwinIR Adapter 5万步
| Dataset  | Avg PSNR |
|----------|----------|
| Set5     | 36.47    |
| Set14    | 32.39    |
| BSDS100  | 31.10    |
| Manga109 | 35.20    |
| Urban100 | 29.16    |



---
SAM 归一化 SwinIR Adapter 5万步
| Dataset  | Avg PSNR | Yadapt       | Adapter       |
|----------|----------|--------------|---------------|
| Set5     | 38.09    | True         | SwinIRAdapter |
| Set14    | 34.09    | True         | SwinIRAdapter |
| BSDS100  | 32.27    | True         | SwinIRAdapter |
| Manga109 | 39.22    | True         | SwinIRAdapter |
| Urban100 | 33.06    | True         | SwinIRAdapter |


SAM 归一化 SwinIR Adapter 7万步
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/exampleSet5.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/70000_model.pth
Avg PSNR: 38.02
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/Set14test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/70000_model.pth
Avg PSNR: 33.95
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/BSDS100.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/70000_model.pth
Avg PSNR: 32.26
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/manga109test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/70000_model.pth
Avg PSNR: 39.14
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/70000_model.pth
Avg PSNR: 33.06


SAM 归一化 SwinIR Adapter 8万步
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/exampleSet5.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/80000_model.pth
Avg PSNR: 38.13
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/Set14test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/80000_model.pth
Avg PSNR: 33.97
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/BSDS100.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/80000_model.pth
Avg PSNR: 32.28
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/manga109test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/80000_model.pth
Avg PSNR: 39.11
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/80000_model.pth
Avg PSNR: 33.07






SAM 归一化 SwinIR Adapter 9万步
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/exampleSet5.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/90000_model.pth
Avg PSNR: 38.11
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/Set14test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/90000_model.pth
Avg PSNR: 34.00
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/BSDS100.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/90000_model.pth
Avg PSNR: 32.27
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/manga109test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/90000_model.pth
Avg PSNR: 39.21
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml
Resume from checkpoint from /home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240404_023741/90000_model.pth
Avg PSNR: 33.12

---
新归一化 SAMAdapter 14万步
(base) mayanze@trainer1:~/PycharmProjects/SwinTF$ bash run_test_adapter.sh 
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/exampleSet5.yaml
Resume from checkpoint from experiments/SwinIR_20240410_011847/140000_model.pth
Avg PSNR: 37.96
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/Set14test.yaml
Resume from checkpoint from experiments/SwinIR_20240410_011847/140000_model.pth
Avg PSNR: 33.85
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/BSDS100.yaml
Resume from checkpoint from experiments/SwinIR_20240410_011847/140000_model.pth
Avg PSNR: 32.12
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/manga109test.yaml
Resume from checkpoint from experiments/SwinIR_20240410_011847/140000_model.pth
Avg PSNR: 38.81
Yadapt is True SwinIRAdapter
Config path: /home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml
Resume from checkpoint from experiments/SwinIR_20240410_011847/140000_model.pth
Avg PSNR: 32.97



---
Manga109 加噪声
| Noise Type  | SwinIR Avg PSNR (Manga109) | SwinIRAdapter Avg PSNR (Manga109) |
|-------------|----------------------------|-----------------------------------|
| blur_aniso  | 19.15                      | 19.15 ≈                            |
| blur_iso    | 25.87                      | 25.74 ↓                            |
| degrade     | 16.55                      | 16.63 ↑                            |
| jpeg        | 30.11                      | 30.10 ↓                            |
| noise       | 22.02                      | 22.43 ↑                            |

Urban100 加噪声
| Noise Type | SwinIR Avg PSNR (Urban100) | SwinIRAdapter Avg PSNR (Urban100) |
|------------|-----------------------------|-----------------------------------|
| blur_aniso | 18.90                       | 18.90 ≈                            |
| blur_iso   | 23.69                       | 23.61 ↓                            |
| degrade    | 16.08                       | 16.20 ↑                            |
| jpeg       | 26.71                       | 26.62 ↓                            |
| noise      | 20.82                       | 21.25 ↑                            |

--> 做定量分析，对 noise

Manga109 针对噪声研究
| Noise Type           | SwinIR Avg PSNR (Manga109) | SwinIRAdapter Avg PSNR (Manga109) |
|----------------------|----------------------------|-----------------------------------|
| noise_01_general     | 7.07                       | 9.59                              |
| noise_01_channel     | 38.54                      | 37.66                             |
| noise_01_gray        | 38.12                      | 37.34                             |
| noise_05_general     | 7.18                       | 9.67                              |
| noise_05_channel     | 33.76                      | 33.24                             |
| noise_05_gray        | 31.23                      | 31.05                             |
| noise_10_general     | 7.35                       | 9.78                              |
| noise_10_channel     | 29.11                      | 28.76                             |
| noise_10_gray        | 25.91                      | 25.85                             |
| noise_15_general     | 7.23                       | 9.67                              |
| noise_15_channel     | 25.96                      | 25.72                             |
| noise_15_gray        | 22.55                      | 22.54                             |

** 这里后来发现算 general noise 的跟 sigma 没关系，所以这里其实是三次随机数出来的 general noise **
--> 发现 general noise 的优化效果好，但是 PSNR 值太低了，看看生成出来的图像
--> 调整了一下 noise 的强度，重新生成一下

Manga109 针对 general 噪声研究
| Noise Type               | SwinIR Avg PSNR (Manga109) | SwinIRAdapter Avg PSNR (Manga109) |
|--------------------------|----------------------------|-----------------------------------|
| noise_sigma_0_010_general | 34.74                     | 34.20                             | *不应该
| noise_sigma_0_050_general | 24.52                     | 24.32                             |
| noise_sigma_0_100_general | 18.86                     | 19.00                             |
| noise_sigma_0_150_general | 16.40                     | 16.70                             |

--> 那 manga109 的部分算是算完了，看看 urban100 的结果

| Noise Type                | SwinIR Avg PSNR (Urban100) | SwinIRAdapter Avg PSNR (Urban100) |
|---------------------------|----------------------------|-----------------------------------|
| noise_01_channel          | 32.96                      | 32.40                             |
| noise_01_gray             | 32.76                      | 32.23                             |
| noise_05_channel          | 30.20                      | 29.83                             |
| noise_05_gray             | 28.73                      | 28.49                             |
| noise_10_channel          | 27.08                      | 26.76                             |
| noise_10_gray             | 24.57                      | 24.45                             |
| noise_15_channel          | 24.58                      | 24.35                             |
| noise_15_gray             | 21.55                      | 21.49                             |
| noise_sigma_0_010_general | 30.95                      | 30.56                             |
| noise_sigma_0_050_general | 22.51                      | 22.40                             |
| noise_sigma_0_100_general | 17.61                      | 17.83                             |
| noise_sigma_0_150_general | 15.56                      | 16.02                             |

Adapter 和 SwinIR 在 Manga109 和 Urban100 再测一下；给的图像都是 matlab 生成的
Set5 和 Set14 也用 KAIR 的测试一把

What we can learn, what we can benifit 

1) SwiIR urban100test 33.41， manga109 39.54，Set14 34.11， Set5 38.32， BSDS100 32.46
2) SwinIR 论文 


| Dataset | SwinIR 我们 | SwinIR 论文 | SwinIR 43w Step | SwinIR 23w Step | SwinIR 44w Step |
|----------|----------|----------|----------|----------|---------|
| Urban100 | 33.41    | 33.40    | 32.95    | 32.88    |38.06    |
| Manga109 | 39.54    | 39.60    | 39.01    | 39.01    |33.98    |
| Set14    | 34.11    | 34.14    | 33.85    | 33.88    |32.21    |
| Set5     | 38.32    | 38.35    | 38.01    | 38.05    |39.17    |
| BSDS100  | 32.46    | 32.44    | 32.25    | 32.26    |33.09    |

