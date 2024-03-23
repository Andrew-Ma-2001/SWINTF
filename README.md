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

