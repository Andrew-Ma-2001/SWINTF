import os
# datasets = ['dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_0.0_quality_30_noise_0.0_noise_0.0_quality_30.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_0.0_quality_40_noise_0.0_noise_0.0_quality_40.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_0.0_quality_60_noise_0.0_noise_0.0_quality_60.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_0.0_quality_70_noise_0.0_noise_0.0_quality_70.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_0.0_quality_80_noise_0.0_noise_0.0_quality_80.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_0.0_quality_90_noise_0.0_noise_0.0_quality_90.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_0.0_quality_100_noise_0.0_noise_0.0_quality_100.yaml']

# datasets = ['dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_0.0_quality_0_noise_0.0_noise_0.0_quality_0.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_1.0_noise_0.0_quality_0_noise_1.0_noise_0.0_quality_0.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_2.0_noise_0.0_quality_0_noise_2.0_noise_0.0_quality_0.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_3.0_noise_0.0_quality_0_noise_3.0_noise_0.0_quality_0.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_4.0_noise_0.0_quality_0_noise_4.0_noise_0.0_quality_0.yaml', 
#             'dataset/testsets/Set14/LRbicx2/noise_sig_5.0_noise_0.0_quality_0_noise_5.0_noise_0.0_quality_0.yaml']

# datasets = [
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_1.0_quality_0_noise_0.0_noise_1.0_quality_0.yaml', 
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_2.0_quality_0_noise_0.0_noise_2.0_quality_0.yaml', 
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_3.0_quality_0_noise_0.0_noise_3.0_quality_0.yaml', 
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_4.0_quality_0_noise_0.0_noise_4.0_quality_0.yaml', 
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_5.0_quality_0_noise_0.0_noise_5.0_quality_0.yaml'
# ]

# datasets = [
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_10.0_quality_0_noise_0.0_noise_10.0_quality_0.yaml', 
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_20.0_quality_0_noise_0.0_noise_20.0_quality_0.yaml', 
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_30.0_quality_0_noise_0.0_noise_30.0_quality_0.yaml', 
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_40.0_quality_0_noise_0.0_noise_40.0_quality_0.yaml', 
#     'dataset/testsets/Set14/LRbicx2/noise_sig_0.0_noise_50.0_quality_0_noise_0.0_noise_50.0_quality_0.yaml'
# ]

datasets=[
    'dataset/testsets/urban100_lrx2/noise_sig_0.0_noise_10.0_quality_0_noise_0.0_noise_10.0_quality_0.yaml', 
    'dataset/testsets/urban100_lrx2/noise_sig_0.0_noise_20.0_quality_0_noise_0.0_noise_20.0_quality_0.yaml', 
    'dataset/testsets/urban100_lrx2/noise_sig_0.0_noise_30.0_quality_0_noise_0.0_noise_30.0_quality_0.yaml', 
    'dataset/testsets/urban100_lrx2/noise_sig_0.0_noise_40.0_quality_0_noise_0.0_noise_40.0_quality_0.yaml', 
    'dataset/testsets/urban100_lrx2/noise_sig_0.0_noise_50.0_quality_0_noise_0.0_noise_50.0_quality_0.yaml'
]


model_1 = 'experiments/SwinIR_20240803080852/500000_model.pth'
model_2 = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'

for dataset in datasets:
    print(dataset)
    os.system(f'python main_test_swinir.py --config {dataset} --model {model_1} --gpu 1,2')


model = '/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth'

for dataset in datasets:
    print(dataset)
    os.system(f'python main_test_swinir.py --config {dataset} --model {model} --test_swinir --gpu 1,2')