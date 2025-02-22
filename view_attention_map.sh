config_path='/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/noise/noise_sigma_0_150_general.yaml'

mode='adapter'

gpu_id='0'

# swinir_mode='pixelshuffle'
# model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20250106143929/500000_model.pth'

swinir_mode='strongnorm'
# model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20250115064406/500000_model.pth'
model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20250124161945/500000_model.pth'

# swinir_mode='rstbadapt'
# model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20250119150031/500000_model.pth'

# swinir_mode='newfeature'
# model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20250117044926/500000_model.pth'

python main_test_swinir.py --config $config_path --model $model_path --gpu $gpu_id --swinir_mode $swinir_mode