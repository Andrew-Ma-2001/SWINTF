# model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240204_022316/290000_model.pth'
model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240803080852/500000_model.pth'
gpu_id='4,5'
yadapt='True'
python /home/mayanze/PycharmProjects/SwinTF/predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/Set5.yaml $model_path $gpu_id $yadapt
python /home/mayanze/PycharmProjects/SwinTF/predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/Set14test.yaml $model_path $gpu_id $yadapt
python /home/mayanze/PycharmProjects/SwinTF/predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/BSDS100.yaml $model_path $gpu_id $yadapt
python /home/mayanze/PycharmProjects/SwinTF/predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/manga109test.yaml $model_path $gpu_id $yadapt
python /home/mayanze/PycharmProjects/SwinTF/predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml $model_path $gpu_id $yadapt