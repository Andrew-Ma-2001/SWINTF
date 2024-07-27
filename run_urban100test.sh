model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240503_113731/430000_model.pth'
gpu_id='1,3'
yadapt='True'
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/urban100_test/blur_aniso.yaml $model_path $gpu_id $yadapt
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/urban100_test/blur_iso.yaml $model_path $gpu_id $yadapt
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/urban100_test/degrade.yaml $model_path $gpu_id $yadapt
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/urban100_test/jpeg.yaml $model_path $gpu_id $yadapt
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/urban100_test/noise.yaml $model_path $gpu_id $yadapt
