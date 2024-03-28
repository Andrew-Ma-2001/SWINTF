# model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240204_022316/290000_model.pth'
model_path='/home/mayanze/PycharmProjects/SwinTF/190000_model_swiniradapter_nan.pth'
gpu_id='2,3'
yadapt='True'
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/exampleSet5.yaml $model_path $gpu_id $yadapt
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/Set14test.yaml $model_path $gpu_id $yadapt
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/BSDS100.yaml $model_path $gpu_id $yadapt
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/manga109test.yaml $model_path $gpu_id $yadapt
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml $model_path $gpu_id $yadapt
