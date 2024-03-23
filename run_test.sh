# model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240204_022316/290000_model.pth'
model_path='experiments/SwinIR_20240319_012910/245000_model.pth'
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/exampleSet5.yaml $model_path
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/Set14test.yaml $model_path
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/BSDS100.yaml $model_path
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/manga109test.yaml $model_path
python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/urban100test.yaml $model_path
