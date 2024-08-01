config_dir="/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/noise"
yaml_files=$(find "$config_dir" -type f -name "*.yaml")
txt_file="urban100_test_noise_adapter.txt"
model_path="/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240503_113731/430000_model.pth"
gpu_id='1,3'
yadapt='True'
for yaml_file in $yaml_files
do
    python predict_adapter.py "$yaml_file" "$model_path" "$gpu_id" "$yadapt">> $txt_file 2>&1
done