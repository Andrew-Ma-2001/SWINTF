config_dir="/home/mayanze/PycharmProjects/SwinTF/config/manga109_test"
yaml_files=$(find "$config_dir" -type f -name "*.yaml")
txt_file="manga109_test_noise_adapter.txt"
model_path="/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240503_113731/430000_model.pth"
gpu_id='1,3'
yadapt='True'
# for yaml_file in $yaml_files
# do
#     python predict_adapter.py "$yaml_file" "$model_path" "$gpu_id" "$yadapt">> $txt_file 2>&1
# done

python predict_adapter.py "/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise/noise_sigma_0_010_general.yaml" "$model_path" "$gpu_id" "$yadapt" >> $txt_file 2>&1
python predict_adapter.py "/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise/noise_sigma_0_050_general.yaml" "$model_path" "$gpu_id" "$yadapt" >> $txt_file 2>&1
python predict_adapter.py "/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise/noise_sigma_0_100_general.yaml" "$model_path" "$gpu_id" "$yadapt" >> $txt_file 2>&1
python predict_adapter.py "/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise/noise_sigma_0_150_general.yaml" "$model_path" "$gpu_id" "$yadapt" >> $txt_file 2>&1