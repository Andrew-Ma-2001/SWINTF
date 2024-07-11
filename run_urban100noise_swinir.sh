config_dir="/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/noise"
yaml_files=$(find "$config_dir" -type f -name "*.yaml")
txt_file="urban100_test_noise_swinir.txt"
gpu_id='1,3'

for yaml_file in $yaml_files
do
    python predict_swinir.py "$yaml_file" >> $txt_file 2>&1
done