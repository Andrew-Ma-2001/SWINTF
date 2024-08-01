config_dir="/home/mayanze/PycharmProjects/SwinTF/config/manga109_test"
yaml_files=$(find "$config_dir" -type f -name "*.yaml")
txt_file="manga109_test_noise_swinir.txt"
gpu_id='1,3'

# for yaml_file in $yaml_files
# do
#     python predict_swinir.py "$yaml_file" >> $txt_file 2>&1
# done


python predict_swinir.py "/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise/noise_sigma_0_010_general.yaml" >> $txt_file 2>&1
python predict_swinir.py "/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise/noise_sigma_0_050_general.yaml" >> $txt_file 2>&1
python predict_swinir.py "/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise/noise_sigma_0_100_general.yaml" >> $txt_file 2>&1
python predict_swinir.py "/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise/noise_sigma_0_150_general.yaml" >> $txt_file 2>&1