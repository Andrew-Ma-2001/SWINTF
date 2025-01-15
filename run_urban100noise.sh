# config_dir="/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise"
config_dir='/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/Set14/LRbicx2'
# config_dir='/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/noise'
# config_dir='/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_lrx2'
# config_dir='/home/mayanze/PycharmProjects/SwinTF/config/manga109_test/noise'

# txt_file="manga109_noise_adapter_previous_adapter.txt"
# txt_file="manga109_test_noise_swinir.txt"
# txt_file="set14_noise_adapter.txt"
# txt_file="set14_noise_swinir.txt"
# txt_file="urban100_noise_swinir.txt"
txt_file="set14_noise_pixelshuffle_network.txt"

# Mode 1
# model_path="/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20241130143441/500000_model.pth"
# Mode 2
# model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20241212143215/500000_model.pth'
# model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20241213152340/500000_model.pth'
model_path='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20250106143929/500000_model.pth'

# Set up a mode for different commands
mode="adapter"
# mode="swinir"

swinir_mode='pixelshuffle'
swinir_path="/home/mayanze/PycharmProjects/SwinTF/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth"
yaml_files=($(find "$config_dir" -type f -name "*.yaml" | sort))
# Assign 8 GPUs, split into four groups
gpu_ids=('4' '5' '6' '7')

# Function to run the command and save the last two lines of output
run_command() {
    local yaml_file=$1
    local gpu_id=$2
    local temp_file=$3
    if [ "$mode" == "adapter" ]; then
        local command="python main_test_swinir.py --config $yaml_file --model $model_path --gpu $gpu_id --swinir_mode $swinir_mode"
    else
        local command="python main_test_swinir.py --config $yaml_file --model $swinir_path --gpu $gpu_id --test_swinir"
    fi
    echo "Running: $command" >> $temp_file
    $command | tail -n 1 >> $temp_file
}

# Create an array to store temporary files for each GPU group
temp_files=()

# Loop through all YAML files
for ((i=0; i<${#yaml_files[@]}; i+=4))
do
    # Run up to four parallel processes
    for j in {0..3}
    do
        if [ $((i+j)) -lt ${#yaml_files[@]} ]; then
            # Create a temporary file for each GPU group
            temp_file="temp_output_gpu_${j}.txt"
            temp_files+=("$temp_file")
            run_command "${yaml_files[$((i+j))]}" "${gpu_ids[$j]}" "$temp_file" &
        fi
    done
    # Wait for the current batch of processes to finish
    wait
done

# Merge all temporary files into the main output file
for temp_file in "${temp_files[@]}"
do
    cat "$temp_file" >> $txt_file
    rm "$temp_file"
done
