# config_dir="config/manga109_test/noise"
# config_dir='dataset/testsets/Set14/LRbicx2'
config_dir='config/urban100_test/noise'
# config_dir='dataset/testsets/urban100_lrx2'
# config_dir='config/manga109_test/noise'


# swinir_mode='pixelshuffle'
# model_path='experiments/SwinIR_20250106143929/500000_model.pth'

swinir_mode='strongnorm'
# model_path='experiments/SwinIR_20250115064406/500000_model.pth'
# model_path='experiments/SwinIR_20250124161945/500000_model.pth'

# swinir_mode='rstbadapt'
# model_path='experiments/SwinIR_20250119150031/500000_model.pth'

# swinir_mode='newfeature'
# model_path='experiments/SwinIR_20250117044926/500000_model.pth'

swinir_mode='psnorm'
model_path='experiments/SwinIR_20250218160256/500000_model.pth'


# Set up a mode for different commands
mode="adapter"
# mode="swinir"


swinir_path="001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth"
yaml_files=($(find "$config_dir" -type f -name "*.yaml" | sort))
# Assign 8 GPUs, split into four groups
gpu_ids=('4' '5' '6' '7')

txt_file="$(basename $(dirname $config_dir))_$(basename $config_dir)_noise_${swinir_mode}.txt"

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
