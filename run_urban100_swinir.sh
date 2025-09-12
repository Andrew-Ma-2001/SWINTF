# config_dir="config/manga109_test/noise"
# config_dir='dataset/testsets/Set14/LRbicx2'
# config_dir='/home/mayanze/PycharmProjects/SwinTF/dataset/testsets/urban100_lrx2'
# config_dir='dataset/testsets/urban100_lrx2'
# config_dir='config/manga109_test/noise'
# config_dir='/home/mayanze/PycharmProjects/SwinTF/config/urban100_test/noise'

config_dir='dataset/testsets/urban100_lrx2'
# swinir_mode='pixelshuffle'
# model_path='experiments/SwinIR_20250106143929/500000_model.pth'

# swinir_mode='strongnorm'
# model_path='experiments/SwinIR_20250115064406/500000_model.pth'
# model_path='experiments/SwinIR_20250124161945/500000_model.pth'

# swinir_mode='rstbadapt'
# model_path='experiments/SwinIR_20250119150031/500000_model.pth'

# swinir_mode='newfeature'
# model_path='experiments/SwinIR_20250117044926/500000_model.pth'

# swinir_mode='psnorm'
# model_path='experiments/SwinIR_20250218160256/500000_model.pth'

# swinir_mode='sam'
# model_path='experiments/SwinIR_20250426080929/450000_model.pth'

# swinir_mode='sam'
# model_path='experiments/SwinIR_20250531035209/500000_model.pth'
# Set up a mode for different commands
swinir_mode='swinir'
# mode="adapter"
mode="swinir"

# Set a custom text for txt_file naming
custom_text="noise60-100增强"
swinir_path="001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth"
yaml_files=($(find "$config_dir" -type f -name "*.yaml" | sort))
# Assign 8 GPUs, split into four groups
gpu_ids=('0' '1' '2' '3')

# Function to filter yaml files with pattern noise_sigma_X_XXX_general where XXX >= 200
# filter_yaml_files() {
#     local files=("$@")
#     local filtered=()
#     for file in "${files[@]}"; do
#         # Extract the filename
#         fname=$(basename "$file")
#         # Use regex to match noise_sigma_X_XXX_general
#         if [[ $fname =~ noise_sigma_([0-9]+)_([0-9]+)_general\.yaml ]]; then
#             # Compose the number from the two groups (e.g., 0_250 -> 0.250)
#             major="${BASH_REMATCH[1]}"
#             minor="${BASH_REMATCH[2]}"
#             # Convert to integer for comparison (e.g., 0_250 -> 250)
#             number=$((10#$minor))
#             if (( number >= 200 )); then
#                 filtered+=("$file")
#             fi
#         fi
#     done
#     echo "${filtered[@]}"
# }

# Function to read config file paths from a txt file
read_yaml_files_from_txt() {
    local txt_path="$1"
    local yaml_list=()
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and lines starting with #
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        # If the path is not absolute, prepend config_dir
        if [[ "$line" = /* ]]; then
            yaml_list+=("$line")
        else
            yaml_list+=("$config_dir/$line")
        fi
    done < "$txt_path"
    echo "${yaml_list[@]}"
}

# Example usage:
yaml_files=($(read_yaml_files_from_txt "config_hard_name_list.txt"))



# # Filter yaml_files array
# yaml_files=($(filter_yaml_files "${yaml_files[@]}"))

txt_file="SAM2_${custom_text}_${swinir_mode}_$(basename $(dirname $config_dir))_noise.txt"

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
            temp_file="temp2_output_gpu_${j}.txt"
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
