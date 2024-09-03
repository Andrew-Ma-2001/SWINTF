model_path='SwinIR_20240813062655'
step=365000

# Run each command in parallel and redirect output to separate log files
# python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X4/BSDS100.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 4 > results/BSDS100.log 2>&1 &

# python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X4/set5.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 5 > results/set5.log 2>&1 &

# python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X4/Set14test.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 5 > results/Set14test.log 2>&1 &

# python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X4/urban100test.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 6 > results/urban100test.log 2>&1 &

# python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X4/manga109test.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 7 > results/manga109test.log 2>&1 &


python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X2/BSDS100.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 4 > results/BSDS100.log 2>&1 &

python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X2/set5.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 5 > results/set5.log 2>&1 &

python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X2/Set14test.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 5 > results/Set14test.log 2>&1 &

python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X2/urban100test.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 6 > results/urban100test.log 2>&1 &

python main_test_swinir.py --config /home/mayanze/PycharmProjects/SwinTF/config/X2/manga109test.yaml --model /home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth --gpu 7 > results/manga109test.log 2>&1 &


# Wait for all background processes to finish
wait

# Combine all log files into a single result file
cat results/*.log > results/adapterx2.txt