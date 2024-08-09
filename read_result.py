import os
import re
import pandas as pd
import numpy as np

def extract_epoch_from_filename(directory_path):
    pattern = re.compile(r'epoch=(\d+)-val-acc=\d+\.\d+\.ckpt')
    for filename in os.listdir(os.path.join(directory_path, 'checkpoints')):
        match = pattern.match(filename)
        if match:
            return match.group(1)
    
    return None

def get_val_acc_for_epoch(directory_path, epoch):
    epoch = int(epoch) # turn to int to remove 0 in case 07, 08, etc
    val_log_path = os.path.join(directory_path, 'val.log')
    with open(val_log_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if parts[0] == str(epoch):
                return parts[1]
    return None


def print_aligned(epoch_value, val_acc, peak_mem, mean_mem, std_mem, total_mem):
    print(f"\n{'Epoch':<10}{'Acc':<20}{'Peak mem':<20}{'Mean and deviation':<30}{'Total mem':<12}")
    # In các giá trị với định dạng đúng
    mean_std_str = f"{mean_mem:.4f} ± {std_mem:.4f}"  # Định dạng chuỗi mean_mem và std_mem
    print("hahahaha: ", epoch_value, val_acc, peak_mem, mean_std_str, total_mem)

    print(f"{epoch_value:<10}{val_acc:<20}{peak_mem:<20}{mean_std_str:<30}{total_mem:<12}")


def extract_config(directory_path):
    config_path = os.path.join(directory_path, 'config.yaml')
    # Initialize variables
    seed = None
    num_of_finetune = None
    model = None
    dataset = None
    with_SVD_with_var_compression = None
    with_HOSVD_with_var_compression = None
    with_grad_filter = None
    base = None
    SVD_var = None
    filt_radius = None
    exp_name = None
    
    with open(config_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if parts[0] == 'seed_everything:':
                seed = parts[1]
            elif parts[0] == 'num_of_finetune:':
                num_of_finetune = parts[1]
            elif parts[0] == 'backbone:':
                model = parts[1]
            elif parts[0] == 'name:':
                dataset = parts[1]
            elif parts[0] == 'with_SVD_with_var_compression:':
                with_SVD_with_var_compression = parts[1]
            elif parts[0] == 'with_HOSVD_with_var_compression:':
                with_HOSVD_with_var_compression = parts[1]
            elif parts[0] == 'with_grad_filter:':
                with_grad_filter = parts[1]
            elif parts[0] == 'SVD_var:':
                SVD_var = parts[1]
            elif parts[0] == 'filt_radius:':
                filt_radius = parts[1]
            elif parts[0] == 'exp_name:':
                exp_name = parts[1]
            
        if with_SVD_with_var_compression == 'false' and with_HOSVD_with_var_compression == 'false' and with_grad_filter == 'false':
            base = 'true'
        else: base = 'false'
    return seed, num_of_finetune, model, dataset, with_SVD_with_var_compression, with_HOSVD_with_var_compression, with_grad_filter, base, SVD_var, filt_radius, exp_name

def calculate_mean_and_deviation(data):
    """
    Calculate mean and standard deviation of the given data.

    Parameters:
        data (list): A list of numeric values.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the data.

    Example:
        If data = [1, 2, 3, 4, 5], the calculated mean and standard deviation will be:
        (3.0, 1.58)
    """
    # Calculate the mean
    mean = np.mean(data)

    squared_diff = sum((x - mean) ** 2 for x in data)
    # Calculate the variance
    variance = squared_diff / (len(data) - 1)

    # Calculate the standard deviation
    deviation = np.sqrt(variance)

    # Round the results to 2 decimal places
    # mean = round(mean, 2)
    # deviation = round(deviation, 2)

    return mean, deviation

def get_mem(directory_path):
    mem_log_path = os.path.join(directory_path, 'activation_memory_Byte.log')
    with open(mem_log_path, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        mem=[]
        for line in lines:
            parts = line.strip().split()
            mem.append(float(parts[1]))
    mean, deviation = calculate_mean_and_deviation(mem)
    return max(mem), sum(mem), mean, deviation

# def find_val_log_files(root_directory):
#     # Duyệt theo thứ tự, này phải specific hơn, đường dẫn gốc phải là định dạng 'runs/tên model/tên data'
#     dirs = sorted(os.listdir(root_directory)) # Thu được 1 dãy link đã sắp xếp
#     for dir in dirs: # Duyệt từng phần tử trong mỗi link
#         dir2_link = os.path.join(root_directory, dir) # Thu được từng link tương ứng
#         dir2s = sorted(os.listdir(dir2_link))  # Thu được 1 dãy các phần tử đã sắp xếp
#         for dir2 in dir2s:
#             dir3_link = os.path.join(dir2_link, dir2)
#             dir3s = sorted(os.listdir(dir3_link))  # Thu được 1 dãy các phần tử đã sắp xếp
#             for dir3 in dir3s:
#                 dir4_link = os.path.join(dir3_link, dir3)
#                 dir4s = sorted(os.listdir(dir4_link))  # Thu được 1 dãy các phần tử đã sắp xếp
#                 for dir4 in dir4s:
#                     dir5_link = os.path.join(dir4_link, dir4)

#                     data = dir5_link.split('/')
#                     print(f"==============Experiment Name: {data[5]}/{data[6]}==============")
#                     print(f"Model: {data[1]}, Dataset: {data[2]}, Method: {data[3]}_{data[4]}, ")
#                     epoch_value = extract_epoch_from_filename(dir5_link)
#                     val_acc = get_val_acc_for_epoch(dir5_link, epoch_value)
#                     peak_mem, total_mem = get_mem(dir5_link)

#                     if epoch_value:
#                         extract_config(dir5_link)
#                         print_aligned(epoch_value, val_acc, peak_mem, total_mem)
#                     else:
#                         print('No matching file found.')
#                     print("====================================================================================================")



    # # Duyệt random        
    # for dirpath, dirnames, filenames in os.walk(root_directory):
    #     for filename in filenames:
    #         if filename == "val.log":
    #             data = dirpath.split('/')
    #             print(f"==============Experiment Name: {data[5]}/{data[6]}==============")
    #             print(f"Model: {data[1]}, Dataset: {data[2]}, Method: {data[3]}_{data[4]}, ")
    #             epoch_value = extract_epoch_from_filename(dirpath)
    #             val_acc = get_val_acc_for_epoch(dirpath, epoch_value)
    #             peak_mem, total_mem = get_mem(dirpath)

    #             if epoch_value:
    #                 extract_config(dirpath)
    #                 print_aligned(epoch_value, val_acc, peak_mem, total_mem)
    #             else:
    #                 print('No matching file found.')
    #             print("====================================================================================================")

def find_val_log_files(root_directory):
    results = []
    def process_directory(current_directory):
        # Duyệt qua tất cả các mục trong thư mục hiện tại
        for entry in sorted(os.listdir(current_directory)):
            entry_path = os.path.join(current_directory, entry)
            if entry == 'val.log':
                # Nếu gặp file 'val.log', thực hiện các hành động cần thiết
                data = current_directory.split(os.sep)
                seed, num_of_finetune, model, dataset, with_SVD_with_var_compression, \
                with_HOSVD_with_var_compression, with_grad_filter, base, SVD_var, \
                    filt_radius, exp_name = extract_config(current_directory)

                print(f"==============Experiment Name: {exp_name}==============")
                epoch_value = extract_epoch_from_filename(current_directory)
                val_acc = get_val_acc_for_epoch(current_directory, epoch_value)

                if epoch_value:
                    print(f"{'Seed:':<20}{seed:<20}\n{'num_of_finetune:':<20}{num_of_finetune:<20}")
                    print(f"{'Model:':<20}{model:<20}")
                    print(f"{'Dataset:':<20}{dataset:<20}")
                    if with_HOSVD_with_var_compression == 'true' or with_SVD_with_var_compression == 'true':
                        peak_mem, total_mem, mean_mem, std_mem = get_mem(current_directory)
                    else:
                        peak_mem, total_mem, mean_mem, std_mem = 0, 0, 0, 0

                    if with_HOSVD_with_var_compression == 'true':
                        print(f"{'Method:':<20}HOSVD_{SVD_var:<20}")
                        method = f"HOSVD_{SVD_var}"
                    elif with_SVD_with_var_compression == 'true':
                        print(f"{'Method:':<20}SVD_{SVD_var:<20}")
                        method = f"SVD_{SVD_var}"
                    elif with_grad_filter == 'true':
                        print(f"{'Method:':<20}GradientFilter_{filt_radius:<20}")
                        method = f"GradientFilter_{filt_radius}"
                    elif base == 'true':
                        print(f"{'Method:':<20}Vanilla BP")
                        method = "Vanilla BP"

                    print_aligned(epoch_value, val_acc, peak_mem, mean_mem, std_mem, total_mem)
                    divisor = 1 #1024*1024
                    peak_mem /= divisor
                    mean_mem /= divisor
                    std_mem /= divisor
                    total_mem /= divisor
                    results.append({
                        'Dataset': dataset,
                        'Method': method,
                        'Model': model,
                        'num_of_finetune': num_of_finetune,
                        'seed': seed,
                        'epoch_value': epoch_value,
                        'val_acc': val_acc,
                        'peak_mem': peak_mem,
                        'mean_mem': mean_mem,
                        'std_mem': std_mem,
                        'mean_mem and std_mem': f"{mean_mem:.2f} ± {std_mem:.2f}",
                        'total_mem': total_mem
                    })
                else:
                    print('No matching file found.')
                print("====================================================================================================")
            elif os.path.isdir(entry_path):
                # Nếu là thư mục, gọi đệ quy để xử lý thư mục con
                process_directory(entry_path)

    # Bắt đầu từ thư mục gốc
    process_directory(root_directory)

    # Export results to Excel
    df = pd.DataFrame(results)
    df.to_excel('results.xlsx', index=False)
    print("Results have been exported to results.xlsx")


# Sử dụng hàm này
root_directory = "runs/mcunet/cifar10/HOSVD/var0.8/HOSVD_l33_var0.8_full_pretrain_imagenet_cifar10" # Link luôn phải là thư mục runs đứng đầu
val_log_files = find_val_log_files(root_directory)

