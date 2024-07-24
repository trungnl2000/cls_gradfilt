import os
import re

def extract_epoch_from_filename(directory_path):
    # Định dạng regex để tìm các file có định dạng tên mong muốn
    pattern = re.compile(r'epoch=(\d+)-val-acc=\d+\.\d+\.ckpt')
    
    for filename in os.listdir(os.path.join(directory_path, 'checkpoints')):
        match = pattern.match(filename)
        if match:
            return match.group(1)
    
    return None

def get_val_acc_for_epoch(directory_path, epoch):
    val_log_path = os.path.join(directory_path, 'val.log')
    with open(val_log_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if parts[0] == epoch:
                return parts[1]
    return None

def get_mem(directory_path):
    mem_log_path = os.path.join(directory_path, 'activation_memory_Byte.log')
    with open(mem_log_path, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        mem=[]
        for line in lines:
            parts = line.strip().split()
            mem.append(float(parts[1]))
    return max(mem), sum(mem)

def print_aligned(epoch_value, val_acc, peak_mem, total_mem):
    print(f"\n{'Epoch':<10}{'Acc':<20}{'Peak mem':<20}{'Total mem':<12}")
    print(f"{epoch_value:<10}{val_acc:<20}{peak_mem:<20}{total_mem:<12}")
    # print(f"{epoch_value}\t{val_acc}\t{peak_mem}\t{total_mem}")

# def extract_number_of_finetune_from_experiment_name(name):
#     match = re.search(r'_l(\d+)_', name)
#     if match:
#         return match.group(1)
#     else:
#         return None

def extract_config(directory_path):
    config_path = os.path.join(directory_path, 'config.yaml')
    with open(config_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if parts[0] == 'seed_everything:':
                # print("Seed: ", parts[1])
                seed = parts[1]
            elif parts[0] == 'num_of_finetune:':
                # print("num_of_finetune: ", parts[1])
                num_of_finetune = parts[1]
    return seed, num_of_finetune

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
    def process_directory(current_directory):
        # Duyệt qua tất cả các mục trong thư mục hiện tại
        for entry in sorted(os.listdir(current_directory)):
            entry_path = os.path.join(current_directory, entry)
            if entry == 'val.log':
                # Nếu gặp file 'val.log', thực hiện các hành động cần thiết
                data = current_directory.split(os.sep)
                print(f"==============Experiment Name: {data[5]}/{data[6]}==============")
                epoch_value = extract_epoch_from_filename(current_directory)
                val_acc = get_val_acc_for_epoch(current_directory, epoch_value)
                peak_mem, total_mem = get_mem(current_directory)

                if epoch_value:
                    seed, num_of_finetune = extract_config(current_directory)
                    print(f"{'Seed:':<20}{seed:<20}\n{'num_of_finetune:':<20}{num_of_finetune:<20}")
                    print(f"{'Model:':<20}{data[1]:<20}")
                    print(f"{'Dataset:':<20}{data[2]:<20}")
                    print(f"{'Method:':<20}{data[3]}_{data[4]:<20}")
                    print_aligned(epoch_value, val_acc, peak_mem, total_mem)
                else:
                    print('No matching file found.')
                print("====================================================================================================")
            elif os.path.isdir(entry_path):
                # Nếu là thư mục, gọi đệ quy để xử lý thư mục con
                process_directory(entry_path)

    # Bắt đầu từ thư mục gốc
    process_directory(root_directory)


# Sử dụng hàm này
root_directory = "runs/resnet34"
val_log_files = find_val_log_files(root_directory)

