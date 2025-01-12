import os
import argparse
import random
import csv

def split_dataset(input_dir, train_num, val_num, test_num):
    # Get all files in the directory
    print( os.listdir(input_dir))
    all_files = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    print(all_files)
    total_num = train_num + val_num + test_num
    if total_num > len(all_files):
        print(f"Error: The requested total number of files ({total_num}) exceeds the available files ({len(all_files)}).")
        return
    
    # Shuffle the file list randomly
    random.shuffle(all_files)
    
    # Split files into training, validation, and test sets
    train_files = all_files[:train_num]
    val_files = all_files[train_num:train_num+val_num]
    test_files = all_files[train_num+val_num:train_num+val_num+test_num]
    
    # Save filenames to CSV files
    with open('train_files.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for filename in train_files:
            writer.writerow([filename])
    
    with open('val_files.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for filename in val_files:
            writer.writerow([filename])
    
    with open('test_files.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for filename in test_files:
            writer.writerow([filename])
    
    print("Dataset splitting completed!")
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print(f"Test set: {len(test_files)} files")

def main():
    parser = argparse.ArgumentParser(description="Split dataset into training, validation, and test sets")
    parser.add_argument('--input_dir', type=str, default="/media/minheng/hdd_3/HCP_cc_0819/HCP_new/",
                        help='Path to the input directory containing files (default: %(default)s)')
    parser.add_argument('--train_num', default=864,type=int, help='Number of files for the training set')
    parser.add_argument('--val_num', default=78,type=int, help='Number of files for the validation set')
    parser.add_argument('--test_num', default=122, type=int, help='Number of files for the test set')
    
    args = parser.parse_args()
    
    split_dataset(args.input_dir, args.train_num, args.val_num, args.test_num)

if __name__ == '__main__':
    main()
