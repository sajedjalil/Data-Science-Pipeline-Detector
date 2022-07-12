import file_reader
import os


def get_dataset_path(dataset_folder_name):
    current_directory = os.getcwd()
    current_directory = current_directory.split(os.sep)[:-1]
    current_directory.append(dataset_folder_name)

    return os.sep.join(current_directory)


if __name__ == '__main__':
    dataset_path = get_dataset_path('dataset')
    reader = file_reader.FileReader(dataset_path)

    print(reader.read_csv())
