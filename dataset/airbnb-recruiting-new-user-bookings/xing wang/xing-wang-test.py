import pandas as pd

def load_train_data():
    # load data
    train_path = '../input/train_users_2.csv'
    train_users = pd.read_csv(train_path)
    return train_users
    
def load_test_data():
    test_path = '../input/test_users.csv'
    test_users = pd.read_csv(test_path)
    return test_users

if __name__ == "__main__":
    data = load_train_data()
    print(data.shape)
    test_data = load_test_data()
    print(test_data.shape)