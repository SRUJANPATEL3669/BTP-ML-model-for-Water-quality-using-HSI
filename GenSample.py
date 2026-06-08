from HyperTools import *
import argparse
import os

def main(args):  
    dataID = args.dataID
    X,Y,train_array,test_array = LoadHSI(dataID,args.train_samples)

    if dataID==1:
        save_pre_dir = './Data/PaviaU/'
    elif dataID==2:
        save_pre_dir = './Data/Salinas/'
    elif dataID==3:
        save_pre_dir = './Data/Indian_pines/'
    elif dataID==4:
        save_pre_dir = './Data/Houston13/'
    elif dataID==5:
        save_pre_dir = './Data/Houston18/'

    Y -= 1

    if os.path.exists(save_pre_dir)==False:
        os.makedirs(save_pre_dir)

    
    np.save(save_pre_dir+'X.npy',X)
    np.save(save_pre_dir+'Y.npy',Y)
    np.save(save_pre_dir+'train_array.npy',train_array)
    np.save(save_pre_dir+'test_array.npy',test_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dataID', type=int, default=0, help='0 for all datasets, 1 for PaviaU, 2 for Salinas')
    parser.add_argument('--train_samples', type=int, default=300)

    args = parser.parse_args()
    if args.dataID == 0:
        for data_id in [1, 2, 3, 4, 5]:
            print(f"=== Running for dataset ID {data_id} ===")
            args.dataID = data_id
            main(args)
    else:
        main(args)