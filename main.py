import argparse, os, torch
from functions import precision_and_recall, realism

G_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'generated_data')
R_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'real_data')

def parse_args():
    desc = "calcualte precision and recall OR realism"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--cal_type', type=str, default='precision_and_recall', choices=['precision_and_recall', 'realism'], help='The type of calcualtion')
    
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size', type=int, default=100)

    parser.add_argument('--generated_dir', default=G_DIRECTORY)
    parser.add_argument('--real_dir', default=R_DIRECTORY)



    

    print(parser.parse_args())
    return check_args(parser.parse_args())


def check_args(args):
    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)    
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

def main():
    # parse arguments
    args = parse_args()

    if args.cal_type == 'precision_and_recall':
        task = precision_and_recall(args)
    else:
        task = realism(args)

    task.run()

    print("Job finised!")

if __name__ == '__main__':
    main()   