import os
import csv
import argparse
import pandas as pd
import time
from tqdm import tqdm

def trans_stmt_to_function(stmt_csv, out_csv):
    with open(stmt_csv, 'r') as fp:
        alls = fp.readlines()[1:]
        inputs = []
        outputs = []
        for line in alls:
            l = line.split('\t')
            inputs.append(l[1])
            outputs.append(l[2])
        df = pd.DataFrame(columns=['input', 'target'])
        df.loc[0] = [' ; '.join(inputs), ' '.join(outputs)]
        df.to_csv(out_csv, index=True, header=True, sep='\t')

def window(stmt_csv, out_csv):
    with open(stmt_csv, 'r') as fp:
        alls = fp.readlines()[1:]
        inputs = []
        outputs = []
        for line in alls:
            l = line.split('\t')
            inputs.extend(l[1].split(' ; '))
            outputs.append(l[2])
        new_inputs = []
        step = int(len(inputs) / len(outputs))
        if step < 1:
            return
        df = pd.DataFrame(columns=['input', 'target'])

        for idx, l in enumerate(outputs):
            tmp = ' ; '.join(inputs[idx*step:min(idx*step+step, len(inputs))])
            df.loc[idx] = [tmp, l]
        df.to_csv(out_csv, index=True, header=True, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='split.py')
    parser.add_argument('-o', '--save', type=str, default='./')
    parser.add_argument('-i', '--ori', type=str, default='./')
    parser.add_argument('-s', '--split', type=str, choices=['func', 'win'])

    args = parser.parse_args()

    files = os.listdir(args.ori)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    start = time.time
    for f in tqdm(files):
        in_path = os.path.join(args.ori, f)
        out_path = os.path.join(args.save, f)
        if args.split == 'func':
            trans_stmt_to_function(in_path, out_path)
        elif args.split == 'win':
            window(in_path, out_path)
        else:
            raise NotImplementedError

    end = time.time()
    print(f"Process {len(files)} files.\nTime: {round((end-start)/3600, 2)} Hours.\n")
    