import os
import csv
import argparse

def extract_exp(d, out, count=1000):
    files = os.listdir(d)
    c1, c2, c3, c4 = [], [], [], []
    for f in files:
        path = os.path.join(d, f)
        with open(path, 'r') as fp:
            # alls = csv.reader(fp, delimiter='\t', skipinitialspace=True)
            alls = fp.readlines()[1:]
            for line in alls:
                l = line.split('\t')
                if l[-1].strip() == '{' or l[-1].strip() == '}':
                    continue
                if len(c2) < count:
                    if 'if' in l[-1]:
                        c2.append(l)
                        continue
                if len(c3) < count:
                    if 'while' in l[-1]:
                        c3.append(l)
                        continue
                if len(c4) < count:
                    if 'f_rand' in l[-1]:
                        c4.append(l)
                        continue
                if len(c1) < count:
                    c1.append(l)
                    continue
        print(f"{len(c1)}\t{len(c2)}\t{len(c3)}\t{len(c4)}")
        if len(c1) >= count and len(c2) >= count and len(c3) >= count and len(c4) >= count:
            break
    with open(os.path.join(out, 'exp.csv'), 'w') as f:
        f.write('\tinput\ttarget\n')
        for l in c1:
            f.write('\t'.join(l))
    with open(os.path.join(out, 'if.csv'), 'w') as f:
        f.write('\tinput\ttarget\n')
        for l in c2:
            f.write('\t'.join(l))
    with open(os.path.join(out, 'while.csv'), 'w') as f:
        f.write('\tinput\ttarget\n')
        for l in c3:
            f.write('\t'.join(l))
    with open(os.path.join(out, 'call.csv'), 'w') as f:
        f.write('\tinput\ttarget\n')
        for l in c4:
            f.write('\t'.join(l))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='eval_ds.py')
    parser.add_argument('-o', '--save', type=str, default='./')
    parser.add_argument('-i', '--csv', type=str)
    parser.add_argument('-n', '--num', type=int, default=1000)

    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    extract_exp(args.csv, args.save, args.num)

    