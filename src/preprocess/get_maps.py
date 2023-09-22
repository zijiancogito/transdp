import os
import sys
import time
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import re

sys.path.append('./dwarfinfo')

import elf_parser
import src_map


def normlize_asm_aarch64(s):
    s = re.sub('\[[^\]]+\]', 'MEM', s, 100)
    s = re.sub('\#0x[0-9a-fA-F]+', 'IMM', s, 100)
    s = re.sub('\#[0-9]+\.[0-9]+', 'FLOAT', s, 100)
    s = re.sub('\#[0-9]+', 'IMM', s, 100)
    return s

def normlize_asm_x64(s):
    s = re.sub('\[[^\]]+\]', 'MEM', s, 100)
    s = re.sub('0x[0-9a-fA-F]+', 'IMM', s, 100)
    s = re.sub(' [0-9]+ ', ' IMM ', s, 100)
    return s

def normlize_src(s):
    s = re.sub('v[0-9]+', 'MEM', s, 100)
    s = re.sub('\(MEM\)', 'MEM', s, 100)
    s = re.sub('[0-9]+', 'IMM', s, 100)
    s = re.sub('\(IMM\)', 'IMM', s, 100)
    return s

def process_file(arch, ir, ir_dir, bin_dir, src_dir, asm_list_dir, src_list_dir, map_dir, pd_dir, asm_norm=1, src_norm=1):
    ir_p = os.path.join(ir_dir, ir)
    bname = ir.split('.')[0]

    src_p = os.path.join(src_dir, f"{bname}.c")
    bin_p = os.path.join(bin_dir, f"{bname}.o")
    asm_p = os.path.join(asm_list_dir, f"{bname}.list")
    src_list_p = os.path.join(src_list_dir, f"{bname}.list")

    insns = elf_parser.disassemble(ir_p, arch)
    with open(asm_p, 'w') as f:
        for insn in insns:
            f.write(f"{insn}:\t{insns[insn]}\n")
    
    srcs = [src_p]
    src_lines = src_map.source(srcs)
    with open(src_list_p, 'w') as f:
        for line in src_lines:
            f.write(f"{line}:\t{src_lines[line]}")
            
    _, line_map, _ = elf_parser.parse_dwarf(bin_p)
    # for k in line_map:
        # print(k)
    # print(line_map)

    # import pdb
    # pdb.set_trace()
    maps = src_map.map_src_vs_asm(src_list_p, asm_p, line_map)
    # map_asm = os.path.join(map_dir, f"{bname}.asm")
    # map_src = os.path.join(map_dir, f"{bname}.c")
    # asm_f = open(map_asm, 'w')
    # src_f = open(map_src, 'w')

    # for key in maps:
        # asm_f.write(key[0].strip())
        # asm_f.write('\n')
        # src_f.write(' ; '.join(key[1]))
        # src_f.write('\n')
    # asm_f.close()
    # src_f.close()
    pd_f = os.path.join(pd_dir, f"{bname}.csv")
    # print(pd_f)
    df = pd.DataFrame(columns=['input', 'target'])
    for idx, key in enumerate(maps):
        # print(key)
        col1 = re.sub(r'\([\s]*(int|float|double)[\s]*\)', '', key[0].strip(), 100)
        if src_norm == 1:
            col1 = normlize_src(col1)
        col2 = None
        if asm_norm == 1:
            if arch == 'aarch64':
                col2 = ' ; '.join([normlize_asm_aarch64(i) for i in key[1]])
            elif arch == 'x64':
                col2 = ' ; '.join([normlize_asm_x64(i) for i in key[1]])
            else:
                raise NotImplementedError
        else:
            col2 = ' ; '.join(key[1])
        df.loc[idx] = [col2, col1]
        # df.add([col1, col2])
    df.to_csv(pd_f, index=True, header=True, sep='\t')
    # import pdb
    # pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='get_maps.py')
    parser.add_argument('-o', '--save', type=str, default='../', help='root path to save the results')
    parser.add_argument('-s', '--src', type=str, default='../', help='path to source code')
    parser.add_argument('-j', '--proc', type=int, default=28, help='number of processing')
    parser.add_argument('-M', '--arch', type=str, default='x64', choices=['x64', 'aarch64', 'mips64'],help='')
    parser.add_argument('-sn', '--asm-norm', type=int, default=1, choices=[0, 1],help='')
    parser.add_argument('-cn', '--src-norm', type=int, default=1, choices=[0, 1],help='')

    args = parser.parse_args()

    ir_dir = os.path.join(args.save, 'ir')
    bin_dir = os.path.join(args.save, 'bin')
    src_dir = args.src
    asm_dir = os.path.join(args.save, 'asm_list')
    src_list_dir = os.path.join(args.save, 'src_list')
    map_dir = os.path.join(args.save, 'map')
    pd_dir = None
    if args.asm_norm == 1 and args.src_norm == 1:
        pd_dir = os.path.join(args.save, 'csv-all')
    elif args.asm_norm == 1 and args.src_norm == 0:
        pd_dir = os.path.join(args.save, 'csv-asm')
    elif args.asm_norm == 0 and args.src_norm == 1:
        pd_dir = os.path.join(args.save, 'csv-src')
    elif args.asm_norm == 0 and args.src_norm == 0:
        pd_dir = os.path.join(args.save, 'csv-no')
    else:
        raise NotImplementedError



    if not os.path.exists(asm_dir):
        os.makedirs(asm_dir)
    if not os.path.exists(src_list_dir):
        os.makedirs(src_list_dir)
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    if not os.path.exists(pd_dir):
        os.makedirs(pd_dir)

    files = os.listdir(ir_dir)
    start = time.time()
    pool = multiprocessing.Pool(processes=args.proc)
    # for f in files:
    #     process_file(args.arch, 
    #                                               f, 
    #                                               ir_dir,
    #                                               bin_dir,
    #                                               src_dir,
    #                                               asm_dir,
    #                                               src_list_dir,
    #                                               map_dir,
    #                                               pd_dir)
        # pool.apply_async(func=process_file, args=(args.arch, 
                                                  # f, 
                                                  # ir_dir,
                                                  # bin_dir,
                                                  # src_dir,
                                                  # asm_dir,
                                                  # src_list_dir,
                                                  # map_dir,
                                                  # pd_dir))
    iter = [(args.arch, f, ir_dir, bin_dir, src_dir, asm_dir, src_list_dir,
             map_dir, pd_dir, args.asm_norm, args.src_norm) for f in files]
    pool.starmap(process_file, iter, 10000)
    
    pool.close()
    pool.join()

    end = time.time()
    print(f"Process {len(files)} files.\nTime: {round((end-start)/3600, 2)} Hours.\nProcs: {args.proc}.")

