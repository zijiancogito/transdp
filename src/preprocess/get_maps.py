import os
import sys
import time
import argparse
import multiprocessing
import numpy as np
import pandas as pd

sys.path.append('./dwarfinfo')

import elf_parser
import src_map

def process_file(arch, ir, ir_dir, bin_dir, src_dir, asm_list_dir, src_list_dir, map_dir, pd_dir):
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
    df = pd.DataFrame(columns=['input', 'target'])
    for key in maps:
        # print(key)
        col1 = key[0].strip()
        col2 = ' ; '.join(key[1])
        print(col1)
        print(col2)
        df.add([col1, col2])
    df.to_csv(pd_f, index=True, header=True, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='get_maps.py')
    parser.add_argument('-o', '--save', type=str, default='../', help='root path to save the results')
    parser.add_argument('-s', '--src', type=str, default='../', help='path to source code')
    parser.add_argument('-j', '--proc', type=int, default=28, help='number of processing')
    parser.add_argument('-M', '--arch', type=str, default='x64', choices=['x64', 'aarch64', 'mips64'],help='')
    args = parser.parse_args()

    ir_dir = os.path.join(args.save, 'ir')
    bin_dir = os.path.join(args.save, 'bin')
    src_dir = args.src
    asm_dir = os.path.join(args.save, 'asm_list')
    src_list_dir = os.path.join(args.save, 'src_list')
    map_dir = os.path.join(args.save, 'map')
    pd_dir = os.path.join(args.save, 'csv')

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
    for f in files:
        process_file(args.arch, 
                                                  f, 
                                                  ir_dir,
                                                  bin_dir,
                                                  src_dir,
                                                  asm_dir,
                                                  src_list_dir,
                                                  map_dir,
                                                  pd_dir)
        # pool.apply_async(func=process_file, args=(args.arch, 
                                                  # f, 
                                                  # ir_dir,
                                                  # bin_dir,
                                                  # src_dir,
                                                  # asm_dir,
                                                  # src_list_dir,
                                                  # map_dir,
                                                  # pd_dir))
    # iter = [(args.arch, f, ir_dir, bin_dir, src_dir, asm_dir, src_list_dir,
             # map_dir, pd_dir) for f in files]
    # pool.starmap(process_file, iter, 10000)
    
    pool.close()
    pool.join()

    end = time.time()
    print(f"Process {len(files)} files.\nTime: {round((end-start)/3600, 2)} Hours.\nProcs: {args.proc}.")

