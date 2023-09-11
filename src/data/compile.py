import os
import sys
import argparse
import multiprocessing
import time

def compile(arch, filename, d, t):
    src = os.path.join(d, filename)
    obj = os.path.join(t, f"{filename.split('.')[0]}.o")
    cc = None
    if arch == 'x64':
        cc = 'gcc'
    elif arch == 'aarch64':
        cc = 'aarch64-linux-gnu-gcc'
    else:
        raise NotImplemented
    cmd = f"{cc} -O0 -fno-inline-functions -g {src} -o {obj}"
    os.system(cmd)

def ddisasm(filename, d, t):
    obj = os.path.join(d, filename)
    ir = os.path.join(t, f"{filename.split('.')[0]}.gtirb")
    cmd = f"ddisasm {obj} --ir {ir} > /dev/null 2>&1"
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='source_gen.py')
    parser.add_argument('-j', '--proc', type=int, default=16, help='number of processing')
    parser.add_argument('-o', '--save', type=str, default='../', help='root path to save the results')
    parser.add_argument('-i', '--input', type=str, default='../', help='path to source code')
    parser.add_argument('-M', '--arch', type=str, default='x64', choices=['x64', 'aarch64'],help='path to source code')


    subparser = parser.add_subparsers(help='sub-command help')
    parser_compile = subparser.add_parser('compile', help='compile help')
    parser_disasm = subparser.add_parser('disasm', help='disasm help')

    parser_compile.set_defaults(func=compile)
    parser_disasm.set_defaults(func=ddisasm)

    args = parser.parse_args()
    start = time.time()
    files = os.listdir(args.input)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    print(args.func)
    pool = multiprocessing.Pool(processes=args.proc)
    args = [(args.arch, f, args.input, args.save) for f in files]
    pool.starmap(args.func, args, 10000)
    # for f in files:
        # pool.apply_async(func=args.func, args=(args.arch, f, args.input, args.save))
    
    pool.close()
    pool.join()

    end = time.time()
    print(f"{args.func} {len(files)} files.\nTime: {end-start}/60 Mins.\nProcs: {args.proc}.")