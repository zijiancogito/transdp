import sys

sys.path.append('.')
sys.path.append('./DSmith')

import argparse
import os
from source import make_src_file, output_to_file
import multiprocessing

class Config:
    def __init__(self):
        self.max_funcs = 1
        self.max_args = 0
        self.min_args = 0
        self.max_block_size=5
        self.min_block_size=3
        self.max_block_depth = 2
        self.max_branch = 10
        self.max_expr_complexity = 3
        self.min_expr_complexity = 1
        self.max_local_variables = 5
        self.min_local_variables = 2
        self.max_const_variables = 5
        self.min_const_variables = 2
        self.max_const_values = 1000
        self.has_divs = True
        self.has_logic = True
        self.has_float = True
        self.has_double = True

def generator(config: Config, filename):
    src = make_src_file(config.max_funcs,
                        config.max_args,
                        config.min_args,
                        config.max_block_size,
                        config.min_block_size,
                        config.max_block_depth,
                        config.max_branch,
                        config.max_expr_complexity,
                        config.min_expr_complexity,
                        config.max_local_variables,
                        config.min_local_variables,
                        config.max_const_variables,
                        config.min_const_variables,
                        config.max_const_values,
                        config.has_divs,
                        config.has_logic,
                        config.has_float,
                        config.has_double,
                        filename)
    output_to_file(filename, src)

def batch(count, save_dir, proc=28, level=3):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config = Config()
    pool = multiprocessing.Pool(processes=proc)
    # args = [(config, os.path.join(save_dir, f"{i}.c")) for i in range(count)]
    # pool.starmap_async(generator, args, 1000)
    for i in range(count):
        save_path = os.path.join(save_dir, f"{i}.c")
        #print(save_path)
        result = pool.apply_async(func=generator, args=(config, save_path))
        # generator(config, save_path)

    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='source_gen.py')
    parser.add_argument('-o', '--save', type=str, default='../', help='root path to save the results')
    parser.add_argument('-n', '--number', type=int, default=1, help='Number of source to generate')
    parser.add_argument('-l', '--level', type=int, default=3, help='Level of expressions', required=False)
    parser.add_argument('-j', '--proc', type=int, default=28, help='number of processing')

    args = parser.parse_args()

    batch(args.number, args.save, args.proc)