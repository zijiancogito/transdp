# transdp


## Install 

- apt-get install git python3 python3-pip gcc-aarch64-linux-gnu
- git clone https://github.com/zijiancogito/transdp.git --recurse-submodules
- pip install cfile
- pip install pyelftools capstone ddisasm
- pip install torch numpy pandas transformers evaluate sacrebleu evaluate aim

## Usage

### dataset

```
    cd src/data/
    python3 datagen.py -o /root/data/src -n 500000 -j 20
    
    # X86
    python3 compile.py -o /root/data/x64/bin -j 20 -i /root/data/src -M x64 compile
    python3 compile.py -o /root/data/x64/ir -j 20 -i /root/data/x64/bin -M x64 disasm
    
    # AARCH64
    python3 compile.py -o /root/data/aarch64/bin -j 20 -i /root/data/src -M aarch64 compile
    python3 compile.py -o /root/data/aarch64/ir -j 20 -i /root/data/aarch64/bin -M aarch64 disasm
    
    # MIPS 
    python3 compile.py -o /root/data/mips/bin -j 20 -i /root/data/src -M mips compile
    python3 compile.py -o /root/data/mips/ir -j 20 -i /root/data/mips/bin -M mips disasm

    cd ../preprocess

    # norm asm and src
    python3 get_maps.py -o /root/data/x64 -s /root/data/src -j 20 -M x64 -sn 1 -cn 1
    python3 get_maps.py -o /root/data/aarch64 -s /root/data/src -j 20 -M aarch64 -sn 1 -cn 1
    python3 get_maps.py -o /root/data/mips -s /root/data/src -j 20 -M mips -sn 1 -cn 1

    # only norm asm
    python3 get_maps.py -o /root/data/x64 -s /root/data/src -j 20 -M x64 -sn 1 -cn 0
    python3 get_maps.py -o /root/data/aarch64 -s /root/data/src -j 20 -M aarch64 -sn 1 -cn 0
    python3 get_maps.py -o /root/data/mips -s /root/data/src -j 20 -M mips -sn 1 -cn 0

    # only norm src
    python3 get_maps.py -o /root/data/x64 -s /root/data/src -j 20 -M x64 -sn 0 -cn 1
    python3 get_maps.py -o /root/data/aarch64 -s /root/data/src -j 20 -M aarch64 -sn 0 -cn 1
    python3 get_maps.py -o /root/data/mips -s /root/data/src -j 20 -M mips -sn 0 -cn 1

    # no norm
    python3 get_maps.py -o /root/data/x64 -s /root/data/src -j 20 -M x64 -sn 0 -cn 0
    python3 get_maps.py -o /root/data/aarch64 -s /root/data/src -j 20 -M aarch64 -sn 0 -cn 0
    python3 get_maps.py -o /root/data/mips -s /root/data/src -j 20 -M mips -sn 0 -cn 0
    
    
    
```

### Eval dataset
```
    cd ./preprocess

    # aarch64
    python3 eval_ds.py -i /root/data/aarch64/csv -o /root/data/aarch64/eval -n 1000
    # x64
    python3 eval_ds.py -i /root/data/x64/csv -o /root/data/x64/eval -n 1000
```

### Segmente

```
    cd ./preprocess

    # aarch64 
    python3 func_sp.py -s func -i /root/data/aarch64/csv -o /root/data/aarch64/csv-func
    python3 func_sp.py -s win -i /root/data/aarch64/csv -o /root/data/aarch64/csv-win

    # x64
    python3 func_sp.py -s func -i /root/data/x64/csv -o /root/data/x64/csv-func
    python3 func_sp.py -s func -i /root/data/x64/csv -o /root/data/x64/csv-win

    # mips
    python3 func_sp.py -s func -i /root/data/mips/csv-all -o /root/data/x64/csv-func
    python3 func_sp.py -s func -i /root/data/mips/csv -o /root/data/x64/csv-win

```


### 