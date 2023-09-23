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
    
    # MIPS64 
    python3 compile.py -o /root/data/mips64/bin -j 20 -i /root/data/src -M mips64 compile
    python3 compile.py -o /root/data/mips64/ir -j 20 -i /root/data/mips64/bin -M mips64 disasm

    cd ../preprocess

    # norm asm and src
    python3 get_maps.py -o /root/data/x64 -s /root/data/src -j 20 -M x64 -sn 1 -cn 1
    python3 get_maps.py -o /root/data/aarch64 -s /root/data/src -j 20 -M aarch64 -sn 1 -cn 1
    python3 get_maps.py -o /root/data/mips64 -s /root/data/src -j 20 -M mips64

    # only norm asm
    python3 get_maps.py -o /root/data/x64 -s /root/data/src -j 20 -M x64 -sn 1 -cn 0
    python3 get_maps.py -o /root/data/aarch64 -s /root/data/src -j 20 -M aarch64 -sn 1 -cn 0

    # only norm src
    python3 get_maps.py -o /root/data/x64 -s /root/data/src -j 20 -M x64 -sn 0 -cn 1
    python3 get_maps.py -o /root/data/aarch64 -s /root/data/src -j 20 -M aarch64 -sn 0 -cn 1

    # no norm
    python3 get_maps.py -o /root/data/x64 -s /root/data/src -j 20 -M x64 -sn 0 -cn 0
    python3 get_maps.py -o /root/data/aarch64 -s /root/data/src -j 20 -M aarch64 -sn 0 -cn 0
    
```

