# transdp


## Install 

- apt-get install git python3 python3-pip
- git clone <url> --recurse-submodules
- pip install cfile
- pip install pyelftools capstone ddisasm
- pip install torch numpy pandas transformers evaluate sacrebleu evaluate aim

## Usage

### dataset

```
    cd src/data/
    python3 datagen.py -o /root/transdata/src -n 500000 -j 20
    
    # X86
    python3 compile.py -o /root/transdata/x64/bin -j 20 -s /root/transdata/src -M x64 compile
    python3 compile.py -o /root/transdata/x64/ir -j 20 -s /root/transdata/x64/bin -M x64 disasm
    
    # AARCH64
    python3 compile.py -o /root/transdata/aarch64/bin -j 20 -s /root/transdata/src -M aarch64 compile
    python3 compile.py -o /root/transdata/aarch64/ir -j 20 -s /root/transdata/aarch64/bin -M aarch64 disasm
    
    cd ../preprocess
    python3 get_maps.py -o /root/transdata/x64 -s /root/transdata/src -j 20 -M x64
    python3 get_maps.py -o /root/transdata/aarch64 -s /root/transdata/src -j 20 -M aarch64

```

