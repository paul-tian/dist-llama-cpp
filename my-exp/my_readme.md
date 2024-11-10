# distributed llama.cpp 

This is the note for my experiment execution

compile the distributed version:

if for CUDA (Linux with Nvidia 3090): 

```bash
mkdir rpc-dist
cd rpc-dist
cmake .. -DGGML_CUDA=ON -DGGML_RPC=ON
cmake --build . --config Releas
```

if for metal (Mac):
```bash
mkdir rpc-dist
cd rpc-dist
cmake.. -DGGML_RPC=ON
cmake --build . --config Releasee
```

then on the distributor machine:

```bash
bin/rpc-server --host 0.0.0.0 --port PORT_NUM
```

notice the ```---host``` need to set otherwise the rpc will only open to the local machine


then on the coordinator machine:

```bash
bin/llama-cli -m ../../llama-model/llama-3-8B-Instruct-Q8_0.gguf -p "HMy name is" --repeat-penalty 1.0 -n -1 --rpc DISTRIBUTOR_IP:PORT_NUM -ngl 200
```

if having multiple distributor machines:
```bash
bin/llama-cli -m ../../llama-model/llama-3-8B-Instruct-Q8_0.gguf -p "HMy name is" --repeat-penalty 1.0 -n -1 --rpc 1_DISTRIBUTOR_IP:PORT_NUM,2_DISTRIBUTOR_IP:PORT_NUM -ngl 200
```


our models coming from:

https://huggingface.co/TheBloke/Llama-2-13B-GGUF


https://huggingface.co/TheBloke/Llama-2-7B-GGUF

https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF


# Demo for ggml and backend

