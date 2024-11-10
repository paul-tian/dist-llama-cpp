# distributed llama.cpp 

This is the note for my experiment execution

all the files for myself are in the ```my-exp``` folder

make sure the current working directory is ```my-exp/```


compile the basic exp:

```bash
cmake ..
cmake --build . --config Release --target exp-backend
./bin/exp-backend
cmake --build . --config Release --target exp-ctx
./bin/exp-ctx
cmake --build . --config Release --target exp-ctx-cxx
./bin/exp-ctx-cxx
```


compile the distributed version:



if for CUDA (Linux with Nvidia 3090): 

```bash
mkdir rpc-dist
cd rpc-dist
cmake ../.. -DEXP_RPC=ON -DGGML_CUDA=ON
cmake --build . --config Release
```

if for metal (Mac):
```bash
mkdir rpc-dist
cd rpc-dist
cmake ../.. -DEXP_RPC=ON
cmake --build . --config Releasee
```


then on the distributor machine:

```bash
./bin/exp-rpc --host 0.0.0.0 --port PORT_NUM
```

notice the ```---host``` need to set otherwise the rpc will only open to the local machine


then on the coordinator machine:

```bash
./bin/llama-cli -m ../../llama-model/llama-3-8B-Instruct-Q8_0.gguf -p "HMy name is" --repeat-penalty 1.0 -n -1 --rpc DISTRIBUTOR_IP:PORT_NUM -ngl 200
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

