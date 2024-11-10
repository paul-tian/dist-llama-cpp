# Distributed llama.cpp 

This is the note for my experiment execution

all the files for myself are in the ```my-exp``` folder

make sure the current working directory is ```my-exp/```


## Demo for ggml and backend
compile the basic exp:

if for CUDA (Linux with Nvidia GPUs, 3070 and 3090 in my case):

```bash
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release --target exp-backend
./bin/exp-backend
cmake --build . --config Release --target exp-ctx
./bin/exp-ctx
cmake --build . --config Release --target exp-ctx-cxx
./bin/exp-ctx-cxx
```

if for Metal (Mac):
```bash
cmake ..
cmake --build . --config Release --target exp-backend
./bin/exp-backend
cmake --build . --config Release --target exp-ctx
./bin/exp-ctx
cmake --build . --config Release --target exp-ctx-cxx
./bin/exp-ctx-cxx
```


## Experiment for distributed llama.cpp

compile the distributed version:

if for CUDA (Linux with Nvidia GPUs, 3070 and 3090 in my case): 

```bash
mkdir rpc-dist
cd rpc-dist
cmake ../.. -DGGML_RPC=ON -DGGML_CUDA=ON
cmake --build . --config Release
```

if for metal (Mac):
```bash
mkdir rpc-dist
cd rpc-dist
cmake ../.. -DGGML_RPC=ON
cmake --build . --config Release
```


then on the distributor machine:

```bash
./bin/exp-rpc --host 0.0.0.0 --port PORT_NUM
```

notice the ```---host``` need to set otherwise the rpc will only open to the local machine


then on the coordinator machine:

```bash
./bin/llama-cli -m PATH_TO_MODEL -p "My name is" --repeat-penalty 1.0 -n -1 --rpc DISTRIBUTOR_IP:PORT_NUM -ngl 200
```

if having multiple distributor machines, they can be separated by comma:
```bash
./bin/llama-cli -m PATH_TO_MODEL -p "My name is" --repeat-penalty 1.0 -n -1 --rpc DISTRIBUTOR_1_IP:PORT_NUM,DISTRIBUTOR_2_IP:PORT_NUM,DISTRIBUTOR_3_IP:PORT_NUM -ngl 200
```








our models coming from:

https://huggingface.co/TheBloke/Llama-2-13B-GGUF


https://huggingface.co/TheBloke/Llama-2-7B-GGUF

https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF

The models are
- llama-2-13b.Q6_K.gguf
- llama-2-13b.Q8_0.gguf
- llama-2-7b.Q6_K.gguf
- llama-2-7b.Q8_0.gguf
- llama-3-8B-Instruct-Q4_K_M.gguf
- llama-3-8B-Instruct-Q6_K.gguf
- llama-3-8B-Instruct-Q8_0.gguf
