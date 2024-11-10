#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// a simple model with 2 tensors, A and B
struct my_model {
    // Tensor A - will store model parameters
    struct ggml_tensor * a;

    // Tensor B - will store model parameters 
    struct ggml_tensor * b;

    // GGML context that manages the lifecycle and memory of the tensors.
    // This context is used to:
    // - Define tensor dimensions and data types
    // - Allocate memory for tensor data
    // - Track tensor dependencies
    // - Handle memory deallocation when context is freed
    struct ggml_context * ctx;
};

// initialize the tensors of the model, in this demo it is 2 matrices A and B
void load_model(my_model & model, float * a, float * b, int rows_A, int cols_A, int rows_B, int cols_B) {
    size_t ctx_size = 0;
    {
        ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // size for tensor A
        ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32); // size for tensor B
        ctx_size += 2 * ggml_tensor_overhead(); // overhead for each tensor
        ctx_size += ggml_graph_overhead(); // overhead for the computation graph
        ctx_size += 1024; // general overhead
    }

    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    model.ctx = ggml_init(params);


    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);

    memcpy(model.a->data, a, ggml_nbytes(model.a)); // memcpy(TO, FROM, SIZE), copy data from a to the model.a 
    memcpy(model.b->data, b, ggml_nbytes(model.b)); // copy data from b to the model.b
}


// build the graph for matrix multiplication
struct ggml_cgraph * build_graph(const my_model & model) {
    struct ggml_cgraph * gf = ggml_new_graph(model.ctx);

    // result = A*B^T
    struct ggml_tensor * result = ggml_mul_mat(model.ctx, model.a, model.b);

    // add the result tensor to the graph
    ggml_build_forward_expand(gf, result);

    return gf;
}

// use the backend to compute the graph
struct ggml_tensor * compute(const my_model & model) {
    struct ggml_cgraph * gf = build_graph(model);

    int n_threads = 1;

    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // in this demo, the output tesnor is the last in the graph
    return ggml_graph_node(gf, -1);
}

// main function
int main(void) {

    ggml_time_init();

    const int rows_A = 4, cols_A = 2;
    const int rows_B = 3, cols_B = 2;

    float matrix_A[rows_A * cols_A] = {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    };

    float matrix_B[rows_B * cols_B] = {
        1, 2, 
        3, 4,
        5, 6
    };


    my_model model;
    load_model(model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);

    struct ggml_tensor * result = compute(model);

    std::vector<float> out_data(ggml_nelements(result));
    memcpy(out_data.data(), result->data, ggml_nbytes(result));

    printf("result for mul mat (%d x %d) (transposed):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1]; j++) { // rows
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0]; i++) { // cols
            printf(" %.2f ", out_data[j * result->ne[0] + i]);
        }
       
    }
     printf("]\n");


    ggml_free(model.ctx);


    return 0;
}

