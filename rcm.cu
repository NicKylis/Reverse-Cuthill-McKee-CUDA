#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>

typedef struct {
    int *data;
    int size;
    int cap;
} IntVec;

void vec_init(IntVec *v) {
    v->cap = 4;
    v->size = 0;
    v->data = (int*)malloc(v->cap * sizeof(int));
}

void vec_push(IntVec *v, int x) {
    if (v->size == v->cap) {
        v->cap *= 2;
        v->data = (int*)realloc(v->data, v->cap * sizeof(int));
    }
    v->data[v->size++] = x;
}

void vec_free(IntVec *v) {
    free(v->data);
    v->data = NULL;
    v->size = v->cap = 0;
}

// CUDA kernel to initialize random number generators
__global__ void init_curand(curandState *state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// CUDA kernel to generate candidate edges
__global__ void generate_edges(int *edges, int n, int m, curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        curandState local_state = state[idx];
        int u = curand(&local_state) % n;
        int v = curand(&local_state) % n;
        edges[idx * 2] = u;
        edges[idx * 2 + 1] = v;
        state[idx] = local_state;
    }
}

// CUDA kernel to validate edges (no self-loops, ensure u < v for uniqueness)
__global__ void validate_edges(int *edges, int *valid, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        int u = edges[idx * 2];
        int v = edges[idx * 2 + 1];
        valid[idx] = (u != v && u >= 0 && u < n && v >= 0 && v < n && u < v) ? 1 : 0;
    }
}

// CUDA kernel to compute degrees for CSR row_ptr
__global__ void compute_degrees(int *row_ptr, int *degrees, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        degrees[idx] = row_ptr[idx + 1] - row_ptr[idx];
    }
}

// CUDA kernel to find min-degree unvisited node
__global__ void find_min_degree_unvisited(int *degrees, int *visited, int n, int *min_deg, int *min_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && !visited[idx]) {
        int deg = degrees[idx];
        atomicMin(min_deg, deg);
        if (deg == *min_deg) {
            atomicMin(min_idx, idx);
        }
    }
}

// CUDA kernel to mark visited nodes and build queue
__global__ void mark_visited(int *col_idx, int start, int size, int *visited, int *queue, int *qtail) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int v = col_idx[start + idx];
        if (!visited[v]) {
            visited[v] = 1;
            int pos = atomicAdd(qtail, 1);
            queue[pos] = v;
        }
    }
}

void generate_random_graph_gpu(int n, int m, int **d_row_ptr, int **d_col_idx, int *total_edges) {
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    // Allocate GPU memory for edge generation
    int *d_edges, *d_valid;
    curandState *d_state;
    cudaMalloc(&d_edges, m * 2 * sizeof(int));
    cudaMalloc(&d_valid, m * sizeof(int));
    cudaMalloc(&d_state, m * sizeof(curandState));

    // Initialize random number generators
    init_curand<<<grid_size, block_size>>>(d_state, time(NULL), m);
    cudaDeviceSynchronize();

    // Generate candidate edges
    generate_edges<<<grid_size, block_size>>>(d_edges, n, m, d_state);
    cudaDeviceSynchronize();

    // Validate edges
    validate_edges<<<grid_size, block_size>>>(d_edges, d_valid, m, n);
    cudaDeviceSynchronize();

    // Count valid edges using CUB
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    int *d_valid_count;
    cudaMalloc(&d_valid_count, sizeof(int));
    cudaMemset(d_valid_count, 0, sizeof(int));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_valid, d_valid_count, m);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_valid, d_valid_count, m);
    int valid_edges;
    cudaMemcpy(&valid_edges, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Allocate memory for CSR format
    cudaMalloc(d_row_ptr, (n + 1) * sizeof(int));
    cudaMemset(*d_row_ptr, 0, (n + 1) * sizeof(int));
    cudaMalloc(d_col_idx, valid_edges * 2 * sizeof(int));

    // Build CSR format (row_ptr and col_idx)
    int *d_edge_counts;
    cudaMalloc(&d_edge_counts, n * sizeof(int));
    cudaMemset(d_edge_counts, 0, n * sizeof(int));

    // Count edges per node
    grid_size = (m + block_size - 1) / block_size;
    for (int i = 0; i < m; ++i) {
        int u, v, valid;
        cudaMemcpy(&u, d_edges + i * 2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v, d_edges + i * 2 + 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&valid, d_valid + i, sizeof(int), cudaMemcpyDeviceToHost);
        if (valid) {
            int one = 1;
            cudaMemcpy(d_edge_counts + u, &one, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_edge_counts + v, &one, sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    // Compute row_ptr using CUB prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_edge_counts, *d_row_ptr + 1, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_edge_counts, *d_row_ptr + 1, n);
    cudaMemcpy(total_edges, *d_row_ptr + n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Populate col_idx
    int *h_col_idx = (int*)malloc(valid_edges * 2 * sizeof(int));
    int pos = 0;
    for (int i = 0; i < m; ++i) {
        int u, v, valid;
        cudaMemcpy(&u, d_edges + i * 2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v, d_edges + i * 2 + 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&valid, d_valid + i, sizeof(int), cudaMemcpyDeviceToHost);
        if (valid) {
            h_col_idx[pos++] = v; // u -> v
            h_col_idx[pos++] = u; // v -> u (undirected)
        }
    }
    cudaMemcpy(*d_col_idx, h_col_idx, valid_edges * 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Cleanup
    cudaFree(d_edges);
    cudaFree(d_valid);
    cudaFree(d_state);
    cudaFree(d_valid_count);
    cudaFree(d_edge_counts);
    cudaFree(d_temp_storage);
    free(h_col_idx);
}

int *rcm_cuda(int *d_row_ptr, int *d_col_idx, int n, int total_edges) {
    // Allocate GPU memory
    int *d_degrees, *d_visited, *d_queue, *d_order;
    int *d_min_deg, *d_min_idx, *d_qtail, *d_visited_count;
    cudaMalloc(&d_degrees, n * sizeof(int));
    cudaMalloc(&d_visited, n * sizeof(int));
    cudaMalloc(&d_queue, n * sizeof(int));
    cudaMalloc(&d_order, n * sizeof(int));
    cudaMalloc(&d_min_deg, sizeof(int));
    cudaMalloc(&d_min_idx, sizeof(int));
    cudaMalloc(&d_qtail, sizeof(int));
    cudaMalloc(&d_visited_count, sizeof(int));

    cudaMemset(d_visited, 0, n * sizeof(int));
    cudaMemset(d_visited_count, 0, sizeof(int));

    // Compute degrees
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    compute_degrees<<<grid_size, block_size>>>(d_row_ptr, d_degrees, n);
    cudaDeviceSynchronize();

    int *order = (int*)malloc(n * sizeof(int));
    int order_pos = 0;
    int visited_count = 0;

    while (visited_count < n) {
        int h_min_deg = INT_MAX, h_min_idx = -1;
        cudaMemset(d_min_deg, 0x7f, sizeof(int));
        cudaMemset(d_min_idx, 0x7f, sizeof(int));
        find_min_degree_unvisited<<<grid_size, block_size>>>(d_degrees, d_visited, n, d_min_deg, d_min_idx);
        cudaMemcpy(&h_min_deg, d_min_deg, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_min_idx, d_min_idx, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (h_min_idx == -1) break;

        int qhead = 0, qtail = 0;
        int *h_queue = (int*)malloc(n * sizeof(int));
        h_queue[qtail++] = h_min_idx;
        int one = 1;
        cudaMemcpy(d_visited + h_min_idx, &one, sizeof(int), cudaMemcpyHostToDevice);
        visited_count++;

        while (qhead < qtail) {
            int u = h_queue[qhead++];
            order[order_pos++] = u;

            // Sort neighbors by degree
            int start = 0;
            cudaMemcpy(&start, d_row_ptr + u, sizeof(int), cudaMemcpyDeviceToHost);
            int size = 0;
            cudaMemcpy(&size, d_row_ptr + u + 1, sizeof(int), cudaMemcpyDeviceToHost);
            size -= start;
            if (size > 0) {
                void *d_temp_storage = NULL;
                size_t temp_storage_bytes = 0;
                int *d_keys = d_col_idx + start;
                int *d_values;
                cudaMalloc(&d_values, size * sizeof(int));

                // Map degrees to neighbors
                for (int i = 0; i < size; ++i) {
                    int v;
                    cudaMemcpy(&v, d_col_idx + start + i, sizeof(int), cudaMemcpyDeviceToHost);
                    int deg;
                    cudaMemcpy(&deg, d_degrees + v, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(d_values + i, &deg, sizeof(int), cudaMemcpyHostToDevice);
                }

                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                d_values, d_values, d_keys, d_keys, size);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                d_values, d_values, d_keys, d_keys, size);
                cudaFree(d_temp_storage);
                cudaFree(d_values);
            }

            // Mark unvisited neighbors
            cudaMemset(d_qtail, 0, sizeof(int));
            mark_visited<<<(size + block_size - 1) / block_size, block_size>>>(d_col_idx, start, size, d_visited, d_queue, d_qtail);
            cudaMemcpy(&qtail, d_qtail, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_queue + qhead, d_queue, qtail * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            visited_count += qtail;
        }
        free(h_queue);
    }

    // Create permutation
    int *perm = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) perm[i] = order[n - 1 - i];

    // Cleanup
    cudaFree(d_degrees);
    cudaFree(d_visited);
    cudaFree(d_queue);
    cudaFree(d_order);
    cudaFree(d_min_deg);
    cudaFree(d_min_idx);
    cudaFree(d_qtail);
    cudaFree(d_visited_count);
    free(order);

    return perm;
}

int main(int argc, char **argv) {
    int n, m;
    int *d_row_ptr, *d_col_idx;
    int total_edges = 0;

    if (argc == 4 && strcmp(argv[1], "--random") == 0) {
        n = atoi(argv[2]);
        m = atoi(argv[3]);
        if (n <= 0 || m < 0) {
            fprintf(stderr, "Invalid N or M\n");
            return 1;
        }
        generate_random_graph_gpu(n, m, &d_row_ptr, &d_col_idx, &total_edges);

        // Print generated graph for verification
        // int *h_row_ptr = (int*)malloc((n + 1) * sizeof(int));
        // int *h_col_idx = (int*)malloc(total_edges * sizeof(int));
        // cudaMemcpy(h_row_ptr, d_row_ptr, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_col_idx, d_col_idx, total_edges * sizeof(int), cudaMemcpyDeviceToHost);
        // printf("# Generated random graph:\n%d %d\n", n, total_edges / 2);
        // for (int i = 0; i < n; ++i) {
        //     for (int j = h_row_ptr[i]; j < h_row_ptr[i + 1]; ++j) {
        //         int v = h_col_idx[j];
        //         if (i < v) { // Print each edge once
        //             printf("%d %d\n", i, v);
        //         }
        //     }
        // }
        // printf("# End of graph\n");
        // free(h_row_ptr);
        // free(h_col_idx);
    } else {
        if (scanf("%d %d", &n, &m) != 2) {
            fprintf(stderr, "Expected: n m\n");
            return 1;
        }
        // Read edges on CPU
        int *h_edges = (int*)malloc(m * 2 * sizeof(int));
        int valid_edges = 0;
        for (int i = 0; i < m; ++i) {
            int u, v;
            if (scanf("%d %d", &u, &v) != 2) {
                fprintf(stderr, "Expected edge list\n");
                free(h_edges);
                return 1;
            }
            if (u < 0 || u >= n || v < 0 || v >= n || u == v) continue;
            h_edges[valid_edges * 2] = u < v ? u : v;
            h_edges[valid_edges * 2 + 1] = u < v ? v : u;
            valid_edges++;
        }

        // Build CSR on GPU
        int *d_edge_counts;
        cudaMalloc(&d_edge_counts, n * sizeof(int));
        cudaMemset(d_edge_counts, 0, n * sizeof(int));
        total_edges = valid_edges * 2;
        cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
        cudaMemset(d_row_ptr, 0, (n + 1) * sizeof(int));
        cudaMalloc(&d_col_idx, total_edges * sizeof(int));

        // Count edges per node
        for (int i = 0; i < valid_edges; ++i) {
            int u = h_edges[i * 2];
            int v = h_edges[i * 2 + 1];
            int one = 1;
            cudaMemcpy(d_edge_counts + u, &one, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_edge_counts + v, &one, sizeof(int), cudaMemcpyHostToDevice);
        }

        // Compute row_ptr
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_edge_counts, d_row_ptr + 1, n);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_edge_counts, d_row_ptr + 1, n);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        // Populate col_idx
        int *h_col_idx = (int*)malloc(total_edges * sizeof(int));
        int pos = 0;
        for (int i = 0; i < valid_edges; ++i) {
            h_col_idx[pos++] = h_edges[i * 2 + 1]; // u -> v
            h_col_idx[pos++] = h_edges[i * 2];     // v -> u
        }
        cudaMemcpy(d_col_idx, h_col_idx, total_edges * sizeof(int), cudaMemcpyHostToDevice);
        free(h_edges);
        free(h_col_idx);
        cudaFree(d_edge_counts);
    }

    int *perm = rcm_cuda(d_row_ptr, d_col_idx, n, total_edges);

    printf("# RCM order:\n");
    for (int i = 0; i < n; ++i) {
        if (i) printf(" ");
        printf("%d", perm[i]);
    }
    printf("\n");

    // Cleanup
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    free(perm);

    return 0;
}