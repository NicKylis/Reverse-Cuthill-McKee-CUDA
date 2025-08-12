#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

int cmp_degree(const void *a, const void *b, void *deg) {
    int da = ((int*)deg)[*(const int*)a];
    int db = ((int*)deg)[*(const int*)b];
    if (da != db) return da - db;
    return (*(const int*)a) - (*(const int*)b);
}

void sort_neighbors_by_degree(IntVec *adj, int node, int *deg) {
#if defined(__GNUC__) && !defined(__clang__)
    qsort_r(adj[node].data, adj[node].size, sizeof(int), cmp_degree, deg);
#else
    qsort(adj[node].data, adj[node].size, sizeof(int),
          (int(*)(const void*, const void*))cmp_degree);
#endif
}

int *rcm(IntVec *adj, int n) {
    int *deg = (int*)malloc(n * sizeof(int));
    int *visited = (int*)calloc(n, sizeof(int));
    int *order = (int*)malloc(n * sizeof(int));
    int order_pos = 0;

    for (int i = 0; i < n; ++i) deg[i] = adj[i].size;

    int *queue = (int*)malloc(n * sizeof(int));

    int visited_count = 0;
    while (visited_count < n) {
        int start = -1;
        for (int i = 0; i < n; ++i)
            if (!visited[i] && (start == -1 || deg[i] < deg[start]))
                start = i;

        int qhead = 0, qtail = 0;
        queue[qtail++] = start;
        visited[start] = 1;
        visited_count++;

        while (qhead < qtail) {
            int u = queue[qhead++];
            order[order_pos++] = u;

            sort_neighbors_by_degree(adj, u, deg);
            for (int i = 0; i < adj[u].size; ++i) {
                int v = adj[u].data[i];
                if (!visited[v]) {
                    visited[v] = 1;
                    visited_count++;
                    queue[qtail++] = v;
                }
            }
        }
    }

    free(queue);
    free(visited);
    free(deg);

    int *perm = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) perm[i] = order[n - 1 - i];
    free(order);

    return perm;
}

void generate_random_graph(int n, int m, IntVec *adj, int **edge_list) {
    srand((unsigned)time(NULL));
    *edge_list = (int*)malloc(m * 2 * sizeof(int));

    int edge_count = 0;
    while (edge_count < m) {
        int u = rand() % n;
        int v = rand() % n;
        if (u == v) continue;

        int duplicate = 0;
        for (int i = 0; i < adj[u].size; ++i) {
            if (adj[u].data[i] == v) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) continue;

        vec_push(&adj[u], v);
        vec_push(&adj[v], u);
        (*edge_list)[edge_count * 2]     = u;
        (*edge_list)[edge_count * 2 + 1] = v;
        edge_count++;
    }
}

int main(int argc, char **argv) {
    int n, m;
    IntVec *adj;
    int *edge_list = NULL;

    if (argc == 4 && strcmp(argv[1], "--random") == 0) {
        n = atoi(argv[2]);
        m = atoi(argv[3]);
        if (n <= 0 || m < 0) {
            fprintf(stderr, "Invalid N or M\n");
            return 1;
        }
        adj = (IntVec*)malloc(n * sizeof(IntVec));
        for (int i = 0; i < n; ++i) vec_init(&adj[i]);
        generate_random_graph(n, m, adj, &edge_list);

        printf("# Generated random graph:\n%d %d\n", n, m);
        for (int i = 0; i < m; ++i) {
            printf("%d %d\n", edge_list[i * 2], edge_list[i * 2 + 1]);
        }
        printf("# End of graph\n");
        free(edge_list);
    } else {
        if (scanf("%d %d", &n, &m) != 2) {
            fprintf(stderr, "Expected: n m\n");
            return 1;
        }
        adj = (IntVec*)malloc(n * sizeof(IntVec));
        for (int i = 0; i < n; ++i) vec_init(&adj[i]);

        for (int i = 0; i < m; ++i) {
            int u, v;
            if (scanf("%d %d", &u, &v) != 2) {
                fprintf(stderr, "Expected edge list\n");
                return 1;
            }
            if (u < 0 || u >= n || v < 0 || v >= n || u == v) continue;
            vec_push(&adj[u], v);
            vec_push(&adj[v], u);
        }
    }

    int *perm = rcm(adj, n);

    printf("# RCM order:\n");
    for (int i = 0; i < n; ++i) {
        if (i) printf(" ");
        printf("%d", perm[i]);
    }
    printf("\n");

    free(perm);
    for (int i = 0; i < n; ++i) vec_free(&adj[i]);
    free(adj);

    return 0;
}
