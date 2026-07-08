%%writefile hybrid_nbody.cu
#include <iostream>
#include <omp.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>


#define MAX_BODIES 10000
#define MAX_NODES 800000 
#define THETA 0.5
#define G_CONST 6.67430e-11
#define EPSILON 1e-9 
#define DT 0.01      


struct Body {
    double x, y, z;
    double mass;
    double vx, vy, vz;
};

struct Node {
    double mass;
    double center_x, center_y, center_z;
    double min_x, min_y, min_z; 
    double max_x, max_y, max_z;
    
    int children[8]; 
    int body_index;  
    bool is_leaf;
    
    omp_lock_t lock; 
};


Node* tree;
int node_count = 0;


int allocateNode(double minX, double minY, double minZ, double maxX, double maxY, double maxZ) {
    int idx;
    #pragma omp atomic capture
    {
        idx = node_count;
        node_count++;
    }
    
    if (idx < MAX_NODES) {
        tree[idx].mass = 0;
        tree[idx].center_x = 0; tree[idx].center_y = 0; tree[idx].center_z = 0;
        tree[idx].min_x = minX; tree[idx].min_y = minY; tree[idx].min_z = minZ;
        tree[idx].max_x = maxX; tree[idx].max_y = maxY; tree[idx].max_z = maxZ;
        
        for (int i = 0; i < 8; i++) tree[idx].children[i] = -1;
        tree[idx].body_index = -1;
        tree[idx].is_leaf = true;
        omp_init_lock(&tree[idx].lock);
    }
    return idx;
}

int getOctant(Node& node, Body& b) {
    double midX = (node.min_x + node.max_x) / 2.0;
    double midY = (node.min_y + node.max_y) / 2.0;
    double midZ = (node.min_z + node.max_z) / 2.0;
    
    int octant = 0;
    if (b.x > midX) octant |= 1;
    if (b.y > midY) octant |= 2;
    if (b.z > midZ) octant |= 4;
    return octant;
}

void insertBody(int node_idx, int body_idx, Body* bodies) {
    Body& b = bodies[body_idx];
    
    while (true) {
        omp_set_lock(&tree[node_idx].lock); 
        
        if (tree[node_idx].body_index == -1 && tree[node_idx].is_leaf) {
            tree[node_idx].body_index = body_idx;
            omp_unset_lock(&tree[node_idx].lock);
            return;
        } 
        else if (tree[node_idx].is_leaf) {
            int old_body_idx = tree[node_idx].body_index;
            tree[node_idx].body_index = -1;
            tree[node_idx].is_leaf = false;
            omp_unset_lock(&tree[node_idx].lock);
            insertBody(node_idx, old_body_idx, bodies);
        } 
        else {
            int octant = getOctant(tree[node_idx], b);
            
            if (tree[node_idx].children[octant] == -1) {
                double midX = (tree[node_idx].min_x + tree[node_idx].max_x) / 2.0;
                double midY = (tree[node_idx].min_y + tree[node_idx].max_y) / 2.0;
                double midZ = (tree[node_idx].min_z + tree[node_idx].max_z) / 2.0;
                
                double nMinX = (octant & 1) ? midX : tree[node_idx].min_x;
                double nMaxX = (octant & 1) ? tree[node_idx].max_x : midX;
                double nMinY = (octant & 2) ? midY : tree[node_idx].min_y;
                double nMaxY = (octant & 2) ? tree[node_idx].max_y : midY;
                double nMinZ = (octant & 4) ? midZ : tree[node_idx].min_z;
                double nMaxZ = (octant & 4) ? tree[node_idx].max_z : midZ;
                
                int child_idx = allocateNode(nMinX, nMinY, nMinZ, nMaxX, nMaxY, nMaxZ);
                tree[node_idx].children[octant] = child_idx;
            }
            int next_node = tree[node_idx].children[octant];
            omp_unset_lock(&tree[node_idx].lock);
            node_idx = next_node;
        }
    }
}

void computeMassDistribution(int node_idx, Body* bodies) {
    if (tree[node_idx].is_leaf) {
        if (tree[node_idx].body_index != -1) {
            Body& b = bodies[tree[node_idx].body_index];
            tree[node_idx].mass = b.mass;
            tree[node_idx].center_x = b.x;
            tree[node_idx].center_y = b.y;
            tree[node_idx].center_z = b.z;
        }
        return;
    }
    
    tree[node_idx].mass = 0;
    tree[node_idx].center_x = 0; tree[node_idx].center_y = 0; tree[node_idx].center_z = 0;
    
    for (int i = 0; i < 8; i++) {
        int child_idx = tree[node_idx].children[i];
        if (child_idx != -1) {
            computeMassDistribution(child_idx, bodies); 
            
            double child_mass = tree[child_idx].mass;
            tree[node_idx].mass += child_mass;
            tree[node_idx].center_x += tree[child_idx].center_x * child_mass;
            tree[node_idx].center_y += tree[child_idx].center_y * child_mass;
            tree[node_idx].center_z += tree[child_idx].center_z * child_mass;
        }
    }
    
    if (tree[node_idx].mass > 0) {
        tree[node_idx].center_x /= tree[node_idx].mass;
        tree[node_idx].center_y /= tree[node_idx].mass;
        tree[node_idx].center_z /= tree[node_idx].mass;
    }
}

void buildOctreeOpenMP(Body* bodies, int num_bodies, double envSize) {
    tree = new Node[MAX_NODES];
    node_count = 0;
    int root = allocateNode(-envSize, -envSize, -envSize, envSize, envSize, envSize);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_bodies; i++) {
        insertBody(root, i, bodies);
    }
    
    computeMassDistribution(root, bodies);
}


__global__ void computeForcesCUDA(Body* d_bodies, Node* d_tree, int num_bodies, int root_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_bodies) {
        Body my_body = d_bodies[idx];
        double force_x = 0.0, force_y = 0.0, force_z = 0.0;
        
        int stack[64]; 
        int top = 0;
        stack[top++] = root_idx;
        
        while (top > 0) {
            int node_idx = stack[--top];
            Node n = d_tree[node_idx];
            
            double dx = n.center_x - my_body.x;
            double dy = n.center_y - my_body.y;
            double dz = n.center_z - my_body.z;
            
            double dist_sq = dx*dx + dy*dy + dz*dz + EPSILON;
            double dist = sqrt(dist_sq);
            double s = n.max_x - n.min_x;
            
            if (n.is_leaf || (s / dist < THETA)) {
                if (n.mass > 0 && n.body_index != idx) {
                    double F = (G_CONST * n.mass) / (dist_sq * dist);
                    force_x += F * dx; force_y += F * dy; force_z += F * dz;
                }
            } else {
                for (int i = 0; i < 8; i++) {
                    if (n.children[i] != -1) stack[top++] = n.children[i];
                }
            }
        }
        
        my_body.vx += force_x * DT; my_body.vy += force_y * DT; my_body.vz += force_z * DT;
        my_body.x += my_body.vx * DT; my_body.y += my_body.vy * DT; my_body.z += my_body.vz * DT;
        
        d_bodies[idx] = my_body;
    }
}


int main() {
    srand(time(NULL));
    int num_bodies = MAX_BODIES;
    double envSize = 1000.0; 
    
    Body* h_bodies = new Body[num_bodies];
    for (int i = 0; i < num_bodies; i++) {
        h_bodies[i].x = (rand() / (double)RAND_MAX) * envSize - (envSize/2.0);
        h_bodies[i].y = (rand() / (double)RAND_MAX) * envSize - (envSize/2.0);
        h_bodies[i].z = (rand() / (double)RAND_MAX) * envSize - (envSize/2.0);
        h_bodies[i].mass = ((rand() / (double)RAND_MAX) * 100.0) + 1.0;
        h_bodies[i].vx = 0.0; h_bodies[i].vy = 0.0; h_bodies[i].vz = 0.0;
    }
    
    
    double total_cpu_time = 0.0;
    float total_gpu_time = 0.0;
    
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    int steps = 10;
    std::cout << "Starting Hybrid N-Body Simulation for " << num_bodies << " particles..." << std::endl;
    
    for (int step = 0; step < steps; step++) {
        
        double start_cpu = omp_get_wtime();
        buildOctreeOpenMP(h_bodies, num_bodies, envSize);
        double end_cpu = omp_get_wtime();
        total_cpu_time += (end_cpu - start_cpu);
        
        
        cudaEventRecord(start_gpu);
        
        Body* d_bodies; Node* d_tree;
        cudaMalloc((void**)&d_bodies, num_bodies * sizeof(Body));
        cudaMalloc((void**)&d_tree, MAX_NODES * sizeof(Node));
        
        cudaMemcpy(d_bodies, h_bodies, num_bodies * sizeof(Body), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree, tree, node_count * sizeof(Node), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int numBlocks = (num_bodies + blockSize - 1) / blockSize;
        computeForcesCUDA<<<numBlocks, blockSize>>>(d_bodies, d_tree, num_bodies, 0);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_bodies, d_bodies, num_bodies * sizeof(Body), cudaMemcpyDeviceToHost);
        
        cudaFree(d_bodies); cudaFree(d_tree);
        
        cudaEventRecord(stop_gpu);
        cudaEventSynchronize(stop_gpu);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        total_gpu_time += milliseconds;
        
        if (step % 2 == 0) {
            std::cout << "Step " << step << " completed." << std::endl;
        }
        
        delete[] tree;
    }
    
    std::cout << "\n=== Simulation Performance Report ===" << std::endl;
    std::cout << "Total Iterations: " << steps << std::endl;
    std::cout << "Total CPU Time (Tree Building): " << total_cpu_time * 1000.0 << " ms" << std::endl;
    std::cout << "Total GPU Time (Force Calc + Memcpy): " << total_gpu_time << " ms" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    delete[] h_bodies;
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    return 0;
}