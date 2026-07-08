#include <iostream>                                  //g++ .\matrix.cpp -o .\main -O2 -fopenmp
#include <vector>
#include <omp.h>
#include <cstdlib>

using namespace std;

int main() {
    
    int n = 1000; 
    int m = 1200; 
    int k = 800;  

    cout << "Matrix Multiplication Project (OpenMP)" << endl;
    cout << "A[" << n << "][" << m << "] * B[" << m << "][" << k << "] = C[" << n << "][" << k << "]\n";
    cout << "Allocating memory..." << endl;

    
    vector<double> A(n * m);
    vector<double> B(m * k);
    vector<double> C(n * k, 0.0);

    
    for (int i = 0; i < n * m; i++) A[i] = rand() % 100 / 10.0;
    for (int i = 0; i < m * k; i++) B[i] = rand() % 100 / 10.0;

    int max_threads = 4;
    cout << "------------------------------------------\n";
    cout << "Starting OpenMP Test...\n";

    
    for (int t = 1; t <= max_threads; t++) {
        omp_set_num_threads(t);
        double start_time = omp_get_wtime();

        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                
                
                double sum = 0.0;
                
                
                for (int l = 0; l < m; l++) {
                    
                    sum += A[i * m + l] * B[l * k + j];
                }
                C[i * k + j] = sum;
            }
        }

        double end_time = omp_get_wtime();
        cout << "Threads: " << t << " \t| Execution Time: " << end_time - start_time << " seconds" << endl;
    }
    
    cout << "------------------------------------------\n";

    return 0;
}