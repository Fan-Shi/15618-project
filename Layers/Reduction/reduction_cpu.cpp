#include <vector>
#include <omp.h>
#include <iostream>

double runExperiment(int totalElement) {
    std::vector<float> input(totalElement, 1.0f);
    float ret = 0.0f;
    double dTime;
    dTime = omp_get_wtime();
    for(float val : input) {
        ret += val;
    }
    double reductionTime = omp_get_wtime() - dTime;
}

int main() {
    double time = runExperiment(3145728);
    std::cout << "Reduction on cpu take " << time << " to complete." << std::endl;
}