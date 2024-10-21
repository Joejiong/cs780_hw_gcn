#include <iostream>
#include <vector>

// Function to perform Sparse Matrix (CSR format) x Dense Matrix multiplication
void spmm_csr(const std::vector<int>& row_ptr, 
              const std::vector<int>& col_idx, 
              const std::vector<float>& values, 
              const std::vector<std::vector<float>>& dense_matrix, 
              std::vector<std::vector<float>>& result) {
    
    int num_rows = row_ptr.size() - 1;   // Number of rows in the sparse matrix
    int num_cols = dense_matrix[0].size(); // Number of columns in the dense matrix

    // Initialize result matrix to zeros
    result.assign(num_rows, std::vector<float>(num_cols, 0.0f));

    // SpMM computation: Multiply sparse matrix with dense matrix
    for (int row = 0; row < num_rows; ++row) {
        for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
            int col = col_idx[idx];         // Column index from sparse matrix
            float value = values[idx];      // Value of the sparse matrix
            for (int j = 0; j < num_cols; ++j) {
                result[row][j] += value * dense_matrix[col][j];
            }
        }
    }
}

int main() {
    // Example sparse matrix in CSR format:
    // 1 0 0 2
    // 0 0 3 0
    // 4 0 0 5
    std::vector<int> row_ptr = {0, 2, 3, 5};
    std::vector<int> col_idx = {0, 3, 2, 0, 3};
    std::vector<float> values = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Dense matrix to multiply (4x2 matrix)
    std::vector<std::vector<float>> dense_matrix = {
        {1.0, 2.0},
        {0.0, 0.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };

    // Result matrix (initialized as empty)
    std::vector<std::vector<float>> result;

    // Perform SpMM
    spmm_csr(row_ptr, col_idx, values, dense_matrix, result);

    // Print the result
    std::cout << "Result of SpMM:\n";
    for (const auto& row : result) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    return 0;
}