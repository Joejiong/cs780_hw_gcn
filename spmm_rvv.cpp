#include <vector>
#include <riscv_vector.h>

// Optimized SpMM using RISC-V Vector Extension (RVV)
void spmm_rvv(const std::vector<int>& row_ptr, 
              const std::vector<int>& col_idx, 
              const std::vector<float>& values, 
              const float* dense_matrix, // Assuming 1D flattened array for dense matrix
              float* result,             // Assuming 1D flattened array for result
              int dense_cols) {          // Number of columns in the dense matrix
    
    int num_rows = row_ptr.size() - 1;   // Number of rows in the sparse matrix
    
    for (int row = 0; row < num_rows; ++row) {
        // For each non-zero value in the row
        for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
            int col = col_idx[idx];          // Column index from sparse matrix
            float sparse_value = values[idx]; // Sparse matrix value
            
            // Process dense matrix row (vectorized)
            for (int j = 0; j < dense_cols;) {
                // Set the vector length based on remaining columns
                size_t vl = vsetvl_e32m1(dense_cols - j);  // Set vector length for 32-bit floats

                // Load dense matrix row elements (column-wise)
                vfloat32m1_t dense_vec = vle32_v_f32m1(&dense_matrix[col * dense_cols + j], vl);

                // Load corresponding result row elements
                vfloat32m1_t result_vec = vle32_v_f32m1(&result[row * dense_cols + j], vl);
                
                // Perform fused multiply-add (result_vec += sparse_value * dense_vec)
                result_vec = vfmacc_vf_f32m1(result_vec, sparse_value, dense_vec, vl);
                
                // Store the updated result back to memory
                vse32_v_f32m1(&result[row * dense_cols + j], result_vec, vl);

                // Move to the next chunk of columns
                j += vl;
            }
        }
    }
}