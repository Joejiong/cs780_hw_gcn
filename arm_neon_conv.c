#include <stdio.h>
#include <arm_neon.h>

// Convolution kernel
void neon_convolve_3x3(const int32_t* image, int32_t* output, const int32_t* kernel, int width, int height) {
    for (int i = 0; i < height - 2; i++) {
        for (int j = 0; j < width - 2; j += 4) { // Process 4 elements at a time
            int32x4_t row1 = vld1q_s32(&image[(i * width) + j]);
            int32x4_t row2 = vld1q_s32(&image[((i + 1) * width) + j]);
            int32x4_t row3 = vld1q_s32(&image[((i + 2) * width) + j]);

            // Load kernel values
            int32_t k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
            int32_t k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
            int32_t k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];

            // Element-wise multiply and accumulate
            int32x4_t sum = vmulq_n_s32(row1, k00);
            sum = vmlaq_n_s32(sum, row1, k01);
            sum = vmlaq_n_s32(sum, row1, k02);

            sum = vmlaq_n_s32(sum, row2, k10);
            sum = vmlaq_n_s32(sum, row2, k11);
            sum = vmlaq_n_s32(sum, row2, k12);

            sum = vmlaq_n_s32(sum, row3, k20);
            sum = vmlaq_n_s32(sum, row3, k21);
            sum = vmlaq_n_s32(sum, row3, k22);

            // Store result
            vst1q_s32(&output[i * (width - 2) + j], sum);
        }
    }
}

// Helper to print the result
void print_matrix(const int32_t* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%4d ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

// Test the convolution with different cases
int main() {
    // Test case 1: Simple 3x3 Image with Identity Kernel
    int32_t image1[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    int32_t identity_kernel[9] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };
    int32_t output1[1];
    neon_convolve_3x3(image1, output1, identity_kernel, 3, 3);
    printf("Test Case 1: Identity Kernel Output:\n");
    print_matrix(output1, 1, 1);

    // Test case 2: 5x5 Image with All-Ones Kernel
    int32_t image2[25] = {
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    };
    int32_t ones_kernel[9] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };
    int32_t output2[9];
    neon_convolve_3x3(image2, output2, ones_kernel, 5, 5);
    printf("Test Case 2: All-Ones Kernel Output:\n");
    print_matrix(output2, 3, 3);

    // Test case 3: 5x5 Image with Edge Detection Kernel
    int32_t image3[25] = {
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1
    };
    int32_t edge_kernel[9] = {
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1
    };
    int32_t output3[9];
    neon_convolve_3x3(image3, output3, edge_kernel, 5, 5);
    printf("Test Case 3: Edge Detection Kernel Output:\n");
    print_matrix(output3, 3, 3);

    return 0;
}