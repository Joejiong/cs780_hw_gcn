#include <stdio.h>
#include <arm_neon.h>

void conv2d_nhwc_neon(float *input, float *output, float *kernel, int N, int H, int W, int C) {
    // Iterate over each image in the batch
    for (int n = 0; n < N; ++n) {
        // Iterate over the height (H) and width (W) of the image
        for (int h = 1; h < H - 1; ++h) { // Avoid borders (no padding)
            for (int w = 1; w < W - 1; ++w) {
                // Prepare a vector to accumulate results (initialize to zero)
                float32x4_t sum_vec = vmovq_n_f32(0.0f);

                // Load and apply 3x3 convolution kernel across the channels (C)
                for (int c = 0; c < C; c += 4) {
                    // Load input data for the 3x3 region, using NEON intrinsics
                    float32x4_t input_vec0 = vld1q_f32(&input[n * H * W * C + (h - 1) * W * C + (w - 1) * C + c]);
                    float32x4_t input_vec1 = vld1q_f32(&input[n * H * W * C + (h - 1) * W * C + (w) * C + c]);
                    float32x4_t input_vec2 = vld1q_f32(&input[n * H * W * C + (h - 1) * W * C + (w + 1) * C + c]);

                    float32x4_t input_vec3 = vld1q_f32(&input[n * H * W * C + (h) * W * C + (w - 1) * C + c]);
                    float32x4_t input_vec4 = vld1q_f32(&input[n * H * W * C + (h) * W * C + (w) * C + c]);
                    float32x4_t input_vec5 = vld1q_f32(&input[n * H * W * C + (h) * W * C + (w + 1) * C + c]);

                    float32x4_t input_vec6 = vld1q_f32(&input[n * H * W * C + (h + 1) * W * C + (w - 1) * C + c]);
                    float32x4_t input_vec7 = vld1q_f32(&input[n * H * W * C + (h + 1) * W * C + (w) * C + c]);
                    float32x4_t input_vec8 = vld1q_f32(&input[n * H * W * C + (h + 1) * W * C + (w + 1) * C + c]);

                    // Load the 3x3 kernel values
                    float32x4_t kernel_vec0 = vld1q_f32(&kernel[0 * C + c]);
                    float32x4_t kernel_vec1 = vld1q_f32(&kernel[1 * C + c]);
                    float32x4_t kernel_vec2 = vld1q_f32(&kernel[2 * C + c]);

                    // Multiply and accumulate the results
                    sum_vec = vmlaq_f32(sum_vec, input_vec0, kernel_vec0);
                    sum_vec = vmlaq_f32(sum_vec, input_vec1, kernel_vec0);
                    sum_vec = vmlaq_f32(sum_vec, input_vec2, kernel_vec0);

                    sum_vec = vmlaq_f32(sum_vec, input_vec3, kernel_vec1);
                    sum_vec = vmlaq_f32(sum_vec, input_vec4, kernel_vec1);
                    sum_vec = vmlaq_f32(sum_vec, input_vec5, kernel_vec1);

                    sum_vec = vmlaq_f32(sum_vec, input_vec6, kernel_vec2);
                    sum_vec = vmlaq_f32(sum_vec, input_vec7, kernel_vec2);
                    sum_vec = vmlaq_f32(sum_vec, input_vec8, kernel_vec2);
                }

                // Store the result into the output tensor
                vst1q_f32(&output[n * (H - 2) * (W - 2) * C + (h - 1) * (W - 2) * C + (w - 1) * C], sum_vec);
            }
        }
    }
}

int main() {
    // Example dimensions (N = batch size, H = height, W = width, C = channels)
    int N = 1, H = 5, W = 5, C = 4; // Simple example (1 image, 5x5 size, 4 channels)

    // Example input image (NHWC format)
    float input[5 * 5 * 4] = {
        // Image 1 (5x5x4) flattened
        1, 1, 1, 1,  1, 2, 1, 1,  1, 3, 1, 1,  1, 4, 1, 1,  1, 5, 1, 1,
        1, 1, 1, 1,  2, 1, 1, 1,  2, 2, 1, 1,  2, 3, 1, 1,  2, 4, 1, 1,
        1, 1, 1, 1,  3, 1, 1, 1,  3, 2, 1, 1,  3, 3, 1, 1,  3, 4, 1, 1,
        1, 1, 1, 1,  4, 1, 1, 1,  4, 2, 1, 1,  4, 3, 1, 1,  4, 4, 1, 1,
        1, 1, 1, 1,  5, 1, 1, 1,  5, 2, 1, 1,  5, 3, 1, 1,  5, 4, 1, 1,
    };

    // Example 3x3 kernel
    float kernel[3 * 3 * 4] = {
        // Kernel for 4 channels
        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  // Row 1
        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  // Row 2
        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  // Row 3
    };

    // Output tensor (smaller than input due to no padding and kernel size)
    float output[3 * 3 * 4]; // Output for 3x3 with 4 channels

    // Perform convolution
    conv2d_nhwc_neon(input, output, kernel, N, H, W, C);

    // Print the output
    for (int i = 0; i < 3 * 3 * 4; ++i) {
        if (i % 4 == 0) printf("\n");
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}