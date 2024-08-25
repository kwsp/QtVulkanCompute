#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) readonly buffer InputImage1 {
    float image1[];
};
layout(binding = 1) readonly buffer InputImage2 {
    float image2[];
};
layout(binding = 2) writeonly buffer OutputImage {
    float result[];
};

layout(push_constant) uniform PushConstants {
    int width;
    int height;
} pushConstants;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint index = y * pushConstants.width + x;

    if (x < pushConstants.width && y < pushConstants.height) {
        // result[index] = image1[index] + image2[index];
        float v1 = image1[index];
        float v2 = image2[index];

        result[index] = v1 + v2;
    }
}
