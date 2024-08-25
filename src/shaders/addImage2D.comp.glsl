#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, r32f) readonly uniform image2D image1;
layout(binding = 1, r32f) readonly uniform image2D image2;
layout(binding = 2, r32f) writeonly uniform image2D resultImage;

layout(push_constant) uniform PushConstants {
    int width;
    int height;
} pushConstants;

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);

    if (coords.x < pushConstants.width && coords.y < pushConstants.height) {
        float value1 = imageLoad(image1, coords).r;
        float value2 = imageLoad(image2, coords).r;
        imageStore(resultImage, coords, vec4(value1 + value2, 0.0, 0.0, 1.0));
    }
}

