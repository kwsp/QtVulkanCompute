#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0) uniform sampler2D inputImage;
layout (binding = 1, rgba8)  writeonly uniform image2D outputImage;

layout(push_constant) uniform PushConstants {
    vec2 center;
    float maxRadius;
    bool logScale;
} ubo;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 normCoord = (vec2(coord) - ubo.center) / ubo.maxRadius;

    // Compute the radius and angle for the polar coordinates
    float radius = length(normCoord);
    float angle = atan(normCoord.y, normCoord.x);

    // Normalize the angle to [0, 1]
    angle = angle / (2.0 * 3.141592653589793) + 0.5;

    // Apply logarithmic scaling
    if (ubo.logScale) {
        radius = log(1.0 + radius) / log(1.0 + ubo.maxRadius);
    }

    // Map the polar coordinates back to the source image coordinates
    vec2 polarCoord = vec2(radius, angle);

    // Convert back to Cartesian coordinates for sampling
    vec2 sourceCoord = vec2(
        polarCoord.x * cos(polarCoord.y * 2.0 * 3.141592653589793),
        polarCoord.x * sin(polarCoord.y * 2.0 * 3.141592653589793)
    );

    sourceCoord = ubo.center + sourceCoord * ubo.maxRadius;

    // Fetch color from inputImage and write to outputImage
    vec4 color = texture(inputImage, sourceCoord / vec2(textureSize(inputImage, 0)));
    imageStore(outputImage, coord, color);
}