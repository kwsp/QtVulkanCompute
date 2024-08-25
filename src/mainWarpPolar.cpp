#include "vcm/vcm.hpp"
#include <cmath>
#include <fmt/core.h>
#include <glm/glm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>

// Question: uniform buffer vs push constant

struct PushConstantData {
  glm::vec2 center; // Center of the polar transform
  float maxRadius;  // Maximum radius for the transformation
  int width;        // Width of the input image
  int height;       // Height of the input image
};

bool verifyOutput(const cv::Mat &mat, const cv::Mat &expect) {
  for (int row = 0; mat.rows; ++row) {
    for (int col = 0; mat.cols; ++col) {
      if (mat.at<float>(col, row) != expect.at<float>(col, row)) {
        fmt::println("Mismatch at ({}, {}); expected {}, but got {}", col, row,
                     expect.at<float>(col, row), mat.at<float>(col, row));
        return false;
      }
    }
  }
  return true;
}

int main() {

  std::string filename = "/Users/tnie/Downloads/stripes.jpg";
  cv::Mat matIn;

  matIn = cv::imread(filename, cv::IMREAD_GRAYSCALE);

  uint32_t WIDTH = 512;
  uint32_t HEIGHT = 512;
  if (!matIn.empty()) {
    matIn.convertTo(matIn, CV_32F);
    matIn = matIn / 255;

    cv::resize(matIn, matIn, {1000, 2000});

    WIDTH = matIn.cols;
    HEIGHT = matIn.rows;

  } else {
    matIn = cv::Mat(HEIGHT, WIDTH, CV_32FC1);
    cv::randu(matIn, 0.0F, 0.1F);
  }
  cv::imshow("original", matIn);
  cv::waitKey();

  vcm::VulkanComputeManager cm;

  /*
  Create buffers
  */
  const int r = std::min(WIDTH, HEIGHT);

  PushConstantData pushConstant{};
  pushConstant.width = r;
  pushConstant.height = r;
  glm::vec2 center = {r / 2, r / 2};
  pushConstant.center = center;
  pushConstant.maxRadius = std::min(center.x, center.y);

  vk::DeviceSize inBufferSize = WIDTH * HEIGHT * sizeof(float);
  vk::DeviceSize outBufSize = r * r * sizeof(float);

  vcm::VulkanBuffer inputBuf1Staging = cm.createStagingBufferSrc(inBufferSize);
  vcm::VulkanBuffer inputBuf1 = cm.createDeviceBufferDst(inBufferSize);

  vcm::ComputeShaderBuffers<1> buffers{};
  auto outputBufStaging = cm.createStagingBufferDst(outBufSize);
  auto outputBuf = cm.createDeviceBufferSrc(outBufSize);
  buffers.inWidth = WIDTH;
  buffers.inHeight = HEIGHT;

  buffers.outWidth = r;
  buffers.outHeight = r;

  buffers.in = {{{inputBuf1.ref(), inputBuf1Staging.ref()}}};
  buffers.out = {outputBuf.ref(), outputBufStaging.ref()};

  // Copy data to staging buffers

  assert(matIn.isContinuous());

  cm.copyToStagingBuffer<float>(
      std::span<float>{(float *)matIn.data, WIDTH * HEIGHT},
      buffers.in[0].staging);

  /* Create shader */
  vcm::ShaderExecutor<1, PushConstantData> shader(
      "shaders/warpPolarInverse.spv", cm, buffers);

  auto &commandBuffer = cm.commandBuffer;

  shader.recordCommandBuffer(cm, commandBuffer, buffers, pushConstant);

  // Submit the command buffer to the compute queue
  {
    vcm::TimeIt<true> timeit("Wait queue");

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    cm.queue.submit(submitInfo);

    cm.queue.waitIdle();
  }

  cv::Mat matOut(r, r, CV_32FC1);
  cm.copyFromStagingBuffer<float>(
      buffers.out.staging, std::span<float>((float *)matOut.data, r * r));

  cv::imshow("Vulkan warp", matOut);
  cv::waitKey();

  cv::Mat matCvOut;
  {
    vcm::TimeIt<true> timeit("OpenCV warp");

    cv::warpPolar(matIn, matCvOut, {r, r}, {(float)r / 2, (float)r / 2},
                  pushConstant.maxRadius,
                  cv::WARP_INVERSE_MAP | cv::WARP_FILL_OUTLIERS);
  }
  verifyOutput(matOut, matCvOut);

  cv::imshow("CV warp", matCvOut);
  cv::waitKey();

  //   {
  //     bool isCorrect = verifyOutput(outputData, 7.0F);
  //     if (isCorrect) {
  //       fmt::println("The output is correct!");
  //     } else {
  //       fmt::println("The output is incorrect.");
  //     }
  //   }

  //   std::vector<float> outputCPU(WIDTH * HEIGHT);
  //   {
  //     vcm::TimeIt<true> timeit("CPU version");
  //     for (int i = 0; i < WIDTH * HEIGHT; ++i) {
  //       outputCPU[i] = std::fmaf(input1[i], input2[i], input3[i]);
  //     }
  //   }

  //   // Verify results
  //   {
  //     bool isCorrect = verifyOutput(outputCPU, 7.0F);
  //     if (isCorrect) {
  //       fmt::println("The output is correct!");
  //     } else {
  //       fmt::println("The output is incorrect.");
  //     }
  //   }

  return 0;
}
