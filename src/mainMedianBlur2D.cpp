#include "vcm/vcm.hpp"
#include <cmath>
#include <fmt/core.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>

// Question: uniform buffer vs push constant
struct PushConstantData {
  int width;
  int height;
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

  //   std::string filename = cv::samples::findFile("lena.jpg");
  std::string filename = "/Users/tnie/Downloads/viper.jpg";
  cv::Mat matIn;

  matIn = cv::imread(filename, cv::IMREAD_GRAYSCALE);

  uint32_t WIDTH = 512;
  uint32_t HEIGHT = 512;
  if (!matIn.empty()) {
    matIn.convertTo(matIn, CV_32F);
    matIn = matIn / 255;

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
  vk::DeviceSize bufferSize = WIDTH * HEIGHT * sizeof(float);

  vcm::VulkanBuffer inputBuf1Staging = cm.createStagingBufferSrc(bufferSize);
  vcm::VulkanBuffer inputBuf1 = cm.createDeviceBufferDst(bufferSize);

  vcm::ComputeShaderBuffers<1> buffers{};
  auto outputBufStaging = cm.createStagingBufferDst(bufferSize);
  auto outputBuf = cm.createDeviceBufferSrc(bufferSize);
  buffers.inWidth = WIDTH;
  buffers.inHeight = HEIGHT;

  buffers.in = {{{inputBuf1.ref(), inputBuf1Staging.ref()}}};
  buffers.out = {outputBuf.ref(), outputBufStaging.ref()};

  // Copy data to staging buffers

  assert(matIn.isContinuous());

  cm.copyToStagingBuffer<float>(
      std::span<float>{(float *)matIn.data, WIDTH * HEIGHT},
      buffers.in[0].staging);

  /* Create shader */
  vcm::ShaderExecutor<1, PushConstantData> shader("shaders/medianBlur2D.spv",
                                                  cm, buffers);

  auto &commandBuffer = cm.commandBuffer;

  PushConstantData pushConstant{static_cast<int>(WIDTH),
                                static_cast<int>(HEIGHT)};
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

  cv::Mat matOut(HEIGHT, WIDTH, CV_32FC1);
  cm.copyFromStagingBuffer<float>(
      buffers.out.staging,
      std::span<float>((float *)matOut.data, WIDTH * HEIGHT));

  cv::imshow("Vulkan blur", matOut);
  cv::waitKey();

  cv::Mat matCvBlur;
  {
    vcm::TimeIt<true> timeit("OpenCV blue");
    cv::medianBlur(matIn, matCvBlur, 3);
  }
  verifyOutput(matOut, matCvBlur);

  cv::imshow("CV blur", matCvBlur);
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
