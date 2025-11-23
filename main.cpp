#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp> // 用于创建文件夹
#include <iostream>
#include <vector>

// 辅助函数：将 BGR 转为 YUYV (YUY2) 的裸数据 std::vector<uint8_t>
// 内存布局：Y0 U0 Y1 V0 ...
std::vector<uint8_t> GetRawYUYVData(const cv::Mat& bgr, int& out_w, int& out_h) {
    int W = bgr.cols;
    int H = bgr.rows;
    
    // 强制宽度为偶数，因为 YUYV 必须成对出现
    if (W % 2 != 0) W--; 

    out_w = W;
    out_h = H;

    // 一个像素2个字节 (CV_8UC2)
    std::vector<uint8_t> buffer(W * H * 2); 
    
    int idx = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; x += 2) {
            // 获取两个像素 BGR
            cv::Vec3b p0 = bgr.at<cv::Vec3b>(y, x);
            cv::Vec3b p1 = bgr.at<cv::Vec3b>(y, x + 1);

            // 简单的 YUV 转换公式 (BT.601)
            auto RGB2Y = [](int r, int g, int b) { return static_cast<uint8_t>((66 * r + 129 * g + 25 * b + 128) >> 8) + 16; };
            auto RGB2U = [](int r, int g, int b) { return static_cast<uint8_t>((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128; };
            auto RGB2V = [](int r, int g, int b) { return static_cast<uint8_t>((112 * r - 94 * g - 18 * b + 128) >> 8) + 128; };

            uint8_t Y0 = RGB2Y(p0[2], p0[1], p0[0]);
            uint8_t U0 = RGB2U(p0[2], p0[1], p0[0]);
            uint8_t V0 = RGB2V(p0[2], p0[1], p0[0]);

            uint8_t Y1 = RGB2Y(p1[2], p1[1], p1[0]);
            uint8_t U1 = RGB2U(p1[2], p1[1], p1[0]);
            uint8_t V1 = RGB2V(p1[2], p1[1], p1[0]);

            // 采样：U 和 V 取两个像素的平均值
            uint8_t U = (U0 + U1) / 2;
            uint8_t V = (V0 + V1) / 2;

            // 填入内存：顺序是 Y0, U, Y1, V
            buffer[idx++] = Y0; // Byte 0: Pixel 0, Channel 0 (Y)
            buffer[idx++] = U;  // Byte 1: Pixel 0, Channel 1 (U)
            buffer[idx++] = Y1; // Byte 2: Pixel 1, Channel 0 (Y)
            buffer[idx++] = V;  // Byte 3: Pixel 1, Channel 1 (V)
        }
    }
    return buffer;
}

int main() {
    // 1. 准备目录
    std::string outDir = "output";
    if (!cv::utils::fs::exists(outDir)) {
        cv::utils::fs::createDirectory(outDir);
    }

    // 2. 读取原始图片
    std::string imgPath = "image/DSC_0822.JPG"; // 请修改为你的图片路径
    cv::Mat bgr = cv::imread(imgPath);
    if (bgr.empty()) {
        std::cerr << "错误：无法读取图片 " << imgPath << std::endl;
        // 如果没有图片，创建一个随机图用于测试
        std::cout << "生成一张 3000x3000 的测试图..." << std::endl;
        bgr = cv::Mat(3000, 3000, CV_8UC3);
        cv::randu(bgr, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        // 画个圆圈方便定位
        cv::circle(bgr, cv::Point(1500, 1500), 100, cv::Scalar(0, 0, 255), -1);
    }

    // 确保图片足够大，能容纳裁切框 (x=1001 + w=1001 = 2002)
    if (bgr.cols < 2100 || bgr.rows < 2100) {
        cv::resize(bgr, bgr, cv::Size(2500, 2500));
    }

    // 3. 转换为 YUYV 裸数据
    std::cout << "正在构建 YUYV 裸数据流..." << std::endl;
    int w, h;
    std::vector<uint8_t> rawData = GetRawYUYVData(bgr, w, h);

    // 4. 将裸数据封装为 CV_8UC2 Mat
    // 此时 src 仅仅是对 rawData 的引用，没有发生拷贝
    cv::Mat src(h, w, CV_8UC2, rawData.data());
    std::cout << "构建 src 完成: " << src.cols << "x" << src.rows << ", Type=" << src.type() << std::endl;

    // 定义裁切框
    cv::Rect roi_odd(2001, 2001, 1001, 1001); // 奇数 x, 奇数 y, 奇数 w
    cv::Rect roi_even(2000, 2000, 1000, 1000); // 偶数 x, 偶数 y, 偶数 w

    // ==========================================
    // 测试 1: 偶数裁切 (标准的做法)
    // ==========================================
    std::cout << "\n--- 测试 1: 偶数裁切 (1000, 1000) ---" << std::endl;
    cv::Mat crop_even = src(roi_even);
    cv::Mat gray_even;
    
    // 提取 Y 通道 (转灰度)
    cv::cvtColor(crop_even, gray_even, cv::COLOR_YUV2GRAY_YUY2);
    
    std::string path_even = outDir + "/result_even_crop.jpg";
    cv::imwrite(path_even, gray_even);
    std::cout << "偶数裁切保存成功: " << path_even << std::endl;

    // ==========================================
    // 测试 2: 奇数裁切 (你的疑问点)
    // ==========================================
    std::cout << "\n--- 测试 2: 奇数裁切 (1001, 1001) ---" << std::endl;
    // 这里裁切的内存指针指向了 第1001个像素 (Byte偏移 2002)
    // 该位置的数据原本是 [Y_1001, V_1000]
    cv::Mat crop_odd = src(roi_odd);
    cv::Mat gray_odd;

    // 提取 Y 通道 (转灰度)
    // 关键点：OpenCV 只是简单地取每个 Vec2b 的第一个分量
    // 对于 crop_odd 的第0个元素，它的第一个分量就是 Y_1001
    // 结果应该是完全正确的灰度图
    try {
        cv::cvtColor(crop_odd, gray_odd, cv::COLOR_YUV2GRAY_YUY2);
        
        std::string path_odd = outDir + "/result_odd_crop.jpg";
        cv::imwrite(path_odd, gray_odd);
        std::cout << "奇数裁切保存成功: " << path_odd << std::endl;
        std::cout << "结果验证：并没有报错，图像已生成。" << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "奇数裁切失败! " << e.what() << std::endl;
    }

    // ==========================================
    // 额外测试: 验证颜色是否错乱 (预期：奇数裁切颜色会乱)
    // ==========================================
    std::cout << "\n--- 额外测试: 转彩色对比 ---" << std::endl;
    cv::Mat color_odd;
    // 奇数裁切转彩色，UV顺序会反，颜色会变成诡异的色调
    cv::cvtColor(crop_odd, color_odd, cv::COLOR_YUV2BGR_YUY2); 
    cv::imwrite(outDir + "/result_odd_crop_wrong_color.jpg", color_odd);
    std::cout << "奇数裁切转彩色已保存 (请观察颜色是否异常): " << outDir << "/result_odd_crop_wrong_color.jpg" << std::endl;
    

    // ==========================================
    // 测试 3: 错误的格式封装 (模拟将 YUYV 当作 CV_8UC1/YUV420 处理)
    // ==========================================
    std::cout << "\n--- 测试 3: 错误格式封装 (强制 CV_8UC1) ---" << std::endl;

    // 1. 错误的封装
    // 原始数据是 YUYV (w * h * 2 字节)
    // 我们强行告诉 OpenCV 这是一个 w * h 的单通道图像 (w * h * 1 字节)
    // 也就是我们只“看到”了内存的前一半，且每一行的步长(step)都错了
    cv::Mat wrong_format_mat(h, w, CV_8UC1, rawData.data());

    // 2. 施加裁切框
    // 这里的 roi_odd (1001, 1001, 1001, 1001)
    // 对于 CV_8UC1 来说，x=1001 意味着向后偏移 1001 个字节
    // 而在 YUYV 逻辑里，x=1001 应该是偏移 2002 个字节
    // 所以这里的裁切位置是完全错误的
    cv::Mat crop_wrong = wrong_format_mat(roi_odd);

    // 3. 深拷贝副本
    // 此时会将错误的内存布局固化下来
    cv::Mat crop_wrong_copy = crop_wrong.clone();

    // 4. 保存结果
    std::string path_garbage = outDir + "/result_wrong_format_garbage.jpg";
    cv::imwrite(path_garbage, crop_wrong_copy);
    
    std::cout << "错误格式图片已保存: " << path_garbage << std::endl;
    std::cout << "请打开查看，这张图应该是严重的'花屏' (错位+纹理混乱)。" << std::endl;

    // ==========================================
    // 额外分析: 为什么会这样?
    // ==========================================
    // 原图一行有 w 个像素，占 2w 个字节。
    // wrong_format_mat 认为一行只有 w 个字节。
    // 结果：原图的 "第1行" 数据，被 wrong_format_mat 拆成了 "第1行" 和 "第2行"。
    // 图像看起来会被压扁，且右半部分接在左半部分下面，完全乱套。

    return 0;
}