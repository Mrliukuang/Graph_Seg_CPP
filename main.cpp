#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "UF.h"     // Weighted-Union-Find algorithm

using namespace cv;
using namespace std;

void image_guassian_blur(Mat& mat, double sigma);

Mat build_graph(Mat& mat);

int main() {
    // Global const params
    const double sigma = 0.8;     // Gaussian Filter sigma
    const int K = 300;            // const
    const int min_size = 4000;    // min component size

    // Load image
    Mat im = imread("/home/kuang/beauty.jpg");

    // Resize image small
    resize(im, im, im.size() / 2); // resize to 1/4 size
    //imshow("im", im);

    // Gaussian blur
    image_guassian_blur(im, sigma);
    //imshow("im2", im);

    // Convert to float type (Not necessary)
    // im.convertTo(im, CV_32FC3);
    // cout << im.rows << endl << im.cols;

    // Build graph
    Mat graph = build_graph(im);

    cout << graph.size();

    waitKey();


    return 0;
}

Mat build_graph(Mat& im) {
    /* Build graph on image
     *  Input:
     *    im:  RGB image [H*W*3]
     *  Output:
     *    graph: [edge_num, 3] each row is [pixelA_idx, pixelB_idx, AB_diff]
     */

    // get image size
    int W = im.cols;
    int H = im.rows;

    int edge_num = 2 * (H - 1) * (W - 1) + (H - 1) * W + (W - 1) * H;
    Mat graph = Mat::zeros(edge_num, 3, CV_32F);
    cout << "building graph..." << endl;

    // iterate all pixels to build the graph
    // TODO: replace norm with native function to speed up
    int cnt = 1;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx = h * W + w;              // current pixel index
            Vec3f pixel = im.at<Vec3b>(h, w); // current pixel color, convert to float to prevent overflow

            if (w < W - 1) {    // has right point
                Vec3f next = im.at<Vec3b>(h, w + 1);            // next pixel in this case right point pixel
                graph.at<float>(cnt, 0) = idx;                  // pixel_current_idx
                graph.at<float>(cnt, 1) = idx + 1;              // pixel_next_idx
                graph.at<float>(cnt, 2) = norm(pixel - next);   // weight: Euclidean distance
                cnt++;
            }

            if (h < H - 1) {    // has down point
                Vec3f next = im.at<Vec3b>(h + 1, w);
                graph.at<float>(cnt, 0) = idx;
                graph.at<float>(cnt, 1) = idx + W;
                graph.at<float>(cnt, 2) = norm(pixel - next);
                cnt++;
            }

            if ((w < W - 1) && (h < H - 1)) { // has right down point
                Vec3f next = im.at<Vec3b>(h + 1, w + 1);
                graph.at<float>(cnt, 0) = idx;
                graph.at<float>(cnt, 1) = idx + 1 + W;
                graph.at<float>(cnt, 2) = norm(pixel - next);
                cnt++;
            }

            if ((w < W - 1) && (h > 0)) { // has right up point
                Vec3f next = im.at<Vec3b>(h - 1, w + 1);
                graph.at<float>(cnt, 0) = idx;
                graph.at<float>(cnt, 1) = idx + 1 - W;
                graph.at<float>(cnt, 2) = norm(pixel - next);
                cnt++;
            }
        }
    }
    cout << "done!" << endl;
    return graph;
}

void image_guassian_blur(Mat& mat, double sigma) {
    /* Smooth image use Guassian filter
     * blur image before build the graph is very important!
     * cause it will prevent lots of 0 weights
     */

    int alpha = 4;  // parameter to control kernel size
    sigma = max(sigma, 0.01);
    int k_size = ceil(sigma * alpha) + 1;

    GaussianBlur(mat, mat, Size(k_size, k_size), sigma, sigma);
}