#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "UF.h"     // Weighted-Union-Find algorithm
#include "color.h"

using namespace cv;
using namespace std;

void image_gaussian_blur(Mat& mat, double sigma);

Mat build_graph(Mat& mat);

UF segment_graph(Mat& mat, const int nodes, const int k);

Mat sort_graph(const Mat& mat);

void process_small_components(UF& uf, Mat& mat, int min_size);

int main() {
    // Global const params
    const double sigma = 0.8;     // Gaussian Filter sigma
    const int K = 300;            // const
    const int min_size = 1500;    // min component size

    // Load image
    Mat im = imread("/home/kuang/beauty.jpg");

    // Resize image small
    resize(im, im, im.size() / 2); // resize to 1/4 size
    //imshow("im", im);

    // Get image size
    const int H = im.rows;
    const int W = im.cols;
    const int num_nodes = H * W;

    // Gaussian blur
    image_gaussian_blur(im, sigma);
    imshow("im2", im);

    // Convert to float type (Not necessary)
    // im.convertTo(im, CV_32FC3);
    // cout << im.rows << endl << im.cols;

    // Build graph
    Mat graph = build_graph(im);

    // Segment
    UF uf = segment_graph(graph, num_nodes, K);

    // Post process small components
    process_small_components(uf, graph, min_size);

    // Plot result image
    vector<RGB> color_map(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        color_map[i].r = (unsigned char) random();
        color_map[i].g = (unsigned char) random();
        color_map[i].b = (unsigned char) random();
    }

    Mat ret_img = Mat::zeros(im.size(), im.type());
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int id = uf.find_id(h * W + w);
            ret_img.at<Vec3b>(h, w)[0] = color_map[id].b;
            ret_img.at<Vec3b>(h, w)[1] = color_map[id].g;
            ret_img.at<Vec3b>(h, w)[2] = color_map[id].r;
        }
    }
    imshow("ret", ret_img);

    waitKey();

    return 0;
}

void process_small_components(UF& uf, Mat& graph, int min_size) {
    int edge_num = uf.id.size();

    for (int i = 0; i < edge_num; i++) {
        vector<float> edge = graph.row(i);
        int a = uf.find_id(edge[0]);
        int b = uf.find_id(edge[1]);
        if ((a != b) && ((uf.sz[a] < min_size) || (uf.sz[b] < min_size)))
            uf.union_two(a, b);
    }
}

UF segment_graph(Mat& graph, const int num_nodes, const int K) {
// Segment graph
//     Inputs:
//          graph: [num_edges, 3] matrix, each row = [idxA, idxB, weight]
//          num_nodes: number of pixels
//          K: a const parameter
//     Outputs:
//          UF: Union-Find data-structure representing graph structure

    int num_edges = graph.rows;

    UF uf(num_nodes);
    vector<float> threshold(num_nodes, K); // init all threshold = K

    // Sort graph by weights ascending
    Mat sorted_graph = sort_graph(graph);

    cout << "segmenting..." << endl;
    for (int i = 0; i < num_edges; ++i) {
//        if (i % 5000 == 0)
//            cout << i << "/" << num_edges << endl;

        // get one edge, convert to one vector/Vec3f that we can use [], amazing!
        vector<float> edge = sorted_graph.row(i);
        float weight = edge[2];

        // get edge's end-point id
        int parent_a = uf.find_id(edge[0]);
        int parent_b = uf.find_id(edge[1]);

        // compare the weight with threshold
        bool condition_a = weight <= threshold[parent_a];
        bool condition_b = weight <= threshold[parent_b];

        if ((parent_a != parent_b) && condition_a && condition_b) {
            // different components & disjoint diff < internal diff
            uf.union_two(parent_a, parent_b);
            int parent_new = uf.find_id(parent_a);
            threshold[parent_new] = weight + double(K) / uf.sz[parent_new];
        }
        //cout << parent_a << " " << parent_b << " " << weight;
    }
    cout << "done!" << endl;
    return uf;
}

Mat sort_graph(const Mat& graph) {
// Sort graph according to 3rd column weights ascending
//  return sorted graph

    Mat col = graph.col(2); // get 3rd column weights
    Mat1i idx;
    sortIdx(col, idx, SORT_EVERY_COLUMN | SORT_ASCENDING);

    Mat dst = Mat::zeros(graph.size(), graph.type());
    for (int i = 0; i < graph.rows; ++i) {
        graph.row(idx(0, i)).copyTo(dst.row(i));
    }

    return dst;
}

Mat build_graph(Mat& im) {
// Build graph on image
//  Input:
//     im:  RGB image [H*W*3]
//  Output:
//     graph: [edge_num, 3] each row is [pixelA_idx, pixelB_idx, AB_diff]


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

void image_gaussian_blur(Mat& mat, double sigma) {
// Smooth image use Guassian filter
//  blur image before build the graph is very important!
//  cause it will prevent lots of 0 weights
    
    int alpha = 4;  // parameter to control kernel size
    sigma = max(sigma, 0.01);
    int k_size = ceil(sigma * alpha) + 1;

    GaussianBlur(mat, mat, Size(k_size, k_size), sigma, sigma);
}