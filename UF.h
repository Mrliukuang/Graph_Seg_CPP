//
// Created by kuang on 15-7-1.
//

#ifndef UF_H
#define UF_H

#include <vector>

using namespace std;

class UF {
// Weighted-Union-Find algorithm

public:
    vector<int> id;     // each component id
    vector<int> sz;     // each component size
    int count;          // component number

public:
    UF(int N) {
        count = N;
        for (int i = 0; i < N; ++i) {
            id.push_back(i);
            sz.push_back(1);
        }
    }

    int get_count() { return count; }

    bool connected(int p, int q) {
        return find_id(p) == find_id(q);
    }

    int find_id(int p) {
        while (p != id[p]) p = id[id[p]];   // path compression
        return p;
    }

    void union_two(int p, int q) { // union two components
        int i = find_id(p);
        int j = find_id(q);
        if (i == j) return;
        if (sz[i] < sz[j]) {
            id[i] = j;
            sz[j] += sz[i];
        } else {
            id[j] = i;
            sz[i] += sz[j];
        }
        count--;
    }
};


#endif //UF_H