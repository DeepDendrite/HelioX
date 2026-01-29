#ifndef permute_order_h
#define permute_order_h

#include <iostream>
#include <vector>

class PermuteInfo;
class TNode;

typedef std::vector<TNode*> VecTNode;

extern PermuteInfo* permute_info_arr;
extern int permute_type;

int* permute_order(int ith, int ncell, int nnode, int* parent);

class PermuteInfo
{
public:
    PermuteInfo();
    ~PermuteInfo();

    int nwarp, nstride, norder;
    int threads_num, nthread_each_cell;
    int *stride, *firstnode, *lastnode;
    int *cellsize, *map_t2c;
    int *max_order_each_thread, *min_order_each_thread;

};

class TNode {
  public:
    TNode(int ix);
    virtual ~TNode();
    TNode* parent;
    VecTNode children;
    size_t mkhash();
    size_t hash;
    size_t treesize;
    size_t nodevec_index;//这是在排序后的nodevec中的下标
    size_t treenode_order;
    size_t level;
    size_t cellindex;//属于哪个cell
    int groupindex; // warp index
    int nodeindex;//这是在原始的coredat中的排序（vec_v）

    int current_index;//从根节点开始的编码，从0开始
    int nodeindex_in_split;
    int branchindex; // 0-31, thread index in warp 记录这是由哪一个线程处理的
    //int leaf_num;
    //int start_thread;
    int down_component;
    int src_dc;
    int nodeorder;//第几轮进行处理
    int branch_size;//处理当前node的线程总共需要处理的任务数
    double reset_v;

};



#endif
