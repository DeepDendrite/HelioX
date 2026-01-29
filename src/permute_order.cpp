#include "permute_order.h"
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <map>
#include <set>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <queue>
#include "utils.h"

using namespace std;

PermuteInfo *permute_info_arr = nullptr;
const int warpsize = 32;
extern int permute_type;
extern int nthread_each_cell;

static void node_interleave_order(int ncell, VecTNode &);
static int dynamic_interleave2(int ncell, VecTNode &nodevec, int nnode, int nthread_per_cell, int *&prev_node, int *&next_node);

static void admin1(int ncell,
                   VecTNode &nodevec,
                   int &nwarp,
                   int &nstride,
                   int *&stride,
                   int *&firstnode,
                   int *&lastnode,
                   int *&cellsize);

static void admin_dynamic2(int branch_per_cell,
                           int ncell,
                           int nbranch,
                           VecTNode &nodevec,
                           int &norder,
                           int *&max_order_per_thread,
                           int *&min_order_per_thread,
                           int *&firstnode,
                           int *&lastnode,
                           int *&stride,
                           int *&map_t2c,
                           int *nodeorder);

int *node_order_branch(int ncell,
                       int nnode,
                       int branch_each_cell,
                       int *parent,
                       int &norder,
                       int &nthread,
                       int *&max_order_each_thread,
                       int *&min_order_each_thread,
                       int *&firstnode,
                       int *&lastnode,
                       int *&stride,
                       int *&map_t2c);

int *node_order(int ncell,
                int nnode,
                int *parent,
                int &nwarp,
                int &nstride,
                int *&stride,
                int *&firstnode,
                int *&lastnode,
                int *&cellsize);

TNode::TNode(int ix)
{
    nodeindex = ix;
    cellindex = 0;
    groupindex = -1;
    level = 0;
    hash = 0;
    treesize = 1;
    nodevec_index = 0;
    treenode_order = 0;
    branchindex = -1;
    nodeorder = -1;
    parent = nullptr;
    children.reserve(2);
}

TNode::~TNode()
{
}

PermuteInfo::PermuteInfo()
{
    nwarp = 0;
    nstride = 0;
    norder = 0;
    threads_num = 0;
    nthread_each_cell = 0;
    stride = nullptr;
    firstnode = nullptr;
    lastnode = nullptr;
    cellsize = nullptr;
    map_t2c = nullptr;
    max_order_each_thread = nullptr;
    min_order_each_thread = nullptr;
}

PermuteInfo::~PermuteInfo()
{
    if (stride)
    {
        delete[] stride;
        stride = nullptr;
    }
    if (firstnode)
    {
        delete[] firstnode;
        firstnode = nullptr;
    }
    if (lastnode)
    {
        delete[] lastnode;
        lastnode = nullptr;
    }
    if (cellsize)
    {
        delete[] cellsize;
        cellsize = nullptr;
    }
    if (map_t2c)
    {
        delete[] map_t2c;
        map_t2c = nullptr;
    }
    if (max_order_each_thread)
    {
        delete[] max_order_each_thread;
        max_order_each_thread = nullptr;
    }
    if (min_order_each_thread)
    {
        delete[] min_order_each_thread;
        min_order_each_thread = nullptr;
    }
}

// permute_order(coredata->id, coredata->n_real_output, coredata->end, coredata->_v_parent_index)
int *permute_order(int ith, int ncell, int nnode, int *parent) // cellorder.cpp :line 346 interleave_order
{
    // return if there are no nodes to permute
    if (nnode <= 0)
        return nullptr;

    // ensure parent of root = -1
    for (int i = 0; i < ncell; i++)
    {
        if (parent[i] == 0)
            parent[i] = -1;
    }

    int nwarp, nstride, threads_num, norder;
    int *stride, *firstnode, *lastnode;
    int *cellsize, *max_order_each_thread, *min_order_each_thread;
    int *map_t2c;

    stride = nullptr;
    firstnode = nullptr;
    lastnode = nullptr;
    cellsize = nullptr;
    max_order_each_thread = nullptr;
    min_order_each_thread = nullptr;
    map_t2c = nullptr;

    int *order = nullptr;
    if (permute_type == 1)
    {
        order = node_order(ncell, nnode, parent, nwarp, nstride, stride, firstnode, lastnode, cellsize);
    }
    else
    {
        //返回的是将细胞重排后的idx
        order = node_order_branch(ncell, nnode, nthread_each_cell, parent,                                                                  // 输入
                                  norder, threads_num, max_order_each_thread, min_order_each_thread, firstnode, lastnode, stride, map_t2c); // 这都是输出
        nwarp = (threads_num + 31) / 32;
    }

    //order实际上是一个map，把原始的node index映射到新的node index，将来要记录电压的时候，需要用到这个map
    

    // for (int i = 0; i < nnode; ++i){
    //     printf_debug("order[%d]=[%d] ",i, order[i]);
    // }
    // printf_debug("\n");

    if (permute_info_arr)
    {
        PermuteInfo &pi = permute_info_arr[ith];
        pi.nwarp = nwarp;
        pi.nstride = nstride;
        pi.nthread_each_cell = nthread_each_cell;
        pi.stride = stride;
        pi.firstnode = firstnode;
        pi.lastnode = lastnode;
        pi.cellsize = cellsize;
        pi.map_t2c = map_t2c;
        pi.max_order_each_thread = max_order_each_thread;
        pi.min_order_each_thread = min_order_each_thread;
        pi.norder = norder;
        pi.threads_num = threads_num;
    }

    return order;
}

static size_t groupsize = 32;

static bool tnode_earlier(TNode *a, TNode *b)
{
    if (a->treesize != b->treesize)
        return a->treesize < b->treesize;

    if (a->hash != b->hash) // if treesize same, keep identical trees together
        return a->hash < b->hash;

    return a->nodeindex < b->nodeindex;
}

static bool ptr_tnode_earlier(TNode *a, TNode *b)
{
    return tnode_earlier(a, b);
}

size_t TNode::mkhash()
{ // call on all nodes in leaf to root order
    // concept from http://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
    std::sort(children.begin(), children.end(), ptr_tnode_earlier);
    hash = children.size();
    treesize = 1;
    for (size_t i = 0; i < children.size(); ++i)
    { // need sorted by child hash
        hash ^= children[i]->hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        treesize += children[i]->treesize;
    }
    return hash; // hash of leaf nodes is 0
}

static void quality(VecTNode &nodevec, size_t max = 32)
{
    size_t qcnt = 0; // how many contiguous nodes have contiguous parents

    // first ncell nodes are by definition in contiguous order
    // 前ncell个节点都代表的是细胞的root，是连续存放的，他这读出有多少个细胞
    for (size_t i = 0; i < nodevec.size(); ++i)
    {
        if (nodevec[i]->parent != NULL)
        {
            break;
        }
        qcnt += 1;
    }
    size_t ncell = qcnt;

    // key is how many parents in contiguous order
    // value is number of nodes that participate in that
    std::map<size_t, size_t> qual;
    size_t ip_last = 10000000000;
    for (size_t i = ncell; i < nodevec.size(); ++i)
    {
        size_t ip = nodevec[i]->parent->nodevec_index;
        // i%max == 0 means that if we start a warp with 8 and then have 32
        // the 32 is broken into 24 and 8. (modify if the arrangement during
        // gaussian elimination becomes more sophisticated.(
        if (ip == ip_last + 1 && i % max != 0)
        { // contiguous
            qcnt += 1;
        }
        else
        {
            if (qcnt == 1)
            {
                // printf("unique %ld p=%ld ix=%d\n", i, ip, nodevec[i]->nodeindex);
            }
            qual[max] += (qcnt / max) * max;
            size_t x = qcnt % max;
            if (x)
            {
                qual[x] += x;
            }
            qcnt = 1;
        }
        ip_last = ip;
    }
    qual[max] += (qcnt / max) * max;
    size_t x = qcnt % max;
    if (x)
    {
        qual[x] += x;
    }

    // print result
    qcnt = 0;
#if 0
  for (map<size_t, size_t>::iterator it = qual.begin(); it != qual.end(); ++it) {
    qcnt += it->second;
    printf("%6ld %6ld\n", it->first, it->second);
  }
#endif
#if 0
  printf("qual.size=%ld  qual total nodes=%ld  nodevec.size=%ld\n",
    qual.size(), qcnt, nodevec.size());
#endif

    // how many race conditions. ie refer to same parent on different core
    // of warp (max cores) or parent in same group of max.
    size_t maxip = ncell;
    size_t nrace1 = 0;
    size_t nrace2 = 0;
    std::set<size_t> ipused;
    for (size_t i = ncell; i < nodevec.size(); ++i)
    {
        TNode *nd = nodevec[i];
        size_t ip = nd->parent->nodevec_index;
        if (i % max == 0)
        {
            maxip = i;
            ipused.clear();
        }
        if (ip >= maxip)
        {
            nrace1 += 1;
        } /*else*/
        {
            if (ipused.find(ip) != ipused.end())
            {
                nrace2 += 1;
                if (ip >= maxip)
                {
                    // printf("race for parent %ld (parent in same group as multiple users))\n",
                    // ip);
                }
            }
            else
            {
                ipused.insert(ip);
            }
        }
    }
#if 0
  printf("nrace = %ld (parent in same group of %ld nodes)\n", nrace1, max);
  printf("nrace = %ld (parent used more than once by same group of %ld nodes)\n", nrace2, max);
#endif
}

size_t level_from_root(VecTNode &nodevec)
{
    size_t maxlevel = 0;
    for (size_t i = 0; i < nodevec.size(); ++i)
    {
        TNode *nd = nodevec[i];
        if (nd->parent)
        {
            nd->level = nd->parent->level + 1;
            if (maxlevel < nd->level)
            {
                maxlevel = nd->level;
            }
        }
        else
        {
            nd->level = 0;
        }
    }
    return maxlevel;
}

static void set_groupindex(VecTNode &nodevec)
{
    for (size_t i = 0; i < nodevec.size(); ++i)
    {
        TNode *nd = nodevec[i];
        if (nd->parent)
        {
            nd->groupindex = nd->parent->groupindex;
        }
        else
        {
            nd->groupindex = i / groupsize;
        }
    }
}

static void set_cellindex(int ncell, VecTNode &nodevec)
{
    for (int i = 0; i < ncell; ++i)
    {
        nodevec[i]->cellindex = i;
    }
    for (size_t i = 0; i < nodevec.size(); ++i)
    {
        TNode &nd = *nodevec[i];
        for (size_t j = 0; j < nd.children.size(); ++j)
        {
            TNode *cnode = nd.children[j];
            cnode->cellindex = nd.cellindex;
        }
    }
}

void tree_analysis(int *parent, int nnode, int ncell, VecTNode &nodevec)
{
    //  VecTNode nodevec;

    // create empty TNodes (knowing only their index)
    nodevec.reserve(nnode); // 创建空节点（对应着仓室）
    for (int i = 0; i < nnode; ++i)
    {
        nodevec.push_back(new TNode(i));
    }

    // determine the (sorted by hash) children of each node
    for (int i = nnode - 1; i >= ncell; --i)
    {
        nodevec[i]->parent = nodevec[parent[i]];
        nodevec[i]->mkhash();
        nodevec[parent[i]]->children.push_back(nodevec[i]);
    } // 根据仓室之间的关系，把节点挂上。但是，为什么前n_cell不弄呢

    // determine hash of the cells
    for (int i = 0; i < ncell; ++i)
    {
        nodevec[i]->mkhash();
    }

    // 排序cell，似乎，[0,ncell)是真实细胞 而[ncell,nnode)是仓室
    std::sort(nodevec.begin(), nodevec.begin() + ncell, tnode_earlier);
}

// 验证子节点比父节点排在vec前面
void check(VecTNode &nodevec)
{
    // printf("check\n");
    size_t nnode = nodevec.size();
    size_t ncell = 0;
    for (size_t i = 0; i < nnode; ++i)
    {
        nodevec[i]->nodevec_index = i;
        if (nodevec[i]->parent == NULL)
        {
            ncell++;
        }
    }
    for (size_t i = 0; i < ncell; ++i)
    { // 前面这是代表真的细胞
        assert(nodevec[i]->parent == NULL);
    }
    for (size_t i = ncell; i < nnode; ++i)
    { // 验证子节点比父节点排在vec前面
        TNode &nd = *nodevec[i];
        if (nd.parent->nodevec_index >= nd.nodevec_index)
        {
            printf("error i=%ld nodevec_index=%ld parent=%ld\n", i, nd.nodevec_index,
                   nd.parent->nodevec_index);
        }
        assert(nd.nodevec_index > nd.parent->nodevec_index);
    }
}

int *node_order(int ncell, // cellorder1.cpp line 321 为什么有一个cellorder1啊，这复制的时候复制多了吗？
                int nnode,
                int *parent,
                int &nwarp,
                int &nstride,
                int *&stride,
                int *&firstnode,
                int *&lastnode,
                int *&cellsize)
{
    VecTNode nodevec;

    // nodevec[0:ncell] in increasing size, with identical trees together,
    // and otherwise nodeindex order
    tree_analysis(parent, nnode, ncell, nodevec);
    check(nodevec);

    set_cellindex(ncell, nodevec);
    set_groupindex(nodevec);
    level_from_root(nodevec);

    // nodevec[ncell:nnode] cells are interleaved in nodevec[0:ncell] cell order
    node_interleave_order(ncell, nodevec);
    check(nodevec);

    quality(nodevec);

    // the permutation
    int *nodeorder = new int[nnode];
    for (int i = 0; i < nnode; ++i)
    {
        TNode &nd = *nodevec[i];
        nodeorder[nd.nodeindex] = i;
    }

    // FILE *fp = fopen("order1", "wb");
    // fwrite(nodeorder, sizeof(int), nnode, fp);
    // fclose(fp);
    //  administrative statistics for gauss elimination
    admin1(ncell, nodevec, nwarp, nstride, stride, firstnode, lastnode, cellsize);

    /*
    原文是：
    看起来把admin2给阉割了？
        // administrative statistics for gauss elimination
        if (interleave_permute_type == 1) {
            admin1(ncell, nodevec, nwarp, nstride, stride, firstnode, lastnode, cellsize);
        } else {
            //  admin2(ncell, nodevec, nwarp, nstride, stridedispl, stride, rootbegin, nodebegin,
            //  ncycles);
            admin2(ncell, nodevec, nwarp, nstride, stridedispl, stride, firstnode, lastnode, cellsize);
        }
    */
    int ntopol = 1;
    for (int i = 1; i < ncell; ++i)
    {
        if (nodevec[i - 1]->hash != nodevec[i]->hash)
        {
            ntopol += 1;
        }
    }
#ifdef DEBUG
    printf("node_order:%d distinct tree topologies\n", ntopol);
#endif

    for (size_t i = 0; i < nodevec.size(); ++i)
    {
        delete nodevec[i];
    }

    return nodeorder;
}

static bool interleave_comp(TNode *a, TNode *b)
{
    bool result = false;
    if (a->treenode_order < b->treenode_order)
    {
        result = true;
    }
    else if (a->treenode_order == b->treenode_order)
    {
        if (a->cellindex < b->cellindex)
        {
            result = true;
        }
    }
    return result;
}

void node_interleave_order(int ncell, VecTNode &nodevec)
{
    int *order = new int[ncell];
    for (int i = 0; i < ncell; ++i)
    {
        order[i] = 0;
        nodevec[i]->treenode_order = order[i]++;
    }
    for (size_t i = 0; i < nodevec.size(); ++i)
    {
        TNode &nd = *nodevec[i];
        for (size_t j = 0; j < nd.children.size(); ++j)
        {
            TNode *cnode = nd.children[j];
            cnode->treenode_order = order[nd.cellindex]++;
        }
    }
    delete[] order;

    //  std::sort(nodevec.begin() + ncell, nodevec.end(), contig_comp);
    std::sort(nodevec.begin() + ncell, nodevec.end(), interleave_comp);

#if 0
  for (size_t i=0; i < nodevec.size(); ++i) {
    TNode& nd = *nodevec[i];
    printf("%ld cell=%ld ix=%d\n",  i, nd.cellindex, nd.nodeindex);
  }
#endif
}

static void admin1(int ncell,
                   VecTNode &nodevec,
                   int &nwarp,
                   int &nstride,
                   int *&stride,
                   int *&firstnode,
                   int *&lastnode,
                   int *&cellsize)
{
    // firstnode[i] is the index of the first nonroot node of the cell
    // lastnode[i] is the index of the last node of the cell
    // cellsize is the number of nodes in the cell not counting root.
    // nstride is the maximum cell size (not counting root)
    // stride[i] is the number of cells with an ith node.
    firstnode = new int[ncell];
    lastnode = new int[ncell];
    cellsize = new int[ncell];

    nwarp = (ncell % warpsize == 0) ? (ncell / warpsize) : (ncell / warpsize + 1);

    for (int i = 0; i < ncell; ++i)
    {
        firstnode[i] = -1;
        lastnode[i] = -1;
        cellsize[i] = 0;
    }

    nstride = 0;
    for (size_t i = ncell; i < nodevec.size(); ++i)
    {
        TNode &nd = *nodevec[i];
        size_t ci = nd.cellindex;
        if (firstnode[ci] == -1)
        {
            firstnode[ci] = i;
        }
        lastnode[ci] = i;
        cellsize[ci] += 1;
        if (nstride < cellsize[ci])
        {
            nstride = cellsize[ci];
        }
    }

    stride = new int[nstride + 1]; // in case back substitution accesses this
    for (int i = 0; i <= nstride; ++i)
    {
        stride[i] = 0;
    }
    for (size_t i = ncell; i < nodevec.size(); ++i)
    {
        TNode &nd = *nodevec[i];
        stride[nd.treenode_order - 1] += 1; // -1 because treenode order includes root
    }
}

int *node_order_branch(int ncell,
                       int nnode,
                       int branch_per_cell,
                       int *parent, // 上面的都是输入，下面的都是输出
                       int &norder,
                       int &nthread,
                       int *&max_order_per_thread,
                       int *&min_order_per_thread,
                       int *&firstnode,
                       int *&lastnode,
                       int *&stride,
                       int *&map_t2c)
{
    VecTNode nodevec;

    // nodevec[0:ncell] in increasing size, with identical trees together,
    // and otherwise nodeindex order
    tree_analysis(parent, nnode, ncell, nodevec);
    check(nodevec);

    set_cellindex(ncell, nodevec); // 从父节点开始，把每一个cell对应的树上节点都标记所属的cell
    set_groupindex(nodevec);       // 设置属于哪个WARP
    level_from_root(nodevec);      // 计算从根节点开始的高度（根节点高度为0）

    int *prev_nodeindex = new int[nnode];
    int *next_nodeindex = new int[nnode];
    int nbranch = 0;
    // 返回值：需要的线程数
    nbranch = dynamic_interleave2(ncell, nodevec, nnode, branch_per_cell, prev_nodeindex, next_nodeindex);
    // nbranch = primary_branch_split(ncell, nodevec, nnode, branch_per_cell, prev_nodeindex, next_nodeindex);
    nthread = nbranch;

    check(nodevec); // 再次检查，排序后的node是正常的

    quality(nodevec); // 查看一下访存效率，TODO:回头优化的时候再看看

    // fp = fopen("cell0_node_3.txt", "w");
    //  the permutation
    // 构建一个map，从原始的vec_v这的下标，寻找对应的TNode下标
    int *nodeorder = new int[nnode];
    for (int i = 0; i < nnode; ++i)
    {
        TNode &nd = *nodevec[i];
        nodeorder[nd.nodeindex] = i;
    }
    // fclose(fp);
    // fp = fopen("order3", "wb");
    // fwrite(nodeorder, sizeof(int), nnode, fp);
    // fclose(fp);

    // administrative statistics for gauss elimination
    // admin_branch(branch_per_cell, ncell, nodevec, norder, prev_nodeindex, next_nodeindex, prev_node, next_node,
    //            max_order_per_thread, min_order_per_thread, firstnode, lastnode, cellsize, nodeorder);
    // admin_dynamic(branch_per_cell, ncell, nodevec, norder, prev_nodeindex, next_nodeindex, prev_node, next_node,
    //            max_order_per_thread, min_order_per_thread, firstnode, lastnode, stride, nodeorder);
    admin_dynamic2(branch_per_cell, ncell, nbranch, nodevec, norder, max_order_per_thread, min_order_per_thread, firstnode, lastnode, stride, map_t2c, nodeorder);

    delete[] prev_nodeindex;
    delete[] next_nodeindex;
    prev_nodeindex = NULL;
    next_nodeindex = NULL;

#if 1
    int ntopol = 1;
    for (int i = 1; i < ncell; ++i)
    {
        if (nodevec[i - 1]->hash != nodevec[i]->hash)
        {
            ntopol += 1;
        }
    }
    printf("node_order_branch:%d distinct tree topologies\n", ntopol);
#endif

    for (size_t i = 0; i < nodevec.size(); ++i)
    {
        // printf("%d:%p\n", i, nodevec[i]);
        delete nodevec[i];
    }
    printf("delete nodevec finish\n");
    return nodeorder;
}

// 排列顺序：WARP、轮次、WARP内线程号、细胞号
static bool sort_by_branch_order(TNode *a, TNode *b)
{
    if (a->groupindex != b->groupindex)
        return a->groupindex < b->groupindex;
    if (a->nodeorder != b->nodeorder)
        return a->nodeorder < b->nodeorder;
    if (a->branchindex != b->branchindex)
        return a->branchindex < b->branchindex;
    if (a->cellindex != b->cellindex)
        return a->cellindex > b->cellindex;
    return a->nodeindex < b->nodeindex;
}

static void modify_orders(TNode *root, int number_to_add)
{
    int start = 1;
    queue<TNode *> q;
    q.push(root);

    // 把除了根节点外的所有节点node order全加上num_to_add，也就是外面的diff
    while (!q.empty())
    {
        TNode *nd = q.front();
        q.pop();
        if (!start)
            nd->nodeorder += number_to_add;
        else
            start = 0;
        for (int ichild = 0, nchild = nd->children.size(); ichild < nchild; ichild++)
        {
            q.push(nd->children[ichild]);
        }
    }
}

static void modify_branchindex(TNode *root, int *sorted_branchindex, int *branch_size)
{
    queue<TNode *> q;
    q.push(root);
    while (!q.empty())
    {
        TNode *nd = q.front();
        q.pop();
        nd->branch_size = branch_size[nd->branchindex];
        nd->branchindex = sorted_branchindex[nd->branchindex];
        for (int ichild = 0, nchild = nd->children.size(); ichild < nchild; ichild++)
        {
            q.push(nd->children[ichild]);
        }
    }
}

static void argsort(int *a, int len, int *idx)
{
    multimap<int, int> m;
    for (int i = 0; i < len; i++)
    {
        m.insert(pair<int, int>(a[i], i));
    }

    int index = len - 1;
    for (const auto &iter : m)
    {
        idx[iter.second] = index;
        // printf("first:%d second:%d sorted:%d\n", iter->first, iter->second, len-index-1);
        index--;
    }
}

struct CompareNode
{
    bool operator()(const TNode *a, const TNode *b) const
    {
        if (a->level != b->level)
            return a->level < b->level;
        return a->current_index > b->current_index;
    }
};

static void init_tree(const VecTNode &nodevec, int icell, TNode **tree_nodes, queue<TNode *> &q)
{
    while (!q.empty())
    {
        q.pop();
    }
    q.push(nodevec[icell]);
    TNode *nd;
    int index = 0;
    while (!q.empty())
    {
        nd = q.front();
        q.pop();
        nd->down_component = nd->treesize;
        nd->current_index = index;
        tree_nodes[index] = nd;
        index++;
        for (int i = 0, j = nd->children.size(); i < j; i++)
            q.push(nd->children[i]);
    }
}

// 统计某个细胞的分支数量是否超过了nthread_per_cell,如果超了，那要分裂一下
static bool need_split(const VecTNode &nodevec, int icell, int nthread_per_cell, queue<TNode *> &q)
{
    TNode *nd;
    while (!q.empty())
    {
        q.pop();
    }
    q.push(nodevec[icell]); // 根节点入队
    bool result = false;
    int total_branch = 0;
    while (!q.empty())
    {
        nd = q.front();
        q.pop();
        int c_size = nd->children.size();
        if (c_size == 0) // 没有子节点了，那这就是一个分支
            total_branch++;
        if (total_branch >= nthread_per_cell)
        {
            result = true;
            break;
        }
        for (int i = 0; i < c_size; i++)
            q.push(nd->children[i]);
    }
    return result;
}

/* s_branch_id: in-warp start thread id of cell
 * iwarp: warp id
 *
 */
static void split_cell(VecTNode &nodevec, int icell, int nthread_per_cell, int s_branch_id, int iwarp,
                       int *branch_size, int &max_order, int *prev_node, int *next_node, queue<TNode *> &q)
{
    int cell_size = (int)nodevec[icell]->treesize;
    int order = 0, max_level = -1;
    TNode **tree_nodes = new TNode *[cell_size];
    TNode **node_to_compute = new TNode *[nthread_per_cell]; // 每一个线程的任务队列
    TNode **prev_compute_node = new TNode *[nthread_per_cell];

    init_tree(nodevec, icell, tree_nodes, q);

    max_order = -1;
    int *children_left = new int[cell_size];
    priority_queue<TNode *, vector<TNode *>, CompareNode> pq;

    // 记录每个节点的子树大小
    for (int inode = 0; inode < cell_size; inode++)
    {
        children_left[inode] = (int)tree_nodes[inode]->children.size();
        if ((int)tree_nodes[inode]->level > max_level)
            max_level = (int)tree_nodes[inode]->level;

        // 将子树大小为0（叶节点入队）
        if (children_left[inode] == 0)
        {
            pq.push(tree_nodes[inode]);
        }
    }

    TNode *nd = NULL;
    while (!pq.empty())
    {
        int nnode_to_compute = 0;
        for (int ithread = 0; ithread < nthread_per_cell; ithread++)
        {
            if (pq.empty())
                break;
            nd = pq.top();
            pq.pop();
            node_to_compute[nnode_to_compute++] = nd;
        } // 每个线程最多取一个叶节点进行处理，例如，有N个叶节点，但是只有16个线程，所以只取16个

        if (nnode_to_compute == 0)
            break;
        // printf("order:%d\n", order);
        for (int ithread = 0; ithread < nnode_to_compute; ithread++)
        {
            nd = node_to_compute[ithread];
            nd->branchindex = s_branch_id + ithread; // 全局的线程ID
            nd->nodeorder = order;
            nd->groupindex = iwarp;

            branch_size[(int)nd->branchindex]++;
            if (order > 0) // 用双向链表来记录计算顺序，然后order是表示这是第几轮
            {
                auto last_node_idx = prev_compute_node[ithread]->nodeindex;
                auto this_node_idx = nd->nodeindex;
                next_node[this_node_idx] = last_node_idx;
                prev_node[last_node_idx] = this_node_idx;
                // printf("prev_nodeindex:%d prev_currentindex:%d\n", prev_compute_node[ithread]->nodeindex, prev_compute_node[ithread]->current_index);
            }
            // printf("nodeindex:%d current_index:%d ", nd->nodeindex, nd->current_index);
            prev_compute_node[ithread] = nd;

            if (nd->parent)
            {
                children_left[nd->parent->current_index]--;
                if (children_left[nd->parent->current_index] == 0)
                    pq.push(nd->parent);
            }
        }

        // printf("\n");
        if (order > max_order)
            max_order = order;
        order++;
        // max_order = nd->nodeorder;
    }

    // printf("cell:%d order:%d max_level:%d\n", icell, max_order, max_level);
    for (int inode = 0; inode < cell_size; inode++)
    {
        tree_nodes[inode]->nodeorder = max_order - tree_nodes[inode]->nodeorder;
    }
    delete[] node_to_compute;
    delete[] prev_compute_node;
    delete[] tree_nodes;
    delete[] children_left;
}

// 从父节点开始，按顺序执行
static void set_branch_info(TNode *root, int branchindex, int groupindex)
{
    // int cellsize = (int)root->treesize;
    int order = 0;
    TNode *nd = NULL;
    queue<TNode *> q;
    q.push(root);

    while (!q.empty())
    {
        nd = q.front();
        q.pop();
        nd->nodeorder = order;
        nd->branchindex = branchindex;
        nd->groupindex = groupindex;
        order++;
        for (int i = 0, j = nd->children.size(); i < j; i++)
            q.push(nd->children[i]);
    }
}

static int dynamic_interleave2(int ncell, VecTNode &nodevec, int nnode, int nthread_per_cell, // 输入
                               int *&prev_node, int *&next_node)                              // 输出
{
    int istart = 0, iend = 0;
    int used_nthread = 0, max_order = -1;
    int s_branch_id = 0, iwarp = 0;
    int nbranch = 0;
    bool *split = NULL;
    memset(prev_node, -1, sizeof(int) * nnode);
    memset(next_node, -1, sizeof(int) * nnode);
    split = new bool[ncell];
    printf("ncell:%d\n", ncell);
    for (int i = 0; i < ncell; i++)
        split[i] = false;

    queue<TNode *> q;
    while (iend < ncell)
    {
        used_nthread = 0;
        max_order = -1;      // 最大的任务量
        int branch_size[32]; // 每个线程上的任务量多大，每次最多调度一个WARP，也就是32个线程
        for (int i = 0; i < 32; i++)
        {
            branch_size[i] = 0;
        }
        // group cells into different warps
        for (iend = istart; iend < ncell; iend++)
        {
            split[iend] = need_split(nodevec, iend, nthread_per_cell, q);
            // printf("icell:%d split:%d used_thread:%d size:%d\n", iend, split[iend], used_nthread, nodevec[iend]->treesize);
            if (!split[iend]) // 如果分叉数小于nthread_per_cell这个阈值，那么只会用一个线程来进行整个cell的操作
                used_nthread++;
            else
                used_nthread += nthread_per_cell; // 否则会使用这么多个线程来进行细粒度并行
            if (used_nthread > 32)                // 如果最后一个cell干超了可用线程数，那不处理最后一个cell
                break;
            else if (used_nthread == 32) // 如果刚好最后一个线程能处理最后一个cell,那这个cell也加入处理
            {
                iend++;
                break;
            }
        }

        s_branch_id = 0;

        // printf("iwarp:%d istart:%d iend:%d\n", iwarp, istart, iend);

        // 遍历每一个待处理的cell
        for (int icell = istart; icell < iend; icell++)
        {
            // printf("\ticell:%d size:%d split:%d s_branch:%d\n", icell, nodevec[icell]->treesize, split[icell], s_branch_id);
            if (split[icell])
            {
                split_cell(nodevec, icell, nthread_per_cell, s_branch_id, iwarp, branch_size, max_order, prev_node, next_node, q);
                s_branch_id += nthread_per_cell;
                nbranch += nthread_per_cell;
            }
            else
            {
                set_branch_info(nodevec[icell], s_branch_id, iwarp); // ？？branch_size这没修改
                s_branch_id++;
                nbranch++;
            }
        }
        int sorted_branchindex[32];
        // sort by branch size from large to small
        // 获取排序之后的下标（原本的数据并没有被修改，只获得了新位置下标）
        argsort(branch_size, 32, sorted_branchindex);

        // 更新每个节点中，记录所属线程的任务量。同时将其索引改成排序后的索引（变量并未真的排序，只改变了变量内部的索引）
        for (int icell = istart; icell < iend; icell++)
        {
            modify_branchindex(nodevec[icell], sorted_branchindex, branch_size);
        }
        istart = iend;
        iwarp++;
    }

    // align nodes in cells with different sizes
    int *max_order_per_cell = NULL, *max_order_per_group = NULL;
    int nwarp = iwarp;
    printf("nwarps:%d nbranch:%d\n", nwarp, nbranch);
    max_order_per_cell = new int[ncell];
    max_order_per_group = new int[nwarp];
    memset(max_order_per_cell, -1, sizeof(int) * ncell);
    memset(max_order_per_group, -1, sizeof(int) * nwarp);

    // 更新max order per cell和per group
    for (int i = 0; i < nnode; i++)
    {
        TNode *nd = nodevec[i];
        if (nd->nodeorder > max_order_per_cell[nd->cellindex])
            max_order_per_cell[nd->cellindex] = nd->nodeorder;
        if (nd->nodeorder > max_order_per_group[nd->groupindex])
            max_order_per_group[nd->groupindex] = nd->nodeorder;
    }
    for (int icell = 0; icell < ncell; icell++)
    {
        TNode *nd = nodevec[icell];
        // 如果一个细胞所属的warp order大于cell order，也就是说同WARP内，存在比这个细胞更“高”的其他细胞
        int diff = max_order_per_group[nd->groupindex] - max_order_per_cell[nd->cellindex];
        if (diff > 0)
            modify_orders(nd, diff); // 把除了根节点外的所有节点order全加上这个差值
    }

    delete[] max_order_per_cell;
    delete[] max_order_per_group;
    delete[] split;

    for (int i = 0; i < nnode; i++) //??
    {
        if (prev_node[i] > nnode || prev_node[i] < -1)
            printf("i:%d prev_node:%d\n", i, prev_node[i]);
        if (next_node[i] > nnode || next_node[i] < -1)
            printf("i:%d next_node:%d\n", i, next_node[i]);
    }
    // 排列顺序：WARP、轮次、WARP内线程号、细胞号
    std::sort(nodevec.begin() + ncell, nodevec.end(), sort_by_branch_order);
    return nwarp * 32;
}

static void admin_dynamic2(int branch_per_cell,
                           int ncell,
                           int nbranch,
                           VecTNode &nodevec,
                           int &norder,
                           int *&max_order_per_thread,
                           int *&min_order_per_thread,
                           int *&firstnode,
                           int *&lastnode,
                           int *&stride,
                           int *&map_t2c,
                           int *nodeorder)
{
    std::cout << "****admin_dynamic2****" << std::endl;
    // nbranch实际上就是线程数，应该是继承了coreneuron中的命名
    firstnode = new int[nbranch];
    lastnode = new int[nbranch];
    max_order_per_thread = new int[nbranch];
    min_order_per_thread = new int[nbranch];

    int nnode = nodevec.size();

    norder = 0; // max node order
    for (int i = 0; i < nnode; i++)
    {
        if (norder < nodevec[i]->nodeorder)
            norder = nodevec[i]->nodeorder;
    }

    int nwarps = (nbranch + 31) / 32;
    stride = new int[nwarps * (norder + 1)]; // 每个WARP内，统一Stride,这样可以减少32倍的开销
    printf("nbranch:%d nwarps:%d norder:%d\n", nbranch, nwarps, norder);

    memset(stride, 0, sizeof(int) * nwarps * (norder + 1));
    memset(firstnode, -1, sizeof(int) * nbranch);
    memset(lastnode, -1, sizeof(int) * nbranch);
    for (int i = 0; i < nbranch; i++)
    {
        max_order_per_thread[i] = -1;
        min_order_per_thread[i] = 99999;
    }

    // thread to cell || cell to thread
    map_t2c = new int[nbranch];
    int *map_c2t = new int[ncell];//第一个处理该cell的线程

    std::fill_n(map_t2c, nbranch, -1);
    std::fill_n(map_c2t, ncell, nbranch + 100);//初始化为一个不可能的值，即，比最大线程数还大

    for (int i = ncell; i < nnode; i++)
    {
        TNode *nd = nodevec[i];
        int tid = nd->groupindex * 32 + nd->branchindex;
        if (tid < map_c2t[nd->cellindex]){
            // assert(nbranch + 100 == map_c2t[nd->cellindex]);//ljf:这地方怎么看起来怪怪的
            map_c2t[nd->cellindex] = tid;
        }
        if (tid >= nbranch)
            printf("**error***\n\t i:%d icell:%ld tid:%d iwarp:%d branch:%d\n", i, nd->cellindex, tid, nd->groupindex, nd->branchindex);

        if (firstnode[tid] == -1)
            firstnode[tid] = i;
        lastnode[tid] = i;

        if (max_order_per_thread[tid] < nd->nodeorder)
            max_order_per_thread[tid] = nd->nodeorder;
        if (min_order_per_thread[tid] > nd->nodeorder)
            min_order_per_thread[tid] = nd->nodeorder;

        stride[tid / 32 * (norder + 1) + nd->nodeorder] += 1;//当前步，每要处理一个数据，那下一步要跳过的数据就++
    }

    for (int i = 0; i < ncell; i++)
    {
        map_t2c[map_c2t[i]] = i;
    }//ljf：这会造成大量空穴

    delete[] map_c2t;
}
