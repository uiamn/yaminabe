#include <opencv2/opencv.hpp>
#include <time.h>
#include <math.h>
#define N_OBJ 16
#define SIZE 600
#define OBJ_R_MAX 50
#define OBJ_R_MIN 20
#define MAX_ITR 10000
#define EDGE_LEN 30

using point = std::pair<double, double>;
using circ = std::pair<point, double>;

auto getLengthSq = [](point p1, point p2) {
    auto xl = p1.first - p2.first, yl = p1.second - p2.second;
    return xl*xl + yl*yl;
};

bool isIntersectPointAndCirc(point p, circ c) {
    auto x = c.first.first, y = c.first.second, r = c.second;
    // 半径より中心との距離が等しいか小さければ交わる
    auto xl = p.first - x, yl = p.second - y;
    return xl*xl + yl*yl <= r*r;
}

bool isIntersectLineAndCirc(point p1, point p2, circ ci) {
    auto x = ci.first.first, y = ci.first.second, r = ci.second;

    /*
        p1,p2を通る直線の方程式をAx+By+C=0とすると，
        A = p1.y - p2.y
        B = p2.x - p1.x
        C = p1.x(p2.y-p1.y) - p1.y(p2.x-p1.x) = p1.x(-A) - p1.y(B)
    */
    auto a = p1.second - p2.second;
    auto b = p2.first - p1.first;
    auto c = -(p1.first*a + p1.second*b);

    /*
        直線Ax+By+C=0と点(x,y)の距離の二乗L^2は
        L^2 = (Ax+By+C)^2/(A^2+B^2)
        これとr^2の大小を比較する(もしr^2 < L^2なら絶対に交はらない)
    */
    if(r*r * (a*a + b*b) < (a*x+b*y+c)*(a*x+b*y+c)) return false;

    if(isIntersectPointAndCirc(p1, ci) || isIntersectPointAndCirc(p2, ci)) return true;

    /*
        円の中心をCとして，角Cp1p2と角Cp2p1が共に鈍角でないなら交はる
        角ABCが鈍角でない⇔BAとBCの内積が非負を利用すると
        交はる⇔角Cp1p2と角Cp2p1が共に鈍角ではない
        ⇔(p1Cとp1p2の内積が非負)かつ(p2Cとp2p1の内積が非負)
        ⇔((x-p1.x)*(p2.x-p1.x) + (y-p1.y)*(p2.y-p1.y) >=0)
        かつ((x-p2.x)*(p1.x-p2.x) + (y-p2.y)*(p1.y-p2.y) >= 0)
    */
    if((x-p1.first)*(p2.first-p1.first)+(y-p1.second)+(p2.second-p1.second) >= 0
    &&(x-p2.first)*(p1.first-p2.first)+(y-p2.second)+(p1.second-p2.second) >= 0) return true;

    return false;
}

void drawObjects(cv::Mat back, circ objs[N_OBJ]) {
    for(int i=0; i<N_OBJ; i++) {
        auto obj = objs[i];
        auto x = obj.first.first, y = obj.first.second, r = obj.second;
        cv::circle(back, cv::Point(x, y), r, cv::Scalar(200, 0, 0), -1, CV_AA);
    }
}

int getNearestNodeIdx(point p, std::vector<point> nodes) {
    int tmp_idx = 0, idx = 0;
    double tmp_len = DBL_MAX;

    for(auto n: nodes) {
        auto l = getLengthSq(p, n);
        if(l < tmp_len) {
            tmp_len = l;
            tmp_idx = idx;
        }
        idx++;
    }

    return tmp_idx;
}

bool createEdge(
    std::vector<point>& nodes,
    circ objs[N_OBJ],
    cv::Mat back,
    point s, point d
) {
    /*
        sからdへ，長さEDGE_LENの辺を張ることを試みる
        張れた場合はnodesに新たな点を追加しtrueを返す
        さうでない場合はfalseを返す
    */
    double d_vec_len = sqrt(getLengthSq(s, d));

    std::pair<double, double> new_node = std::make_pair(
        s.first + (d.first-s.first) / d_vec_len * EDGE_LEN,
        s.second + (d.second-s.second) / d_vec_len * EDGE_LEN
    );

    for(int i=0; i<N_OBJ; i++) if(isIntersectLineAndCirc(s, new_node, objs[i])) {
        return false;
    };

    nodes.push_back(new_node);

    cv::line(
        back, cv::Point(s.first, s.second),
        cv::Point(new_node.first, new_node.second),
        cv::Scalar(0, 200, 0), 1
    );

    return true;
}

std::vector<int> getFinalPath(std::vector<std::pair<int, int>> edges, int finalNodesSize, int firstNodeIdx=0) {
    std::pair<int, int>* nodes;
    bool* searched;

    nodes = new std::pair<int, int>[finalNodesSize];  // pairの1つ目がstartからの距離，2つ目が直前のnodeのidx(-1は始点)
    searched = new bool[finalNodesSize];
    for(int i=0; i<finalNodesSize; i++) {
        nodes[i] = i==firstNodeIdx?std::make_pair(0, -1)
                                  :std::make_pair(INT_MAX, -2);
        searched[i] = false;
    }

    for(int _=0; _<finalNodesSize; _++) {
        // dijkstra
        int idx, tmpDist = INT_MAX;
        for(int i=0; i<finalNodesSize; i++) {
            if(!searched[i] && nodes[i].first < tmpDist) {
                idx = i;
                tmpDist = nodes[i].first;
            }
        }
        searched[idx] = true;

        for(auto e: edges) {
            int nbInd;
            if(e.first == idx) {
                nbInd = e.second;
            } else if(e.second == idx) {
                nbInd = e.first;
            } else {
                continue;
            }

            if(nodes[idx].first + 1 < nodes[nbInd].first) {
                nodes[nbInd].first = nodes[idx].first + 1;
                nodes[nbInd].second = idx;
            }
        }
    }

    std::vector<int> path;
    path.push_back(finalNodesSize-1);
    auto n = nodes[finalNodesSize-1].second;

    while(n != -1) {
        path.push_back(n);
        n = nodes[n].second;
    }

    delete[] nodes;
    delete[] searched;

    return path;
}

void rrt(cv::Mat back, circ objs[N_OBJ]) {
    std::vector<point> nodes = {
        std::make_pair(0, 0)
    };
    auto goal = std::make_pair(SIZE, SIZE);
    std::vector<std::pair<int, int>> edges;

    for(int _=0; _<MAX_ITR; _++) {

        // 1. 点をランダムサンプリング
        point p = std::make_pair(rand() % SIZE, rand() % SIZE);

        // 2. 最も近い点を見つける
        auto nearestNodeIdx = getNearestNodeIdx(p, nodes);
        auto nearestNode = nodes[nearestNodeIdx];

        // 3. 最も近い点から辺を張る
        if(createEdge(nodes, objs, back, nearestNode, p)) {
            edges.push_back(std::make_pair(nearestNodeIdx, nodes.size()-1));
        } else continue;

        // 4. ゴール(右下)からEDGE_LEN以内にnodeがあれば終了
        auto nearestNodeByGoal = nodes[getNearestNodeIdx(goal, nodes)];
        if(getLengthSq(nearestNodeByGoal, goal) < EDGE_LEN*EDGE_LEN) break;

        // cv::waitKey(100);
        // cv::imshow("Example", back);
    }

    // 最終的なパスを得る
    auto finalPath = getFinalPath(edges, nodes.size());
    for(int i=0; i<finalPath.size()-1; i++) cv::line(
        back,
        cv::Point(nodes[finalPath[i]].first, nodes[finalPath[i]].second),
        cv::Point(nodes[finalPath[i+1]].first, nodes[finalPath[i+1]].second),
        cv::Scalar(0, 0, 200), 1
    );
}


int main() {
    srand((unsigned) time(NULL));

    circ objs[N_OBJ];
    cv::Mat back = cv::Mat::zeros(SIZE, SIZE, CV_8UC3);

    cv::namedWindow("Example", cv::WINDOW_AUTOSIZE);

    for(int i=0; i<N_OBJ; i++) {
        auto x = rand() % SIZE, y = rand() % SIZE, r = rand() % (OBJ_R_MAX - OBJ_R_MIN) + OBJ_R_MIN;
        if(x < r || y < r || SIZE - x < r || SIZE - y < r) {
            i--;
            continue;
        }
        objs[i] = std::make_pair(std::make_pair(x, y), r);
    }

    drawObjects(back, objs);

    cv::circle(back, cv::Point(SIZE, SIZE), EDGE_LEN, cv::Scalar(0, 0, 200), -1, CV_AA);


    rrt(back, objs);

    cv::imshow("Example", back);
    cv::imwrite("a.png", back);
    cv::waitKey(0);
    cv::destroyWindow("Example");
    return 0;
}
