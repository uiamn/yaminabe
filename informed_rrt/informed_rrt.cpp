#include <opencv2/opencv.hpp>
#include <time.h>
#include <math.h>
#include <random>
#include <stdio.h>
#include <Eigen/Dense>
#include<unistd.h>

#define SIZE 1000
#define MAX_ITER 2400
#define EXPAND_DISTANCE 100
#define GOAL_REGION_RADIUS 5
#define TERMINATE_RATIO 1.01

#define DUMMY -1

std::random_device rnd;
std::mt19937 mt(rnd());
std::vector<std::uniform_real_distribution<double>> dist_space = {
    std::uniform_real_distribution<double>(0.0, 1000.0),
    std::uniform_real_distribution<double>(0.0, 1000.0)
};
std::uniform_real_distribution<double> dist_unit(0.0, 1.0);
std::normal_distribution<double> dist_gauss(0.0, 1.0);

struct node {
    double x; double y; double cost; int parent_index;
};

struct edge {
    node v1; node v2;
};

double distance(node v1, node v2) {
    return sqrt(pow(v2.x - v1.x, 2) + pow(v2.y - v1.y, 2));
}

Eigen::MatrixXd calc_rotation_to_world_flame(node start, node goal){
    auto dist = distance(start, goal);
    std::vector<double> a1_v = {
        (goal.x - start.x) / dist, (goal.y - start.y) / dist, 0.0
    };

    auto M = Eigen::Map<Eigen::VectorXd>(&*a1_v.begin(), a1_v.size()) * Eigen::MatrixXd::Identity(1, a1_v.size());
    auto svd = Eigen::JacobiSVD<Eigen::MatrixXd>(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

    auto diag_v = std::vector<double>(a1_v.size(), 1.0);
    diag_v[diag_v.size() - 1] = svd.matrixV().determinant();
    diag_v[diag_v.size() - 2] = svd.matrixU().determinant();

    return svd.matrixU() * Eigen::Map<Eigen::VectorXd>(&*diag_v.begin(), diag_v.size()).asDiagonal() * svd.matrixV().transpose();
}


node sample(node start, node goal, double c_max, Eigen::MatrixXd rotate_mat) {
    node sampled_node;
    if(c_max == std::numeric_limits<double>::max()) {
        sampled_node.x = dist_space[0](mt);
        sampled_node.y = dist_space[1](mt);
    } else {
        // definition of diagonal element
        auto c_min = distance(start, goal);
        auto diag_val = std::sqrt(std::pow(c_max, 2) - std::pow(c_min, 2)) / 2.0;
        auto diag_v   = std::vector<double>(2 + 1, diag_val);
        diag_v[0]     = c_max / 2.0;

        // random sampling on unit n-ball
        double x, y;
        while(1) {
            x = dist_gauss(mt);
            y = dist_gauss(mt);
            auto r = std::sqrt(x*x+y*y);
            if(r != 0.0) {
                x = x / r * std::pow(dist_unit(mt), 0.5);
                y = y / r * std::pow(dist_unit(mt), 0.5);
                break;
            }
        }

        std::vector<double> x_ball_v = {x, y, 0.0};
        // printf("%f %f\n", c_max, c_min);

        Eigen::VectorXd x_centre(3);
        x_centre(0) = (start.x + goal.x) / 2;
        x_centre(1) = (start.y + goal.y) / 2;
        x_centre(2) = 0.0;

        // trans sampling pt
        auto rand = rotate_mat * Eigen::Map<Eigen::VectorXd>(&*diag_v.begin(), diag_v.size()).asDiagonal() * Eigen::Map<Eigen::VectorXd>(&*x_ball_v.begin(), x_ball_v.size()) + x_centre;

        sampled_node.x = rand(0, 0);
        sampled_node.y = rand(1, 0);

        // printf("%f, %f\n", sampled_node.x, sampled_node.y);
    }

    sampled_node.cost = NULL;
    sampled_node.parent_index = DUMMY;
    return sampled_node;
}

int search_nearest_node_index(node target, std::vector<node> nodes) {
    int ret_node_index;
    auto min_dist = std::numeric_limits<double>::max();

    for(auto i=0; i<nodes.size(); i++) {
        auto v = nodes[i];
        auto dist = distance(target, v);
        if(dist < min_dist) {
            ret_node_index = i;
            min_dist = dist;
        }
    }
    return ret_node_index;
}

node generate_steer_node(node src_node, node dst_node, double expand_distance) {
    auto distance_src_to_dst = distance(src_node, dst_node);
    node new_node;

    if (distance_src_to_dst < expand_distance) {
        new_node = {dst_node.x, dst_node.y, src_node.cost + distance_src_to_dst, DUMMY};
    } else {
        new_node = {
            src_node.x + (dst_node.x - src_node.x) / distance_src_to_dst * expand_distance,
            src_node.y + (dst_node.y - src_node.y) / distance_src_to_dst * expand_distance,
            src_node.cost + expand_distance,
            DUMMY
        };
    }

    // printf("%f %f\n", distance(new_node, src_node), distance_src_to_dst);

    return new_node;
}

bool is_collision(node n1, node n2) {
    // 未定義領域に存在しないか
    if(n1.x > SIZE || n1.y > SIZE || n2.x > SIZE || n2.y > SIZE
    || n1.x < 0 || n1.y < 0 || n2.x < 0 || n2.y < 0) return true;

    auto y_val = [n1, n2](double x) {return (n2.y - n1.y)/(n2.x - n1.x)*(x - n1.x) + n1.y;};
    auto x_val = [n1, n2](double y) {return (n2.x - n1.x)*(y - n1.y) / (n2.y - n1.y) + n1.x;};


    // 辺の長さが十分に小さいなら，どちらかのnodeが障害物と接触していないかどうかで判定できる
    if(300 < n1.x && n1.x < 700 && 250 < n1.y && n1.y < 750) return true;
    if(300 < n2.x && n2.x < 700 && 250 < n2.y && n2.y < 750) return true;


    // x:0~200, y:250~750に触れていないか
    // x軸に垂直な直線との接触
    // auto y = y_val(200);
    // if(250 < y && y < 750) return true;

    // if(n1.x < 400 && n2.x > 600) return true;
    // if(n2.x < 400 && n1.x > 600) return true;

    // auto y1 = y_val(300.0);
    // auto y2 = y_val(700.0);

    // if(250.0 < y1 && y1 < 750.0) return true;
    // if(250.0 < y2 && y2 < 750.0) return true;

    // auto x1 = x_val(250.0);
    // auto x2 = x_val(750.0);
    // if(300.0 < x1 && x1 < 700.0) return true;
    // if(300.0 < x2 && x2 < 700.0) return true;

    // y軸に垂直な直線との接触

    // auto x1 = x_val(250);
    // auto x2 = x_val(750);

    // if(0 <= x1 && x1 < 200) return true;
    // if(0 <= x2 && x2 < 200) return true;

    return false;
}


std::vector<node> searchNBHD(node n, std::vector<node> nodes, double radius) {
    std::vector<node> ret_nodes = {};
    for(auto v : nodes) {
        auto dist = distance(n, v);
        if(dist < radius) {
            ret_nodes.push_back(v);
        }
    }
    return ret_nodes;
}

double estimate_cost(node n, node goal) {
    return n.cost + distance(n, goal);
}

int main() {
    cv::Mat back = cv::Mat::zeros(SIZE, SIZE, CV_8UC3);
    cv::namedWindow("Informed-RRT*", cv::WINDOW_AUTOSIZE);
    cv::Rect obstacle(300, 250, 400, 500);
    cv::rectangle(back, obstacle, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    // node start = {1.0, 2.0, 0.0, DUMMY};
    // node goal = {0.0, SIZE, 0.0, DUMMY};

    node start = {250.0, 480.0, 0.0, DUMMY};
    node goal = {750.0, 520.0, 0.0, DUMMY};

    std::vector<node> solution = {};

    std::vector<node> nodes = {start};

    auto rotate_mat = calc_rotation_to_world_flame(start, goal);

    double best = std::numeric_limits<double>::max();
    node best_node;

    for(auto _=0; _<MAX_ITER; _++) {
        node x_rand;
        if(dist_unit(mt) < 0.25){
            x_rand = {750.0, 520.0, 0.0, DUMMY};
        } else {
            x_rand = sample(start, goal, best, rotate_mat);
        }

        // if(400.0 < x_rand.x && x_rand.x < 600.0 && 250.0 < x_rand.y && x_rand.y < 750.0) continue;

        auto x_nearest_index = search_nearest_node_index(x_rand, nodes);
        auto x_nearest = nodes[x_nearest_index];
        node x_new = generate_steer_node(x_nearest, x_rand, EXPAND_DISTANCE);

        if(!is_collision(x_nearest, x_new)) {
            auto nof_node = nodes.size();
            auto radius = std::min(
                (double) EXPAND_DISTANCE,
                EXPAND_DISTANCE * 1.5 * std::pow((std::log(nof_node) / nof_node), 1.0 / 2)
            );

            // 新しくできたNodeの周囲のノードを列挙
            auto near_nodes = searchNBHD(x_new, nodes, radius);

            // 新しくできたNodeのparentを更新する
            node x_min = x_nearest;
            int x_min_index = x_nearest_index;
            auto c_min = x_min.cost + distance(x_nearest, x_new);

            for(auto i=0; i<near_nodes.size(); i++) {
                auto x_near = near_nodes[i];
                auto c_new = x_near.cost + distance(x_new, x_near);
                if (c_new < c_min) {
                    if (!is_collision(x_new, x_near)) {
                        x_min_index = i;
                        c_min = c_new;
                    }
                }
            }

            if(c_min - nodes[x_min_index].cost > EXPAND_DISTANCE) {
                continue;
            }

            x_new.parent_index = x_min_index;
            x_new.cost = c_min;

            // 近くのノードはコストが更新される可能性があるので探索する
            for(auto& x_near: near_nodes) {
                auto c_near = x_near.cost;
                auto c_new = x_new.cost + distance(x_new, x_near);
                if(c_new < c_near) {
                    if(!is_collision(x_new, x_near)) {
                        x_near.parent_index = nodes.size();
                        x_near.cost = c_new;
                    }
                }
            }

            nodes.push_back(x_new);

            if(distance(goal, x_new) < GOAL_REGION_RADIUS) {
                solution.push_back(x_new);
            }

            for(auto s: solution) {
                if(s.cost + distance(s, goal) < best) {
                    best = s.cost + distance(s, goal);
                    best_node = s;
                }
            }

            if(best < TERMINATE_RATIO * distance(start, goal)) {
                printf("find solution!\n");
                break;
            }

        }
    }

    printf("======\n");

    for(auto n: nodes) {
        // printf("%f, %f\n", n.x, n.y);
        if(n.parent_index == DUMMY) continue;
        cv::line(
            back,
            cv::Point(n.x, n.y),
            cv::Point(nodes[n.parent_index].x, nodes[n.parent_index].y),
            cv::Scalar(200, 0, 0), 1
        );
    }

    printf("%d\n", nodes.size());
    node n = best_node;
    while(n.parent_index != DUMMY) {
        // printf("%d\n", n.parent_index);
        cv::line(
            back,
            cv::Point(n.x, n.y),
            cv::Point(nodes[n.parent_index].x, nodes[n.parent_index].y),
            cv::Scalar(0, 0, 200), 1
        );
        n = nodes[n.parent_index];
    }

    // printf("%d\n", is_collision({500, 100, 0, 0}, {800, 800, 0, 0}));

    cv::imshow("Informed-RRT*", back);
    cv::imwrite("result.png", back);
    cv::waitKey(0);
    cv::destroyWindow("Informed-RRT*");
    return 0;
}
