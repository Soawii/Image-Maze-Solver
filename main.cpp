#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <ctime>
#include <exception>
#include <string>
#include <filesystem>

using namespace cv;
namespace fs = std::filesystem;

const float sqrt_2 = sqrt(2.0f);

bool is_color_in_range(const cv::Vec3b& value, const cv::Vec3b& lower, const cv::Vec3b& upper) {
    for (int i = 0; i < 3; ++i)
    {
        if (value[i] < lower[i] || value[i] > upper[i])
        {
            return false;
        }
    }
    return true;
}

float ManhattanDistance(const Point2i& l, const Point2i& r) { return abs(l.x - r.x) + abs(l.y - r.y); };
float EuclidDistance(const Point2i& l, const Point2i& r) { return sqrt((r.x - l.x) * (r.x - l.x) + (r.y - l.y) * (r.y - l.y)); };

struct PathfindingResult
{ 
    std::vector<Point2i> path;
    std::vector<Point2i> animation_order;
};

struct PathfindingSettings
{
    float(*AStar_heuristic)(const Point2i&, const Point2i&) = ManhattanDistance;
    bool animation_enabled = false;
    int animation_length = 8;
};

struct MazeSettings
{
    std::vector<std::pair<Vec3i, Vec3i>> acceptable_start_hsv_ranges = { {{0, 100, 100}, {10, 255, 255}},
                                                                     {{170, 100, 100}, {180, 255, 255}} };
    std::vector<std::pair<Vec3i, Vec3i>> acceptable_end_hsv_ranges = { {{36, 100, 100}, {86, 255, 255}} };
    Vec3i display_start_color = { 0, 0, 255 }, display_end_color = { 0, 255, 0 };
    Vec3i animation_start_color = { 100, 50, 0 }, animation_end_color = { 50, 100, 0 };
    PathfindingResult(*pathfinding_alorithm)(const Mat&, Point2i, Point2i, PathfindingSettings);
    int threshold = 190;
};

PathfindingResult BFS(const Mat& maze, Point2i start_point, Point2i end_point, PathfindingSettings settings = {})
{
    struct BFSTile {
        float distance_from_start;
        Point2i point;
    };

    struct BFSTile_compare {
        bool operator()(const BFSTile& l, const BFSTile& r) {
            return l.distance_from_start > r.distance_from_start;
        }
    };

    PathfindingResult result;

    MatSize size = maze.size;

    const int8_t EMPTY_VALUE = -1;
    const int8_t WALL_VALUE = 16;

    std::vector<int8_t> where_we_came_from(size[0] * size[1], EMPTY_VALUE);

    const uchar* p;
    for (int i = 0; i < size[0]; i++)
    {
        p = maze.ptr<uchar>(i);
        for (int j = 0; j < size[1]; j++)
        {
            if (p[j] == 0) {
                where_we_came_from[i * size[1] + j] = WALL_VALUE;
            }
        }
    }
    where_we_came_from[start_point.x * size[1] + start_point.y] = 0;
    where_we_came_from[end_point.x * size[1] + end_point.y] = EMPTY_VALUE;

    std::priority_queue< BFSTile, std::vector<BFSTile>, BFSTile_compare> pq;

    BFSTile start = { 0.0, start_point };
    pq.push(start);

    bool search_done = false;

    while (!pq.empty())
    {
        const BFSTile& top = pq.top();
        float distance_from_start = top.distance_from_start;
        Point2i point = top.point;
        pq.pop();

        if (settings.animation_enabled)
            result.animation_order.push_back(point);

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0)
                    continue;
                const Point2i new_point = { point.x + i, point.y + j };

                if (new_point.x < 0 || new_point.x == size[0] || new_point.y < 0 || new_point.y == size[1])
                    continue;

                const int index = new_point.x * size[1] + new_point.y;
                if (where_we_came_from[index] != EMPTY_VALUE) continue;
                where_we_came_from[index] = i * 3 + j + 4;

                float new_distance;
                if (abs(i) + abs(j) == 2) {
                    new_distance = distance_from_start + sqrt_2;
                }
                else {
                    new_distance = distance_from_start + 1;
                }
                if (new_point == end_point) {

                    search_done = true;
                    break;
                }
                pq.push({ new_distance, new_point });
            }
        }

        if (search_done) break;
    }

    Point2i current_point = end_point;
    std::vector<std::pair<int, int>> path_decoder = { {1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, 0}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1} };

    while (true)
    {
        result.path.push_back(current_point);
        if (current_point == start_point)
            break;
        std::pair<int, int> didj = path_decoder[where_we_came_from[current_point.x * size[1] + current_point.y]];
        current_point = { current_point.x + didj.first, current_point.y + didj.second };
    }
    return result;
}

PathfindingResult AStar(const Mat& maze, Point2i start_point, Point2i end_point, PathfindingSettings settings = {})
{
    struct AStarTile {
        float distance_from_start;
        float heuristic_to_end;
        Point2i point;
    };

    struct AStarTile_compare {
        bool operator()(const AStarTile& l, const AStarTile& r) {
            return (l.distance_from_start + l.heuristic_to_end) > (r.distance_from_start + r.heuristic_to_end);
        }
    };

    PathfindingResult result;
    auto heuristic = settings.AStar_heuristic;
    MatSize size = maze.size;

    const int8_t EMPTY_VALUE = -1;
    const int8_t WALL_VALUE = 16;

    std::vector<int8_t> where_we_came_from(size[0] * size[1], EMPTY_VALUE);

    const uchar* p;
    for (int i = 0; i < size[0]; i++)
    {
        p = maze.ptr<uchar>(i);
        for (int j = 0; j < size[1]; j++)
        {
            if (p[j] == 0) {
                where_we_came_from[i * size[1] + j] = WALL_VALUE;
            }
        }
    }
    where_we_came_from[start_point.x * size[1] + start_point.y] = 0;
    where_we_came_from[end_point.x * size[1] + end_point.y] = EMPTY_VALUE;

    std::priority_queue< AStarTile, std::vector<AStarTile>, AStarTile_compare> pq;

    AStarTile start = { 0.0, heuristic(start_point, end_point), start_point };
    pq.push(start);

    bool search_done = false;

    while (!pq.empty())
    {
        const AStarTile& top = pq.top();
        float distance_from_start = top.distance_from_start;
        Point2i point = top.point;
        pq.pop();

        if (settings.animation_enabled)
            result.animation_order.push_back(point);

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0)
                    continue;
                const Point2i new_point = { point.x + i, point.y + j };

                if (new_point.x < 0 || new_point.x == size[0] || new_point.y < 0 || new_point.y == size[1])
                    continue;

                const int index = new_point.x * size[1] + new_point.y;
                if (where_we_came_from[index] != EMPTY_VALUE) continue;
                where_we_came_from[index] = i * 3 + j + 4;

                float new_distance;
                if (abs(i) + abs(j) == 2)
                    new_distance = distance_from_start + sqrt_2;
                else
                    new_distance = distance_from_start + 1;

                if (new_point == end_point) {

                    search_done = true;
                    break;
                }
                pq.push({ new_distance, heuristic(new_point, end_point), new_point });
            }
        }

        if (search_done) break;
    }

    Point2i current_point = end_point;
    std::vector<std::pair<int, int>> path_decoder = { {1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, 0}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1} };

    while (true)
    {
        result.path.push_back(current_point);
        if (current_point == start_point)
            break;
        int test = where_we_came_from[current_point.x * size[1] + current_point.y];
        std::pair<int, int> dxdy = path_decoder[test];
        current_point = { current_point.x + dxdy.first, current_point.y + dxdy.second };
    }

    return result;
}

// maze should be an image in BGR format, where start is a red pixel, end is a green pixel
Mat solve_maze(const Mat& bgr_image,
    PathfindingSettings pathfinding_settings,
    MazeSettings maze_settings)
{
    Mat hsv_image;
    cvtColor(bgr_image, hsv_image, COLOR_BGR2HSV);

    // Detect start and end
    Point2i start{ -1, -1 }, end{ -1, -1 };
    MatSize size = bgr_image.size;

    for (int i = 0; i < size[0]; i++)
    {
        for (int j = 0; j < size[1]; j++)
        {
            Vec3b& pixel_value = hsv_image.at<Vec3b>(i, j);

            if (start.x == -1)
            {
                for (const auto& p : maze_settings.acceptable_start_hsv_ranges)
                    if (is_color_in_range(pixel_value, p.first, p.second))
                        start = Point2i(i, j);
            }

            if (end.x == -1)
            {
                for (const auto& p : maze_settings.acceptable_end_hsv_ranges)
                    if (is_color_in_range(pixel_value, p.first, p.second))
                        end = Point2i(i, j);
            }
        }
        if (start.x != -1 && end.x != -1)
            break;
    }

    if (start.x == -1 || end.x == -1)
    {
        throw std::invalid_argument("START AND / OR END could not be found on the image - check that START is a red pixel, END is a green pixel");
    }

    // Prepare image for the pathfinding algorithm (it only consist of 0's (WALLS) and 255's (FREE))
    Mat gray_image, thresh_image;
    cvtColor(bgr_image, gray_image, COLOR_BGR2GRAY);
    threshold(gray_image, thresh_image, maze_settings.threshold, 255, THRESH_BINARY);

    // Finding the path
    PathfindingResult result = maze_settings.pathfinding_alorithm(thresh_image, start, end, pathfinding_settings);

    // Drawing the path
    if (maze_settings.display_end_color[0] == -1)
        maze_settings.display_end_color = maze_settings.display_start_color;
    Vec3i display_color_diff = { maze_settings.display_start_color[0] - maze_settings.display_end_color[0],
        maze_settings.display_start_color[1] - maze_settings.display_end_color[1] ,
        maze_settings.display_start_color[2] - maze_settings.display_end_color[2] };

    if (maze_settings.animation_end_color[0] == -1)
        maze_settings.animation_end_color = maze_settings.animation_start_color;
    Vec3i animation_color_diff = { maze_settings.animation_end_color[0] - maze_settings.animation_start_color[0],
        maze_settings.animation_end_color[1] - maze_settings.animation_start_color[1] ,
        maze_settings.animation_end_color[2] - maze_settings.animation_start_color[2] };

    Mat solved_maze(size[0], size[1], CV_8UC3), animation_image(size[0], size[1], CV_8UC3);
    bgr_image.copyTo(solved_maze);
    bgr_image.copyTo(animation_image);


    // Animation is done by wrtiting it to a temporary file using cv::VideoWriter and renaming it in the main loop
    VideoWriter writer;
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    int fps = 30;
    int frame_counter = 0;
    int tiles_per_frame = 0;

    // Animate
    if (pathfinding_settings.animation_enabled)
    {
        writer.open((fs::path("data") / fs::path("maze_animations") / fs::path("TEMP.avi")).string(), codec, fps, cv::Size(size[1], size[0]));

        if (!writer.isOpened()) {
            throw std::invalid_argument("Could not open temp video file");
        }

        tiles_per_frame = std::max(1, int(result.animation_order.size() + result.path.size()) / (fps * pathfinding_settings.animation_length));
        for (int i = 0; i < result.animation_order.size(); i++)
        {
            const Point2i& p = result.animation_order[i];
            const float current_multiplier = 1.0 * (i + 1) / result.animation_order.size();

            Vec3b current_color = { uchar(maze_settings.animation_start_color[0] + animation_color_diff[0] * current_multiplier),
                                    uchar(maze_settings.animation_start_color[1] + animation_color_diff[1] * current_multiplier),
                                    uchar(maze_settings.animation_start_color[2] + animation_color_diff[2] * current_multiplier) };

            animation_image.at<Vec3b>(p.x, p.y) = current_color;

            if (frame_counter == tiles_per_frame)
            {
                writer.write(animation_image);
                frame_counter = 0;
            }
            else frame_counter++;
        }
    }

    for (int i = 0; i < result.path.size(); i++)
    {
        const Point2i& p = result.path[i];
        const float current_multiplier = 1.0 * (i) / result.path.size();

        Vec3b current_color = { uchar(maze_settings.display_end_color[0] + display_color_diff[0] * current_multiplier),
                                uchar(maze_settings.display_end_color[1] + display_color_diff[1] * current_multiplier),
                                uchar(maze_settings.display_end_color[2] + display_color_diff[2] * current_multiplier) };

        solved_maze.at<Vec3b>(p.x, p.y) = current_color;
        if (pathfinding_settings.animation_enabled)
        {
            animation_image.at<Vec3b>(p.x, p.y) = current_color;
            if (frame_counter == tiles_per_frame)
            {
                writer.write(animation_image);
                frame_counter = 0;
            }
            else frame_counter++;
        }
    }

    if (pathfinding_settings.animation_enabled) {
        for (int i = 0; i < fps * 3; i++) {
            writer.write(animation_image);
        }
        writer.release();
    }

    return solved_maze;
}

std::pair<Vec3b, Vec3b> parse_colors(std::string s) {
    s = s.substr(1);
    std::vector<unsigned char> colors = {};
    std::string current_color = "";
    for (char c : s) {
        if (c == ',' || c == ')') {
            colors.push_back(std::stoi(current_color));
            current_color = "";
        }
        else {
            current_color.push_back(c);
        }
    }
    return { { colors[0], colors[1], colors[2] }, { colors[3], colors[4], colors[5] } };
}

// Command line arguments:
// 
// !!! DO NOT PUT ANY SPACES BETWEEN THE ARGUMENTS THAT ARE IN () BRACKETS, ONLY COMMAS !!!
// 
// 1 - path to the folder with your mazes   
//          VALUES: any string                               
//          DEFAULT VALUE: mazes   
//          examples: mazes-test, C:\\mazes_test
// 2 - algorithm settings                   
//          VALUES: BFS, AStar-MH, AStar-EU
//          DEFAULT VALUE: BFS  
//          -- BFS - Breadth First Search, AStar-MH - AStar with manhattan heuristic, AStar-EU - AStar with euclidian heuristic
// 3 - path colors in BGR format (start and end)
//          VALUES: (uint8,uint8,uint8,uint8,uint8,uint8)
//          DEFAULT VALUE: (0,0,255,0,255,0)
//          -- first color - 3 first numbers, second color - the last 3
// 4 - video animation enabled (takes some time)       
//          VALUES: 0, 1                                  
//          DEFAULT VALUE: 0
// 5 - video animation length per image (in seconds)     
//          VALUES: 1 to 60                           
//          DEFAULT VALUE: 8
// 6 - animation colors in BGR format (start and end)
//          VALUES: (uint8,uint8,uint8,uint8,uint8,uint8)
//          DEFAULT VALUE: (100,50,0,50,100,0)
//          -- first color - 3 first numbers, second color - the last 3
// 7 - threshold                            
//          VALUES: 0 to 255      
//          DEFAULT VALUE: 190
//          -- grayscale value to be used in cv::threshold function
//          -- any pixel value in the original grayscale image that is higher than threshold will be considered free space
//          -- any pixel value lower will be considered a wall
// 
// EXAMPLES:
// 
// 1) .\exe_name C:\\mazes_test [BFS] [0,255,255,255,0,0] 1 6 [255,255,255,0,0,0] 200
// 2) .\exe_name C:\\mazes_test [AStar,MH] [0,0,255,255,0,0]
// 3) .\exe_name C:\\mazes_test
// 
// 
// The output mazes will be put in the folder solved_mazes in the project path


int main(int argc, char** argv)
{
    // each pair in mazes vector has {path, name}
    // path - relative or absolute path to the maze image
    // name - how the maze will be named when it is solved
    std::vector<std::pair<std::string, std::string>> mazes;
    std::string algorithm_name = "BFS";

    PathfindingSettings pathfinding_settings = {};
    pathfinding_settings.AStar_heuristic = ManhattanDistance;
    pathfinding_settings.animation_enabled = false;
    pathfinding_settings.animation_length = 8;

    MazeSettings maze_settings = {};
    maze_settings.pathfinding_alorithm = BFS;

    fs::path folderPath = fs::path("data") / fs::path("mazes");

    if (argc >= 2) {
        folderPath = fs::path(std::string(argv[1]));
    }
    if (argc >= 3) {
        algorithm_name = std::string(argv[2]);
        if (algorithm_name == "BFS")
            maze_settings.pathfinding_alorithm = BFS;
        else if (algorithm_name.substr(0, 5) == "AStar") {
            maze_settings.pathfinding_alorithm = AStar;
            std::string astar_mode = algorithm_name.substr(6, 2);
            if (astar_mode == "MH")
                pathfinding_settings.AStar_heuristic = ManhattanDistance;
            else if (astar_mode == "EU")
                pathfinding_settings.AStar_heuristic = EuclidDistance;
            else {
                std::cout << "invalid AStar mode: " + astar_mode;
                return 0;
            }
        }
        else {
            std::cout << "invalid algorithm mode: " + algorithm_name;
            return 0;
        }
    }
    if (argc >= 4) {
        auto color_pair = parse_colors(std::string(argv[3]));
        maze_settings.display_start_color = color_pair.first;
        maze_settings.display_end_color = color_pair.second;
    }
    if (argc >= 5) {
        pathfinding_settings.animation_enabled = bool(argv[4][0] - '0');
        if (pathfinding_settings.animation_enabled)
            std::cout << "Warning: animation on big files with big animation length may take some time (up to 3 minutes)\n";
    }
    if (argc >= 6) {
        pathfinding_settings.animation_length = std::stoi(std::string(argv[5]));
        pathfinding_settings.animation_length = std::max(pathfinding_settings.animation_length, 1);
        pathfinding_settings.animation_length = std::min(pathfinding_settings.animation_length, 60);
    }
    if (argc >= 7) {
        auto color_pair = parse_colors(std::string(argv[6]));
        maze_settings.animation_start_color = color_pair.first;
        maze_settings.animation_end_color = color_pair.second;
    }
    if (argc >= 8) {
        maze_settings.threshold = std::stoi(std::string(argv[7]));
    }

    for (const auto& entry : fs::directory_iterator(folderPath))
    {
        if (fs::is_regular_file(entry))
        {
            mazes.push_back({ entry.path().string(), entry.path().filename().string() });
        }
    }

    for (int i = 0; i < mazes.size(); i++)
    {
        Mat bgr_image = imread(mazes[i].first);

        float start_time = clock();

        Mat solved_maze = solve_maze(bgr_image, pathfinding_settings, maze_settings);

        if (pathfinding_settings.animation_enabled) {
            fs::rename(fs::path("data") / fs::path("maze_animations") / fs::path("TEMP.avi"), (fs::path("data") / fs::path("maze_animations") / fs::path(algorithm_name + "_" + mazes[i].second)).replace_extension(fs::path("avi")));
        }

        float end_time = clock();
        std::cout << "Maze " + mazes[i].second + " solved, time taken : " << (end_time - start_time) / CLOCKS_PER_SEC << " seconds.\n";

        std::string file_name = (fs::path("data") / fs::path("solved_mazes") / fs::path(algorithm_name + "_" + mazes[i].second)).string();
        imwrite(file_name, solved_maze);
        imshow(file_name, solved_maze);
    }

    waitKey(0);
    return 0;
}