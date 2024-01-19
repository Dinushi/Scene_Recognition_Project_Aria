#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;


class PointsToPLYConverter {
public:
    PointsToPLYConverter(const std::string& csv_filename) : csv_filename_(csv_filename) {}

    // read the points from the dense_points.csv and add to co-ordinate lists
    void readDataFromCSV(const std::string& folder_path, std::vector<float>& x_coords, std::vector<float>& y_coords, std::vector<float>& z_coords) {
        std::string file_path = folder_path + "/" + csv_filename_;
        std::ifstream file(file_path);
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            //std::cout << "Line: " << line << std::endl;

            try {
                std::getline(lineStream, cell, ','); //read first column uid 
                std::getline(lineStream, cell, ','); //read second column graphUID
         
                // Read x coordinate
                if (!std::getline(lineStream, cell, ',') || cell.empty()) throw std::runtime_error("Missing or empty x coordinate");
                x_coords.push_back(std::stof(cell));

                // Read y coordinate
                if (!std::getline(lineStream, cell, ',') || cell.empty()) throw std::runtime_error("Missing or empty y coordinate");
                y_coords.push_back(std::stof(cell));

                // Read z coordinate
                if (!std::getline(lineStream, cell) || cell.empty()) throw std::runtime_error("Missing or empty z coordinate");
                z_coords.push_back(std::stof(cell));
            } catch (const std::invalid_argument& ia) {
                std::cerr << "Invalid argument: " << ia.what() << "\nLine content: " << line << "\nProblematic cell: " << cell << std::endl;
            } catch (const std::runtime_error& re) {
                std::cerr << "Runtime error: " << re.what() << "\nLine content: " << line << std::endl;
            }
        }
    }

    // save the ply version of the file at the given output folder
    PointCloud::Ptr createPointCloud(const std::vector<float>& x_coords, const std::vector<float>& y_coords, const std::vector<float>& z_coords) {
        PointCloud::Ptr cloud(new PointCloud);
        for (size_t i = 0; i < x_coords.size(); ++i) {
            cloud->push_back(PointT(x_coords[i], y_coords[i], z_coords[i]));
        }
        return cloud;
    }

private:
    std::string csv_filename_;
};

int main(int argc, char** argv) {
    if (argc < 3){
        PCL_ERROR ("Error: Syntax is: %s <path_to_input_data>\n\n", argv[0]);
        return -1;
    }

    // update the paths and scene ids
    std::string aria_input_data_path= argv[1];
    std::string output_ply_write_path = argv[2];
    // std::string aria_input_data_path = "/home/tran5174/temp_env/Scene_Recognition_Project_Aria/SynEnvDataset";
    // std::string output_ply_write_path = "/home/tran5174/temp_env/Scene_Recognition_Project_Aria/PLYDataset";

    int scene_number_start = std::stoi(argv[3]);
    int scene_number_end = std::stoi(argv[4]);
    // int scene_number_start = 0;
    // int scene_number_end = 0;

    std::string aria_dense_points_csv_name = "semidense_points.csv";
    PointsToPLYConverter converter(aria_dense_points_csv_name);

    for (int scene_number = scene_number_start; scene_number <= scene_number_end; ++scene_number) {
        std::vector<float> x_cord_list, y_cord_list, z_cord_list;
        std::string folder_path = aria_input_data_path + "/" + std::to_string(scene_number);
        converter.readDataFromCSV(folder_path, x_cord_list, y_cord_list, z_cord_list);

        // std::cout << "X-coordinates: ";
        // for (const auto& value : z_cord_list) {
        //     std::cout << value << " ";
        //     break;
        // }
        // std::cout << std::endl;

        PointCloud::Ptr cloud = converter.createPointCloud(x_cord_list, y_cord_list, z_cord_list);

        std::string ply_folder = output_ply_write_path + "/" + std::to_string(scene_number);
        std::filesystem::create_directories(ply_folder);
        std::string ply_path = ply_folder + "/" + std::to_string(scene_number) + "_original.ply";
        pcl::io::savePLYFileBinary(ply_path, *cloud);
    }

    return 0;
}
