/*
A simple filter to keep only points approximately within the field of view of the camera

by Carlos Argueta

November 2, 2022
*/


#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

//#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h> 

#include <pcl/filters/frustum_culling.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <string>
#include <thread>
#include <sstream>
#include <iostream>
#include <filesystem>

#include <pcl/common/centroid.h>

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h> 

using namespace std::chrono_literals;
using namespace std; 

// Convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// Convenient structure to handle our pointclouds
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name; // Full Path
  std::string file_name; // File name only (without .ply ext)
  std::string path; // Path without the file name

  PCD() : cloud (new PointCloud) {};
};


/* Function to load PCD (Point Cloduds) from a directory. 
  param files_in_directory: path to directory containing PCD files
  param &data: vector with the loaded PCDs
*/
void loadData (std::vector<std::filesystem::path> files_in_directory, std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
{
  PCL_INFO("\n\nLooking for '_original.ply' file...\n\n");
  // Go over all of the entries in the path
  for (const auto & entry : files_in_directory){
    // if the entry is a file with extension .pcd, load it
    //if (entry.extension() == ".pcd"){
    //if (entry.extension() == ".ply" && entry.path().filename().string().find("_original") != std::string::npos){ //".pcd"
    if (entry.extension() == ".ply" && entry.filename().string().find("_original") != std::string::npos){ //".pcd"
      
      // Create pcd structure, assign it the path of the file as name and load the pcd to the cloud portion
      PCD p;
      p.f_name = entry.string();
      p.file_name = entry.stem().string(); // File name only
      p.path = entry.parent_path().string(); // Path without the file name

      //pcl::io::loadPCDFile (entry.string(), *p.cloud);
      pcl::io::loadPLYFile (entry.string(), *p.cloud);

      // Remove NAN points from the cloud
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*p.cloud,*p.cloud, indices);

      // Add PCD structure to the vector
      data.push_back (p);
    }
  }
}

// Function to apply a translation to a point cloud and estimate normals
PointCloud::Ptr transformPointCloud(const PointCloud::Ptr& input_cloud, float tx, float ty, float tz) {

    // Load the point cloud
    PointCloud::Ptr cloud(new PointCloud);
    //pcl::io::loadPLYFile("/home/tran5174/temp_env/camaraPoseExtraction/my_work/pcl_experiments/pcds/pcl_transform/0_original.ply", *cloud);
    cloud = input_cloud;

    // Define the new origin
    //Eigen::Vector4f new_origin(1.512153615099417, 3.8880719330446656, 1.4342093633104345, 1.0); // scene 0 -> trajectory at 0
    Eigen::Vector4f new_origin(tx, ty, tz, 1.0); // scene 0 -> trajectory at 0
    
    // Translate the point cloud
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -new_origin[0], -new_origin[1], -new_origin[2];
    pcl::transformPointCloud(*cloud, *cloud, transform);

    // Estimate normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.1);
    ne.compute(*normals);
    //pcl::io::savePLYFile("0_original_tran.ply", *cloud);
    std::cout << "Transformed Point Cloud: " << std::endl;
    cout<<"Saved tranfomed PCD "<<endl<<"origin to translation"<<endl<<endl;
    return cloud;
}

std::string createDirectory(const std::string& directoryName) {
    // Create the directory if it does not exist
    if (mkdir(directoryName.c_str(), 0777) == -1) {
        if (errno != EEXIST) { // Check if the error is because the folder already exists
            std::cerr << "Error creating directory: " << strerror(errno) << std::endl;
            return ""; // Return an empty string to indicate failure
        }
    } else {
        std::cout << "Directory created: " << directoryName << std::endl;
    }

    // Return the full path of the directory
    // If you need the absolute path, you'll need additional logic here
    return directoryName;
}

/*
  Function that uses the FrustumCulling filter to filter out points not possibly visible by the camera
  param: &data vector with the loaded PCDs
*/
void filter(std::vector<PCD, Eigen::aligned_allocator<PCD> > &data, float vFOV, float hFOV, float scene_number, float timestamp, 
                                                        float tx, float ty, float tz, float qx, float qy, float qz, float qw){
  cout<<endl<<endl<<"Applying Filter"<<endl;

  PointCloud::Ptr cloud_filtered (new PointCloud);

  // Create the filter  
  pcl::FrustumCulling<PointT> fc;
  // The following parameters were defined by trial and error. 
  // You can modify them to better match your expected results
  fc.setVerticalFOV (vFOV);
  fc.setHorizontalFOV (hFOV);
  fc.setNearPlaneDistance (0);
  fc.setFarPlaneDistance (20);
   
  // Define the camera pose as a rotation and translation with respect to the LiDAR pose.
  Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
  Eigen::Matrix3f rotation = Eigen::Quaternionf(qw, qx, qy, qz).toRotationMatrix(); 
  //Eigen::Matrix3f rotation = Eigen::Quaternionf(1, 0, 0, 0).toRotationMatrix();  // 0 degrees

  //Eigen::RowVector3f translation(tx, ty, tz); 
  Eigen::RowVector3f translation(0,0,0); // Assume the PCD is tranformed to new axis where 0,0,0 is the place where you need to place the camara

  // This is the most important part, it tells you in which direction to look in the Point Cloud
  camera_pose.block(0,0,3,3) = rotation; 
  camera_pose.block(3,0,1,3) = translation;
  cout<<"Camera Pose "<<endl<<camera_pose<<endl<<endl;
  fc.setCameraPose (camera_pose);

  // Go over each Point Cloud and filter it
  for (auto & d : data){
    // tranform the point cloud to set the current translation point as the origin - Note : this is done because this filter did not work when the translation point is non other than origin - could find what was the reason though
    PointCloud::Ptr transformed_cloud = transformPointCloud(d.cloud, tx, ty, tz);
    d.cloud = transformed_cloud;
    
    // Run the filter on the cloud
    PointCloud::Ptr cloud_filtered (new PointCloud);
    fc.setInputCloud (d.cloud);
    fc.filter(*cloud_filtered);
    // Update the cloud 
    d.cloud = cloud_filtered;
    // Replace the PCD file with the filtered data
    std::string new_filename = d.path  + "/" + to_string(static_cast<int>(scene_number))+ "_" + to_string(static_cast<int>(timestamp)) + ".ply";

    // re-tranform the extracted point cloud back to its original reference coordinate system - this makes the visualizations of filtered scenes align with original scene
    PointCloud::Ptr transformed_cloud_original_coord_sys = transformPointCloud(d.cloud, -tx, -ty, -tz);
    d.cloud = transformed_cloud_original_coord_sys;

    pcl::io::savePLYFileASCII (new_filename, *d.cloud);
    cout<<"Saved Filtered PCD "<<endl<<new_filename<<endl<<endl;
    //pcl::io::savePCDFileASCII (d.f_name, *d.cloud);  
  }

}

/* A very simple visualisation function to see the results of the filtering
  param cloud: the Point Cloud to visualize
*/
 pcl::visualization::PCLVisualizer::Ptr simpleVis (PointCloud::ConstPtr cloud)
{
  // Open 3D viewer and add point cloud
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<PointT> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


int main (int argc, char** argv)
{
	
  if (argc < 3){
    PCL_ERROR ("Error: Syntax is: %s <path_to_pcds>\n\n", argv[0]);
    return -1;
  }

  // Parse FOV arguments
  float vFOV = std::stof(argv[2]);
  float hFOV = std::stof(argv[3]);

  // timestamp of the tragectory
  float scene_number = std::stof(argv[4]);
  float timestamp = std::stof(argv[5]);
  //std::string timestamp = "0";

  // camera translation coordinates
  float tx = std::stof(argv[6]);
  float ty = std::stof(argv[7]);
  float tz = std::stof(argv[8]);
  //float tx = 1.51215361509941;
  //float ty = 3.88807193304466;
  //float tz = 1.43420936331043;
    
  // camara rotation quaternion values
  float qx = std::stof(argv[9]);
  float qy = std::stof(argv[10]);
  float qz = std::stof(argv[11]);
  float qw = std::stof(argv[12]);
  //float qx = 0.17498913549234;
  //float qy = 0.642008353975212;
  //float qz = -0.291806301477169;
  //float qw = 0.6870612478548739;
  
  // Get the PCD paths using the directory path received as argument then sort them
  std::vector<std::filesystem::path> files_in_directory;
  std::copy(std::filesystem::directory_iterator(argv[1]), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
  std::sort(files_in_directory.begin(), files_in_directory.end());
  
  // Load data
  std::vector<PCD, Eigen::aligned_allocator<PCD> > data;

  loadData (files_in_directory, data);

  // Check user input
  if (data.empty ())
  {
    PCL_ERROR ("Syntax is: %s path_to_pcds ", argv[0]);
    
    return (-1);
  }
  PCL_INFO ("Loaded %d Point Clouds.", (int)data.size ());

  //Filter and save the Point Clouds
  filter(data, vFOV, hFOV, scene_number, timestamp, tx, ty, tz, qx, qy, qz, qw);
}