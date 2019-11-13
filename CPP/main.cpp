#include <iostream>
#include <sstream>

#include <pcl/io/pcd_io.h> // Reading in point clouds
#include <pcl/features/normal_3d.h> // Normal estimation
#include <pcl/point_types.h> // Basic types PointXYZRGBA etc.
#include <pcl/features/vfh.h> // VFH estimation

#include "good.cpp"

using namespace std;
using namespace pcl;

typedef pcl::PointXYZRGBA PointT;

enum ShapeDescriptor { VFH, GOOD_5, GOOD_15 }; // VFH = 0, GOOD_5 = 1, GOOD_15 = 2

int calculateGOODHistogram(boost::shared_ptr<PointCloud<PointT> > cloud, unsigned int number_of_bins, float threshold) {
    // Setup the GOOD descriptor
    GOODEstimation<PointT> GOOD_descriptor(number_of_bins, threshold);

    // Provide the original point cloud
    GOOD_descriptor.setInputCloud(cloud);

    // Compute GOOD descriptor for the given pointcload
    std::vector< float > object_description;
    GOOD_descriptor.compute(object_description);

    // Print it so it can be read as python array
    cout << '[';
    for (size_t i = 0; i < object_description.size() - 1; ++i)
        std::cout << object_description.at(i) << ',';
    std::cout << object_description.back() << "]" << std::endl;
}

int estimateViewpointFeatureHistogram(boost::shared_ptr<PointCloud<PointT> > cloud,  float normal_estimation_radius)
{
	// Compute normals
	pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>);
	NormalEstimation<PointT, Normal> normal_estimation;
	normal_estimation.setInputCloud (cloud);
	normal_estimation.setSearchMethod (kdtree);
	normal_estimation.setRadiusSearch ( normal_estimation_radius/*0.05*/);
	PointCloud<Normal>::Ptr normal (new PointCloud< Normal>);
	normal_estimation.compute (*normal);

	// Create the VFH estimation class, and pass the input dataset+normals to it
	pcl::VFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud (cloud);
	vfh.setInputNormals (normal);
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
	vfh.setSearchMethod (tree);
 	pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
	vfh.compute (*vfhs);

    size_t vfhs_size = sizeof(vfhs->points.at(0).histogram) / sizeof(float);
    // Push the representation to the object_representation container
    cout << '[';
    for (size_t i = 0; i < vfhs_size ; ++i)
        cout << vfhs->points.at(0).histogram[i] << ((i == vfhs_size - 1) ? "" : ", ");
    cout << ']';
}

int main (int argc, const char *argv[])
{
    std::string pcd_file_address = argv[1];
    unsigned int descriptor_type = 0;
    if (argc > 2) {
        istringstream convertStream(argv[2]);
        convertStream >> descriptor_type;
    }

    boost::shared_ptr<PointCloud<PointT>> target_pc (new PointCloud<PointT>);

    if (io::loadPCDFile <PointT> (pcd_file_address.c_str(), *target_pc) == -1)
    {
        cout << "\t\t[-]-Could not read given object %s :" << pcd_file_address.c_str();
        return(0);
    }

    float normal_estimation_radius = 0.03; // used in VFH descriptor
    switch (descriptor_type) {
        case VFH:
            estimateViewpointFeatureHistogram(target_pc, normal_estimation_radius);
            break;
        case GOOD_5:
            calculateGOODHistogram(target_pc, 5, 0.0015);
            break;
        case GOOD_15:
            calculateGOODHistogram(target_pc, 15, 0.0015);
            break;
    }

    return 0;
}