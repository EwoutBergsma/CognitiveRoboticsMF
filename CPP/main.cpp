#include <iostream>

#include <pcl/io/pcd_io.h> // Reading in point clouds
#include <pcl/features/normal_3d.h> // Normal estimation
#include <pcl/point_types.h> // Basic types PointXYZRGBA etc.
#include <pcl/features/vfh.h> // VFH estimation

using namespace std;
using namespace pcl;

typedef pcl::PointXYZRGBA PointT;


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
    for (size_t i = 0; i < vfhs_size ; i++)
        cout << vfhs->points.at(0).histogram[i] << ((i == vfhs_size - 1) ? "" : ", ");
    cout << ']';
}

int main (int argc, const char *argv[])
{
    std::string pcd_file_address = argv[1];

//    std::string pcd_file_address;
//    pcd_file_address = "/home/gitaar9/AI/COR/CPP_try/datasets/pliers_Category/pliers_object_1.pcd";

    boost::shared_ptr<PointCloud<PointT>> target_pc (new PointCloud<PointT>);

    if (io::loadPCDFile <PointT> (pcd_file_address.c_str(), *target_pc) == -1)
    {
        cout << "\t\t[-]-Could not read given object %s :" << pcd_file_address.c_str();
        return(0);
    }

    float normal_estimation_radius = 0.03; // used in VFH descriptor
    estimateViewpointFeatureHistogram(target_pc, normal_estimation_radius);

    return (0);
}