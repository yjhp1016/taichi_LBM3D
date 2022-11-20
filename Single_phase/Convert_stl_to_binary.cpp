#include <iostream>
#include <math.h>


#include <igl/readSTL.h>
#include <igl/readOFF.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/signed_distance.h>
#include <igl/AABB.h>
#include <igl/fast_winding_number.h>
#include <igl/writeOBJ.h>

#include <numeric>
#include<fstream>
#include<sstream>
#include <chrono>

using namespace std;
using namespace chrono;

float**** alloc4float(int,int,int,int);
void free4float(float****, int, int, int, int);
int*** memory_allocation_int(int, int, int);
double*** memory_allocation_double(int, int, int);
void memory_clean_3(double*** , int, int, int);
void memory_clean_3int(int*** , int, int, int);


int main(int argc, char *argv[])
{
    
    auto start = system_clock::now();
    
    
    char stl_filename[128];
    
   
    float**** grid;
   
    int*** meshflag;
    ofstream out2;

    strcpy(stl_filename, "GenNACAXXXX.stl");



    float nx0 = -10.0;
    float ny0 = -5.0;
    float nz0 = -15.0;
    
    float nx1 = 50.0;
    float ny1 = 5.0;
    float nz1 = 15.0;

    float dx = 0.5;

    int nx,ny,nz;
    nx = ceil((nx1-nx0)/dx);
    ny = ceil((ny1-ny0)/dx);
    nz = ceil((nz1-nz0)/dx);

        

    Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd N_faces;
    try
        {
            std::ifstream rfile;
            rfile.open(stl_filename);
            igl::readSTL(rfile, V, F, N_faces);
            rfile.close();
        }
        catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        }
    // Save the mesh in OBJ format
    igl::writeOBJ("Check1.obj",V,F);   


    grid = alloc4float(nx, ny, nz,3);
    for (int k=0;k<nz;k++)
        for (int j = 0;j<ny;j++)
            for (int i=0;i<nx;i++)
            {
                grid[i][j][k][0] = nx0+dx*i;
                grid[i][j][k][1] = ny0+dx*j;
                grid[i][j][k][2] = nz0+dx*k;
            }

    cout<<"Mesh size: "<<nx<<" X "<<ny<<" X "<<nz<<endl;

    meshflag = memory_allocation_int(nx,ny,nz);


    Eigen::VectorXd sqrD;
    Eigen::VectorXi I;
    Eigen::MatrixXd C;
    Eigen::MatrixXd P;
    Eigen::MatrixXd P3;P3.resize(1,3);
    Eigen::MatrixXd S;

    igl::FastWindingNumberBVH fwn_bvh;
    igl::AABB<Eigen::MatrixXd,3> tree;
    double*** meshflag_Solid;
    //~~~~~~~~~~~~~~~~~~~
    igl::fast_winding_number(V, F, 2, fwn_bvh);
    tree.init(V,F);
    //~~~~~~~~~~~~~~~~~~~

    meshflag_Solid = memory_allocation_double(nx-1,ny-1,nz-1);

    double mid_x, mid_y, mid_z;
    for (int i=0;i<nx;i++)
        for (int j=0;j<ny;j++)
            for (int k=0;k<nz;k++)
            {

                    P3 << grid[i][j][k][0],grid[i][j][k][1], grid[i][j][k][2];
                    igl::signed_distance_fast_winding_number(P3, V, F, tree, fwn_bvh, S);
                    
                    //meshflag[i][j][k] = S(0);
                    if (S(0)>0)
                        meshflag[i][j][k] = 0;
                    else
                        meshflag[i][j][k] = 1;
                }

    cout<<"Binarilization is complete"<<endl;

    
    out2.open("binary_geometry_check.vtk");
    out2<<"# vtk DataFile Version 2.0"<<endl;
	out2<<"J.Yang Lattice Boltzmann Simulation 3D Single Phase-Solid-Geometry"<<endl;
	out2<<"ASCII"<<endl;
	out2<<"DATASET STRUCTURED_POINTS"<<endl;
	out2<<"DIMENSIONS         "<<nx<<"         "<<ny<<"         "<<nz<<endl;
	out2<<"ORIGIN 0 0 0"<<endl;
	out2<<"SPACING 1 1 1"<<endl;
	out2<<"POINT_DATA     "<<nx*ny*nz<<endl;
	out2<<"SCALARS sample_scalars float"<<endl;
	out2<<"LOOKUP_TABLE default"<<endl;


    for (int k=0;k<nz;k++)
        for (int j = 0;j<ny;j++)
            for (int i=0;i<nx;i++)
            {
                out2<<meshflag[i][j][k]<<" ";
                
            }	

    out2.close();


    
}



float**** alloc4float(int nx,int ny,int nz,int ns)
{
    int i,j,k;

    float**** result;

    result = new float***[nx];

    for (i=0;i<nx;i++)
    {
        result[i] = new float**[ny];
        for (int j=0;j<ny;j++)
            result[i][j] = new float*[nz];
    }

    result[0][0][0] = new float[nx*ny*nz*ns];

    for (int k=1;k<nz;k++)
        result[0][0][k] = result[0][0][k-1] + ns;

    for (int j=1;j<ny;j++)
        result[0][j][0] = result[0][j-1][0] + ns*nz;

    for (int i=1;i<nx;i++)
        result[i][0][0] = result[i-1][0][0] + ns*nz*ny;    
    


    for (int i=0;i<nx;i++)
    {
        if (i>0)
            result[i][0][0] = result[i-1][0][0] + ns*nz*ny;
        for (int j=0;j<ny;j++)
        {
            if (j>0)
                result[i][j][0] = result[i][j-1][0] + ns*nz;

            for (int k=1;k<nz;k++)
                result[i][j][k] = result[i][j][k-1] + ns;
        }
    }

    return result;
}


void free4float(float**** pdel,int nx, int ny, int nz, int ns)
{
    delete [] pdel[0][0][0];

    for (int i=0;i<nx;i++)
    {
        for (int j=0;j<ny;j++)
            delete [] pdel[i][j];
        delete [] pdel[i];
    }

    delete [] pdel;
    
}


int*** memory_allocation_int(int size_x, int size_y, int size_z)
{
	int*** name_int;

	int NX = size_x - 1;
    	int NY = size_y - 1;
    	int NZ = size_z - 1;

	name_int = new int**[size_x];
      	
	
	
	for (int i=0;i<size_x;i++)				
		name_int[i]=new int*[size_y];

	name_int[0][0]=new int[size_x*size_y*size_z];

	
 	for (int i=1;i<size_y;i++)
               name_int[0][i]=name_int[0][i-1]+size_z;
       
       	for (int i=1;i<size_x;i++)
       	{
               name_int[i][0]=name_int[i-1][0]+size_y*size_z;
               for (int j=1;j<size_y;j++)
                       name_int[i][j]=name_int[i][j-1]+size_z;
       	}	
	
      
      	for(int k=0 ; k<=NZ ; k++)
	for(int j=0 ; j<=NY ; j++)
	for(int i=0 ; i<=NX ; i++)
		name_int[i][j][k]=0;
      
	return name_int;
}

void memory_clean_3int(int*** pointer_clean, int size_x, int size_y, int size_z)
{
	
	delete [] pointer_clean[0][0];
		for (int i=0;i<size_x;i++)
			delete [] pointer_clean[i];
		delete [] pointer_clean;
}


double*** memory_allocation_double(int size_x, int size_y, int size_z)
{
	double*** name_double;

	int NX = size_x - 1;
    	int NY = size_y - 1;
    	int NZ = size_z - 1;

	name_double = new double**[size_x];
      	
	
	
	for (int i=0;i<size_x;i++)				
		name_double[i]=new double*[size_y];

	name_double[0][0]=new double[size_x*size_y*size_z];

	
 	for (int i=1;i<size_y;i++)
               name_double[0][i]=name_double[0][i-1]+size_z;
       
       	for (int i=1;i<size_x;i++)
       	{
               name_double[i][0]=name_double[i-1][0]+size_y*size_z;
               for (int j=1;j<size_y;j++)
                       name_double[i][j]=name_double[i][j-1]+size_z;
       	}	
	
      
      	for(int k=0 ; k<=NZ ; k++)
	for(int j=0 ; j<=NY ; j++)
	for(int i=0 ; i<=NX ; i++)
		name_double[i][j][k]=0;
      
	return name_double;
}

void memory_clean_3(double*** pointer_clean, int size_x, int size_y, int size_z)
{
	
	delete [] pointer_clean[0][0];
		for (int i=0;i<size_x;i++)
			delete [] pointer_clean[i];
		delete [] pointer_clean;
}
