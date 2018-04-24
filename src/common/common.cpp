
#include "common.h"

void PadMatrix(std::vector<int> &v, int M, int N) {

	int NP = N+2; int MP = M+2;

	v.resize(NP*MP);

	for (int i = M-1; i >= 0; i--) {
		for (int j = N-1; j >= 0; j--) {
			v[(i+1)*(NP) + j+1] = v[i*N+j];
		}
	}

	for (int i = 0; i < MP; i++) { v[i*(NP) + 0] = 0; v[i*(NP) + NP-1] = 0; }
	for (int j = 0; j < NP; j++) { v[j] = 0; }
}

void PadMatrix(std::vector<int> &v) {
	PadMatrix(v, std::sqrt(v.size()), std::sqrt(v.size()));
}

void UnPadMatrix(std::vector<int> &v) {
	UnPadMatrix(v, std::sqrt(v.size()), std::sqrt(v.size()));
}

void UnPadMatrix(std::vector<int> &v, int M, int N) {
	int NP = N-2; int MP = M-2;

	for (int i = 0; i < MP; i++) {
		for (int j = 0; j < NP; j++) {
			v[i*NP + j] = v[(i+1)*N+(j+1)];
		}
	}

	v.resize(NP*MP);
}


void PrintOpenCLInfo()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	int platform_id = 0;
	int device_id = 0;

	std::cout << "Number of Platforms: " << platforms.size() << std::endl;

	for( std::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it) {
		cl::Platform platform(*it);

		std::cout << "Platform ID: " << platform_id++ << std::endl;
		std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
		std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

		for(std::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2){
			cl::Device device(*it2);

			std::cout << "\tDevice " << device_id++ << ": " << std::endl;
			std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
			std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
			std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
			std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
			std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
			//std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_KERNEL_WORK_GROUP_SIZE>() << std::endl;
			//std::cout << "\t\tDevice Global Memory: " << device.getInfo<>() << std::endl;
			std::cout << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
			std::cout << "\t\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
			std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
			std::cout << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
		}
		std::cout<< std::endl;
	}
}


void SaveMatrix(const std::vector<int> &tiling, std::string filename) {
	SaveMatrix(tiling, std::sqrt(tiling.size()), std::sqrt(tiling.size()), filename );
}

void SaveMatrix(const std::vector<double> &tiling, std::string filename) {
	SaveMatrix(tiling, std::sqrt(tiling.size()), std::sqrt(tiling.size()), filename );
}

void PrintMatrix(const std::vector<int> &tiling ) {
	PrintMatrix(tiling, sqrt(tiling.size()), sqrt(tiling.size()));
}

void SaveMatrix(const std::vector<double> &tiling, int M, int N, std::string filename) {

	std::ofstream outputFile(filename.c_str());

	std::string fs = "test.txt";

	// save the domain to text file
	for (int i=0; i<M; ++i){
		for(int j=0; j<N; ++j) {
			outputFile<<tiling[i*N+j]<<" ";
		}
		outputFile<<"\n";
	}
	outputFile.close();
}

void SaveMatrix(const std::vector<int> &tiling, int M, int N, std::string filename) {

	std::ofstream outputFile(filename.c_str());

	std::string fs = "test.txt";

	for (int i=0; i<M; ++i){
		for(int j=0; j<N; ++j) {
			outputFile<<tiling[i*N+j]<<" ";
		}
		outputFile<<"\n";
	}
	outputFile.close();
}



void PrintMatrix(const std::vector<int> &tiling, int M, int N) {

	for (int i=0; i<M; ++i){
		for(int j=0; j<N; ++j) {
			if ( tiling[i*N+j] < infty)
				std::cout<<tiling[i*N+j]<<" ";
			else
				std::cout<<"  ";
		}
		std::cout<<"\n";
	}
}


std::vector<int> LoadMatrix(std::string filename) {
	std::vector<int> mat;
	std::ifstream is(filename.c_str());     // open file
	if(!is.is_open()) {
		std::cout << "Cannot open file: " << filename << std::endl;
		exit(1);
	}
	char c;
	int t = 0;

	while (is.get(c) ) {
		if ( c > 47 && c < 58 ) {
			t = c-48;
			while ( is.get(c) && c > 47 && c < 58 ) { t = 10*t + c-48; }
			mat.push_back(t);
		}
	}

	is.close();
	return mat;
}

cl::Program LoadCLProgram(cl::Context context, std::vector<cl::Device> devices,std::string input) {

	std::ifstream sourceFile(input.c_str());

	if (!sourceFile.is_open()) {
		std::cout << "Cannot open file: " << input << std::endl;
		exit(1);
	}

	std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

	cl_int err;

	cl::Program program = cl::Program(context, source, &err);

	try {
		err = program.build(devices);
	} catch (cl::Error &e) {
		if (e.err() != CL_SUCCESS) {
			if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
				std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
				std::cout << "compile error:" << std::endl;
				std::cout << str << std::endl;
			} else {
				std::cout << "build error but not program failure err:" << e.err() << " " << e.what() << std::endl;
			}
		}
		throw e;
	}

	return program;
}


