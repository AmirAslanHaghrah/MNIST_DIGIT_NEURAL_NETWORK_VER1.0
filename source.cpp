#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <math.h>
#include <ppl.h>

#define Train_Set_Size 1000
#define N 50
#define epsilon 0.001
#define epoch 0.01

#define images_number_of_rows 28
#define images_number_of_columns 28
#define images_pixel_size 784
#define lambda 1
#define digit_count 10

double c[N] = {};
double W[N][images_pixel_size] = {};
double V[digit_count][N] = {};
double b[digit_count] = {};
double err[Train_Set_Size] = {};

double sigmoid(double x) {
	return (1.0f / (1.0f + std::exp(-x)));
}

double* f_theta(unsigned char* x) {
	double result[digit_count];
	double W_temp[N] = {};

	for (int i = 0; i < N; i++) {
		for (int k = 0; k < images_pixel_size; k++) {
			W_temp[i] += W[i][k] * (int)x[k];
		}
	}

	for (int k = 0; k < digit_count; k++) {
		for (int i = 0; i < N; i++) {
			result[k] += V[k][i] * sigmoid(c[i] + W_temp[i]);
		}
	}

	for (int k = 0; k < digit_count; k++) {
		if (result[k] > 0) {
			result[k] = 1;
		}
		else {
			result[k] = 0;
		}
	}
	return result;
}


int l = 0;
inline void train(unsigned char* x, int* y) {
	double W_temp[N] = {};

	for (int i = 0; i < N; i++) {
		for (int k = 0; k < images_pixel_size; k++) {
			W_temp[i] += W[i][k] * (int)x[k];
		}
	}

	double error[digit_count] = {};
	for (int k = 0; k < digit_count; k++) {
		error[k] = (f_theta(x)[k] - y[k]);
	}

	//std::cout << error << "\n";

	//err[l] = error[0];
	//if (l < Train_Set_Size) {
	//	l++;
	//}
	//else {
	//	l = 0;
	//}

	for (int k = 0; k < digit_count; k++) {
		for (int i = 0; i < N; i++) {		
			for (int j = 0; j < images_pixel_size; j++) {
				W[i][j] = W[i][j] - epsilon * (2 * lambda * W[i][j] +  2.0f * error[k] * V[k][i] * (int)x[j] * (1 - sigmoid(c[i] + W_temp[i])) * sigmoid(c[i] + W_temp[i]));
			}
				V[k][i] = V[k][i] - epsilon * (2 * lambda * V[k][i] + 2.0f * error[k] * sigmoid(c[i] + W_temp[i]));
				c[i] = c[i] - epsilon * 2.0f * error[k] * V[k][i] * (1 - sigmoid(c[i] + W_temp[i])) * sigmoid(c[i] + W_temp[i]);			
		}
		b[k] = b[k] - epsilon * 2.0f * error[k];
	}


	//int start_s = clock();

	//for (int i = 0; i < images_pixel_size; i++) {
	//	std::cout << x[i] << " ";
	//	if (i % images_number_of_columns == 0) {
	//		std::cout << "\n";
	//	}
	//}
	//std::cout << "\n############################################\n";
	//std::cout << y << "\n";
	//int stop_s = clock();
	//std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << std::endl;

}




int main() {
	int trainlabels_MagicNumber;
	int trainlabels_Count;
	unsigned char* trainlabels;

	int trainimages_MagicNumber;
	int trainimages_Count;
	//int images_number_of_rows;
	//int images_number_of_columns;
	//int images_pixel_size;
	unsigned char* trainimages;


	std::streampos trainlabels_size;
	char* trainlabels_memblock;
	std::fstream trainlabelsFile("data\\train-labels.idx1-ubyte", std::ios::in | std::ios::binary | std::ios::ate);
	if (!trainlabelsFile.is_open()) {
		std::cout << "Unable to open labels file. \n";
		system("pause");
		return -1;
	}
	trainlabels_size = trainlabelsFile.tellg();
	trainlabels_memblock = new char[trainlabels_size];
	trainlabelsFile.seekg(0, std::ios::beg);
	trainlabelsFile.read(trainlabels_memblock, trainlabels_size);
	trainlabelsFile.close();
	trainlabels_MagicNumber = (unsigned char)trainlabels_memblock[0] << 24 | (unsigned char)trainlabels_memblock[1] << 16 | (unsigned char)trainlabels_memblock[2] << 8 | (unsigned char)trainlabels_memblock[3];
	trainlabels_Count = (unsigned char)trainlabels_memblock[4] << 24 | (unsigned char)trainlabels_memblock[5] << 16 | (unsigned char)trainlabels_memblock[6] << 8 | (unsigned char)trainlabels_memblock[7];
	trainlabels = (unsigned char *)trainlabels_memblock + 8;


	std::streampos trainimages_size;
	char* trainimages_memblock;
	std::fstream trainimagesFile("data\\train-images.idx3-ubyte", std::ios::in | std::ios::binary | std::ios::ate);
	if (!trainimagesFile.is_open()) {
		std::cout << "Unable to open images file. \n";
		system("pause");
		return -1;
	}
	trainimages_size = trainimagesFile.tellg();
	trainimages_memblock = new char[trainimages_size];
	trainimagesFile.seekg(0, std::ios::beg);
	trainimagesFile.read(trainimages_memblock, trainimages_size);
	trainimagesFile.close();
	trainimages_MagicNumber = (unsigned char)trainimages_memblock[0] << 24 | (unsigned char)trainimages_memblock[1] << 16 | (unsigned char)trainimages_memblock[2] << 8 | (unsigned char)trainimages_memblock[3];
	trainimages_Count = (unsigned char)trainimages_memblock[4] << 24 | (unsigned char)trainimages_memblock[5] << 16 | (unsigned char)trainimages_memblock[6] << 8 | (unsigned char)trainimages_memblock[7];
	//images_number_of_rows = (unsigned char)images_memblock[8] << 24 | (unsigned char)images_memblock[9] << 16 | (unsigned char)images_memblock[10] << 8 | (unsigned char)images_memblock[11];
	//images_number_of_columns = (unsigned char)images_memblock[12] << 24 | (unsigned char)images_memblock[13] << 16 | (unsigned char)images_memblock[14] << 8 | (unsigned char)images_memblock[15];
	//images_pixel_size = images_number_of_rows * images_number_of_columns;
	trainimages = (unsigned char *)trainimages_memblock + 16;

	if (trainlabels_Count != trainimages_Count) {
		std::cout << "Labels file and Images file count mismatch \n";
		return -1;
	}

	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < images_pixel_size; j++) {
			W[i][j] = 2.0f * rand() / RAND_MAX - 1;
		}
		for (int k = 0; k < digit_count; k++) {
			V[k][i] = 2.0f * rand() / RAND_MAX - 1;
			b[k] = 2.0f * rand() / RAND_MAX - 1;
		}
		c[i] = 2.0f * rand() / RAND_MAX - 1;
	}



	for (int j = 0; j < epoch; j++) {
//#pragma omp parallel for
		for (int n = 0; n < Train_Set_Size; n++) {
				int imagelabel[digit_count] = {};
				imagelabel[(int)trainlabels[n]] = 1;
				train(&trainimages[n * images_pixel_size], imagelabel);
				std::cout << "step :" << n << "\n";
		}
		std::cout << "Epoch :" << j << "\n";
	}

	delete[] trainlabels_memblock;
	delete[] trainimages_memblock;

	std::cout << "#################    Training FINISH    ##############" << "\n";

	int testlabels_MagicNumber;
	int testlabels_Count;
	unsigned char* testlabels;

	int testimages_MagicNumber;
	int testimages_Count;
	//int images_number_of_rows;
	//int images_number_of_columns;
	//int images_pixel_size;
	unsigned char* testimages;


	std::streampos testlabels_size;
	char* testlabels_memblock;
	std::fstream testlabelsFile("data\\t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary | std::ios::ate);
	if (!testlabelsFile.is_open()) {
		std::cout << "Unable to open labels file. \n";
		system("pause");
		return -1;
	}
	testlabels_size = testlabelsFile.tellg();
	testlabels_memblock = new char[testlabels_size];
	testlabelsFile.seekg(0, std::ios::beg);
	testlabelsFile.read(testlabels_memblock, testlabels_size);
	testlabelsFile.close();
	testlabels_MagicNumber = (unsigned char)testlabels_memblock[0] << 24 | (unsigned char)testlabels_memblock[1] << 16 | (unsigned char)testlabels_memblock[2] << 8 | (unsigned char)testlabels_memblock[3];
	testlabels_Count = (unsigned char)testlabels_memblock[4] << 24 | (unsigned char)testlabels_memblock[5] << 16 | (unsigned char)testlabels_memblock[6] << 8 | (unsigned char)testlabels_memblock[7];
	testlabels = (unsigned char *)testlabels_memblock + 8;


	std::streampos testimages_size;
	char* testimages_memblock;
	std::fstream testimagesFile("data\\t10k-images.idx3-ubyte", std::ios::in | std::ios::binary | std::ios::ate);
	if (!testimagesFile.is_open()) {
		std::cout << "Unable to open images file. \n";
		system("pause");
		return -1;
	}
	testimages_size = testimagesFile.tellg();
	testimages_memblock = new char[testimages_size];
	testimagesFile.seekg(0, std::ios::beg);
	testimagesFile.read(testimages_memblock, testimages_size);
	testimagesFile.close();
	testimages_MagicNumber = (unsigned char)testimages_memblock[0] << 24 | (unsigned char)testimages_memblock[1] << 16 | (unsigned char)testimages_memblock[2] << 8 | (unsigned char)testimages_memblock[3];
	testimages_Count = (unsigned char)testimages_memblock[4] << 24 | (unsigned char)testimages_memblock[5] << 16 | (unsigned char)testimages_memblock[6] << 8 | (unsigned char)testimages_memblock[7];
	//images_number_of_rows = (unsigned char)images_memblock[8] << 24 | (unsigned char)images_memblock[9] << 16 | (unsigned char)images_memblock[10] << 8 | (unsigned char)images_memblock[11];
	//images_number_of_columns = (unsigned char)images_memblock[12] << 24 | (unsigned char)images_memblock[13] << 16 | (unsigned char)images_memblock[14] << 8 | (unsigned char)images_memblock[15];
	//images_pixel_size = images_number_of_rows * images_number_of_columns;
	testimages = (unsigned char *)testimages_memblock + 16;

	if (testlabels_Count != testimages_Count) {
		std::cout << "Labels file and Images file count mismatch \n";
		return -1;
	}




	int r = 0;
	int T = testlabels_Count;
	for (int t = 0; t < T; t++) {
		if ((int)f_theta(&testimages[(t) * images_pixel_size])[(int)testlabels[(t)]] == 1) {
			r++;
		}
	}
	std::cout << r << "\n";
	std::cout << r * 100.0f / T << " %" << "\n";




	for (int n = 0; n < 10; n++) {
		for (int i = n * images_pixel_size; i < (n + 1) * images_pixel_size; i++) {
			std::cout << testimages[i] << " ";
			if (i % images_number_of_columns == 0) {
				std::cout << "\n";
			}
		}
		std::cout << "\n############################################\n";
		std::cout << "Label : " << (int)testlabels[(n)] << "\n";
		for (int j = 0; j < digit_count; j++) {
			if (f_theta(&testimages[(n)* images_pixel_size])[j] == 1) {
				std::cout << "Result : " << j << "\n";	
			}
		}
	}
	
	
	
	
	//int start_s = clock();
	//int stop_s = clock();
	//std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << std::endl;



	delete[] testlabels_memblock;
	delete[] testimages_memblock;


















	//std::vector<float> x;

	//for (int i = 0; i < l; i++) {
	//	x.push_back(i);
	//}

	//FILE * gp = _popen("gnuplot", "w");
	//fprintf(gp, "set terminal wxt size 600,400 \n");
	//fprintf(gp, "set grid \n");
	//fprintf(gp, "set title '%s' \n", "f(x) = sin (x)");
	//fprintf(gp, "set style line 1 lt 3 pt 7 ps 0.1 lc rgb 'green' lw 1 \n");
	//fprintf(gp, "set style line 2 lt 3 pt 7 ps 0.1 lc rgb 'red' lw 1 \n");
	//fprintf(gp, "plot '-' w p ls 1, '-' w p ls 2 \n");

	//for (int k = 0; k < x.size(); k++) {
	//	fprintf(gp, "%f %f \n", x[k], err[k]);
	//}
	//fprintf(gp, "e\n");

	//for (int k = 0; k < x.size(); k++) {
	//	fprintf(gp, "%f %f \n", x[k], err[k]);
	//}
	//fprintf(gp, "e\n");

	//fflush(gp);

	system("pause");
	//_pclose(gp);
	return 0;
}
