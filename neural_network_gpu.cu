#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>

#define ID(i,j,k) (( i * k ) + j)
#define ARG(i,j) <<< dim3(i,j,1), dim3(1,1,1) >>>

// computes the sum of matrices:  c = (a + b)
// a : n->k, b : n->k, c : n->k
// call it with n x k blocks
__global__ void matrixSum(float* a, float* b, size_t const n, size_t const k, float* c){
	size_t i{ blockIdx.x };
	size_t j{ blockIdx.y };
	if ((i >= n) || (j >= k))
		return;
	c[ID(i, j, k)] = (a[ID(i, j, k)] + b[ID(i, j, k)]);
}

// computes the product of matrices:  c = (a.b)
// a : n->k, b : k->p, c : n->p
// call it with n x p blocks
// we assume c != a, c != b
__global__ void matrixMultiply(float* a, float* b, size_t const n, size_t const k, size_t const p, float* c){
	assert((a != c) && (b != c));
	size_t i{ blockIdx.x };
	size_t j{ blockIdx.y };
	if ((i >= n) || (j >= p))
		return;
	c[ID(i, j, p)] = 0.0f;
	for (size_t z = 0; z<k; ++z)
		c[ID(i, j, p)] += (a[ID(i, z, k)] * b[ID(z, j, p)]);
}

// computes the product of matrix a and scalar b:  c = b*a
// a : n->k, b : 1->1
// call it with n x k blocks
__global__ void matrixScalarMultiply(float* a, float b, size_t const n, size_t const k, float* c){
	size_t i{ blockIdx.x };
	size_t j{ blockIdx.y };
	if ((i >= n) || (j >= k))
		return;
	c[ID(i, j, k)] = b * a[ID(i, j, k)];
}

// computes c = (a * b), where * is elementwise multiplication
// a : n->k, b : n->k, c : n->k
// call it with n x k blocks
__global__ void matrixElementwiseMultiply(float* a, float* b, size_t const n, size_t const k, float* c){
	size_t i{ blockIdx.x };
	size_t j{ blockIdx.y };
	if ((i >= n) || (j >= k))
		return;
	c[ID(i, j, k)] = (a[ID(i, j, k)] * b[ID(i, j, k)]);
}

// computes the sigmoid function: c = ( 1 / (1 + e^(-a)) ) elementwise on matrices
// a : n -> k, c : n ->k
// call it with n x p blocks
__global__ void sigmoid(float* a, size_t const n, size_t const k, float* c){
	size_t i{ blockIdx.x };
	size_t j{ blockIdx.y };
	if ((i >= n) || (j >= k))
		return;
	c[ID(i, j, k)] = 1.0f / (1.0f + expf((-1.0f) * a[ID(i, j, k)]));
}

// computes transpose of a:n->p
// returns it in c:p->n
// call it with n x p blocks
// we assume a != c
__global__ void transpose(float* a, size_t n, size_t p, float* c){
	assert(a != c);
	size_t i{ blockIdx.x };
	size_t j{ blockIdx.y };
	if ((i >= n) || (j >= p))
		return;
	c[ID(j, i, n)] = a[ID(i, j, p)];
}

// return the zero matrix c:n->k
// call it with n x k blocks
__global__ void zeroMatrix(float* c, size_t n, size_t k){
	size_t i{ blockIdx.x };
	size_t j{ blockIdx.y };
	if ((i >= n) || (j >= k))
		return;
	c[ID(i, j, k)] = 0.0f;
}

// *adds* x to the diagonal of a:n->n
// call it with n x 1 blocks
__global__ void addDiagonal(float* a, size_t n, float x){
	size_t i{blockIdx.x};
	if (i >= n)
		return;
	a[ID(i, i, n)] += x;
}

// zeroes out column j, except for its j'th row
// using rank-preserving transformations
// c is the extension matrix of a
// call it with n x n blocks
__global__ void column_zeroer(float *a, float *c, size_t n, size_t j){
	size_t i{ blockIdx.x };
	size_t k{ blockIdx.y };
	if ((j == i) || ((i >= n) || (k >= n)))
		return;	
	float ratio{ a[ID(i, j, n)] / a[ID(j, j, n)] };
	a[ID(i, k, n)] -= (ratio * a[ID(j, k, n)]); // unstable
	c[ID(i, k, n)] -= (ratio * c[ID(j, k, n)]);
}

// divides row j with divisor
// it is a rank-preserving transformation
// c is the extension matrix of a
// call it with n x 1 blocks
__global__ void row_divider(float *a, float *c, size_t n, size_t j){
	size_t k{ blockIdx.x };
	if (k >= n)
		return;
	float divisor{ a[ID(j, j, n)] };
	a[ID(j, k, n)] /= divisor; // unstable
	c[ID(j, k, n)] /= divisor;
}

// computes the inverse of a:n->n
// using rank-preserving transformations
// numerically unstable
// returns the inverse in c:n->n
// we assume a != c
inline void inverse(float* a, size_t n, float* c, float damping){
	assert(a != c);
	zeroMatrix ARG(n, n) (c, n, n);
	addDiagonal ARG(n, 1) (c, n, 1.0f);
	addDiagonal ARG(n, 1) (a, n, damping); // damping
	for (size_t j = 0; j < n; ++j)
		column_zeroer ARG(n, n) (a, c, n, j);
	for (size_t j = 0; j < n; ++j)
		row_divider ARG(n, 1) (a, c, n, j);
}

inline void printCudaMatrix(float* m, size_t n, size_t k){
	float *h_m = new float[n*k];
	cudaMemcpy(h_m, m, sizeof(float)*n*k, cudaMemcpyDeviceToHost);
	for (size_t j = 0; j < n; ++j){
		for (size_t i = 0; i < k; ++i)
			std::cerr << h_m[ID(j, i, k)] << " ";
		std::cerr << std::endl;
	}
	std::cerr << std::endl;
	delete[] h_m;
}

// traning the network
// -------------------
// input matrix :  x : d->n, target matrix :  t : c->n
// 
// the set of learning samples consists of the input-output pairs: (x_i, t_i),
// where x_i is the i'th column of x, t_i is the i'th column of t, respectively.
// 
// weight matrices of layers:  w : d->l, u : l->c
// 
// rho : ( initial ) learning rate 
//
// number_of_iterations : number of iterations to be performed
//
inline void train(size_t d, size_t l, size_t c, size_t n, float* x, float* t, float* w, float* u, float rho, size_t number_of_iterations) {
	float *w_t, *h, *t_t, *temp_lxl, *temp_lxl_inv, *temp_lxn_1, *temp_lxn_2, *u_t, *e; // device pointers
	float const damping{ 0.0f };
	// memory allocation on device
	cudaMalloc(&w_t, sizeof(float) * l * d);
	cudaMalloc(&h, sizeof(float) * l * n);
	cudaMalloc(&u_t, sizeof(float) * c * l);
	cudaMalloc(&t_t, sizeof(float) * c * n);
	cudaMalloc(&temp_lxl, sizeof(float) * l * l);
	cudaMalloc(&temp_lxl_inv, sizeof(float) * l * l);
	cudaMalloc(&temp_lxn_1, sizeof(float) * l * n);
	cudaMalloc(&temp_lxn_2, sizeof(float) * l * n);
	cudaMalloc(&e, sizeof(float) * d * l);

	// computing t_t
	transpose ARG(c,n) (t, c, n, t_t);
	// now t_t = Transpose(t)

	for (size_t j = 0; j < number_of_iterations; ++j) {

		// computing w_t
		transpose ARG(d,l) (w, d, l, w_t);
		// now w_t = Transpose(w)

		// computing h
		matrixMultiply ARG(l,n) (w_t, x, l, d, n, h);
		// now h = Transpose(w).x

		sigmoid ARG(l,n) (h, l, n, h);
		// now h = sigmoid(Transpose(w).x)

		// computing h_t
		transpose ARG(l,n) (h, l, n, temp_lxn_1);
		// now temp_lxn_1 = Transpose(h)

		// computing u
		matrixMultiply ARG(l,l) (h, temp_lxn_1, l, n, l, temp_lxl);
		// now temp_lxl = h.Transpose(h)

		// printCudaMatrix(temp_lxl, l, l);
		inverse(temp_lxl, l, temp_lxl_inv, damping);
		// printCudaMatrix(temp_lxl_inv, l, l);

		// now temp_lxl = inverse(h.Transpose(h))
		matrixMultiply ARG(l,c) (h, t_t, l, n, c, u_t);
		// now u = h.Transpose(t)
		matrixMultiply ARG(l,c) (temp_lxl_inv, u_t, l, l, c, u);
		// now u = inverse(h.Transpose(h)).h.Transpose(t)

		transpose ARG(l, c) (u, l, c, u_t);
		// now u_t = Transpose(u)

		// computing e
		matrixMultiply ARG(l,l) (u, u_t, l, c, l, temp_lxl);
		matrixMultiply ARG(l,n) (temp_lxl, h, l, l, n, temp_lxn_1);
		matrixMultiply ARG(l,n) (u, t, l, c, n, temp_lxn_2);
		matrixScalarMultiply ARG(l,n) (temp_lxn_2, -1.0f, l, n, temp_lxn_2);
		matrixSum ARG(l,n) (temp_lxn_1, temp_lxn_2, l, n, temp_lxn_1);
		// now temp_lxn_1 contains the third part
		zeroMatrix ARG(l,n) (temp_lxn_2, l, n);
		matrixSum ARG(l,n) (temp_lxn_2, h, l, n, temp_lxn_2);
		matrixScalarMultiply ARG(l,n) (temp_lxn_2, -1.0f, l, n, temp_lxn_2);
		addDiagonal ARG((l < n) ? l : n, (l < n) ? l : n) (temp_lxn_2, (l < n) ? l : n, 1.0f);
		// now temp_lxn_2 contains the second part
		// and h contains the first part
		matrixElementwiseMultiply ARG(l, n) (h, temp_lxn_2, l, n, temp_lxn_2);
		matrixElementwiseMultiply ARG(l, n) (temp_lxn_2, temp_lxn_1, l, n, temp_lxn_1);
		transpose ARG(l, n) (temp_lxn_1, l, n, temp_lxn_2);
		matrixScalarMultiply ARG(n, l) (temp_lxn_2, (rho*(-2.0f)), n, l, temp_lxn_2);
		matrixMultiply ARG(d, l) (x, temp_lxn_2, d, n, l, e);
		// now e is computed
		matrixSum ARG(d, l) (w, e, d, l, w);

	}

	// freeing memory on device
	cudaFree(w_t);
	cudaFree(h);
	cudaFree(u_t);
	cudaFree(t_t);
	cudaFree(temp_lxl);
	cudaFree(temp_lxl_inv);
	cudaFree(temp_lxn_1);
	cudaFree(temp_lxn_2);
	cudaFree(e);

}

// processing the network
// ----------------------
// input matrix :  x : d->n, output matrix :  y : c->n, 
// 
// weight matrices of layers:  w : d->l, u : l->c
//
inline void compute(size_t d, size_t l, size_t c, size_t n, float* x, float* w, float* u, float* y){
	float *w_t, *h, *u_t; // device pointers

	// memory allocation on device
	cudaMalloc(&w_t, sizeof(float) * l * d);
	cudaMalloc(&h, sizeof(float) * l * n);
	cudaMalloc(&u_t, sizeof(float) * c * l);

	// computing w_t
	transpose ARG(d, l) (w, d, l, w_t);
	// now w_t = Transpose(w)

	// computing h
	matrixMultiply ARG(l, n) (w_t, x, l, d, n, h);
	// now h = Transpose(w).x
	sigmoid ARG(l, n) (h, l, n, h);
	// now h = sigmoid(Transpose(w).x)

	// computing u_t
	transpose ARG(l, c) (u, l, c, u_t);
	// now u_t = Transpose(u)

	// computing y
	matrixMultiply ARG(c, n) (u_t, h, c, l, n, y);
	// now y = u_t.h

	// freeing memory on device
	cudaFree(w_t);
	cudaFree(h);
	cudaFree(u_t);
}

// reads 'matrix' : n->k from 'file'
// first like contains n k
// the further lines contain n x k floats
// we assume matrix == nullptr
bool read_matrix_from_file(std::string const &file, float* &matrix, size_t &n, size_t &k) {
	assert(matrix == nullptr);
	std::string line;
	std::ifstream input(file);
	size_t i{ 0 };
	float temp{ 0.0f };
	if (input.is_open()) {
		std::getline(input, line);
		std::istringstream in(line);
		in >> n;
		in >> k;
		matrix = new float[n * k];
		while (std::getline(input, line)){
			std::istringstream in(line);
			while (in >> temp)
				matrix[i++] = temp;
		}
		input.close();
	}
	else {
		std::cerr << "Unable to open file " << file << "\n";
		return false;
	}
	return true;
}

// writes 'matrix' : n->k into 'file'
// first like contains n k
// the further lines contain n x k floats
// we assume matrix != nullptr
bool write_matrix_into_file(std::string const &file, float* const matrix, size_t const &n, size_t const &k) {
	assert(matrix != nullptr);
	std::string line;
	std::ofstream output(file);
	size_t i{ 0 };
	size_t j{ 0 };
	if (output.is_open()) {
		output << n << " " << k << std::endl;
		for (j = 0; j < n; ++j){
			for (i = 0; i < (k - 1); ++i)
				output << matrix[ID(j, i, k)] << " ";
			output << matrix[ID(j, (k - 1), k)] << std::endl;
		}
		output.close();
	}
	else {
		std::cerr << "Unable to open file " << file << "\n";
		return false;
	}
	return true;
}

// main
int main(){
	size_t d, l, c, n;
	float *x{ nullptr }, *d_x{ nullptr }, *t{ nullptr }, *d_t{ nullptr }, *w{ nullptr };
	float *d_w{ nullptr }, *u{ nullptr }, *d_u{ nullptr }, *y{ nullptr }, *d_y{ nullptr };

	// setting up constants
	// --------------------- 
	// rho : initial learning rate
	float rho{ 1.0f };
	// number_of_iterations : iterations performed while learning
	size_t number_of_iterations{ 1 };

	// reading input
	read_matrix_from_file("x.txt", x, d, n);
	assert(x != nullptr);

	// reading weights of first layer
	read_matrix_from_file("w.txt", w, d, l);
	assert(w != nullptr);

	// reading weights of second layer
	read_matrix_from_file("u.txt", u, l, c);
	assert(u != nullptr);

	// memory allocation on the device
	cudaMalloc(&d_x, sizeof(float) * d * n);
	cudaMalloc(&d_w, sizeof(float) * d * l);
	cudaMalloc(&d_u, sizeof(float) * l * c);

	// copy x, w, u to the device
	cudaMemcpy(d_x, x, sizeof(float) * d * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, w, sizeof(float) * d * l, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, u, sizeof(float) * l * c, cudaMemcpyHostToDevice);

	delete[] x;

	//                                             //
	// at this point, the network is on the device //
	//                                             //
	
	if (!(std::ifstream("y.txt").good())){

		// if y.txt does not exist, then 
		// we are going to train the network

		read_matrix_from_file("t.txt", t, c, n);
		
		cudaMalloc(&d_t, sizeof(float) * c * n);

		cudaMemcpy(d_t, t, sizeof(float) * c * n, cudaMemcpyHostToDevice);
		
		train(d, l, c, n, d_x, d_t, d_w, d_u, rho, number_of_iterations);
		
		cudaMemcpy(w, d_w, sizeof(float) * d * l, cudaMemcpyDeviceToHost);
		cudaMemcpy(u, d_u, sizeof(float) * l * c, cudaMemcpyDeviceToHost);
		
		write_matrix_into_file("w.txt", w, d, l);
		write_matrix_into_file("u.txt", u, l, c);

		cudaFree(d_t);

		delete[] t;

		delete[] w;

		delete[] u;

	} else {

		// y.txt exists, so we are going to
		// process the network on the inputs
		// we will not need either w or u

		delete[] w;

		delete[] u;

		cudaMalloc(&d_y, sizeof(float) * c * n);

		compute (d, l, c, n, d_x, d_w, d_u, d_y);

		y = new float[c * n];

		cudaMemcpy(y, d_y, sizeof(float) * c * n, cudaMemcpyDeviceToHost);

		write_matrix_into_file("y.txt", y, c, n);
		
		cudaFree(d_y);

		delete[] y;

	}

	// freeing memory on device
	cudaFree(d_x);
	
	cudaFree(d_w);
	
	cudaFree(d_u);
	
	return 0;
}
