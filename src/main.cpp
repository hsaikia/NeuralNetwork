/*
** author : Himangshu Saikia
** email : saikia@kth.se
*/

#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

void testXOR() {
	std::vector<size_t> topo = { 2, 4, 1 };
	std::vector<double> eta = { 0.15, 0.15, 0.15 };
	std::vector<double> alpha = { 0.2, 0.2, 0.2 };
	std::vector<ActivationFunction> actFuns = { TANH, TANH, TANH };
	NeuralNetwork NN;
	NN.init(topo, eta, alpha, actFuns);

	struct XOR {

		XOR() {

		}

		XOR(double x, double y, double r) {
			inp.push_back(x);
			inp.push_back(y);
			out.push_back(r);
		}

		std::vector<double> inp;
		std::vector<double> out;
	};

	std::vector<XOR> all;
	all.push_back(XOR(0, 0, 1));
	all.push_back(XOR(0, 1, 0));
	all.push_back(XOR(1, 0, 0));
	all.push_back(XOR(1, 1, 1));

	//int count = 1000;

	//int R = 3;
	int run = 1;
	//while (count--) {
	while (true) {
		auto R = rand() % 4;
		//R = (R + 1) % 4;

		std::cout << "\nRun : " << run;
		run++;
		std::cout << "\nInput : " << all[R].inp[0] << ", " << all[R].inp[1] << "\n";
		std::cout << "Expected Output : " << all[R].out[0] << "\n";

		NN.feedForward(all[R].inp);

		NN.backProp(all[R].out);

		std::vector<double> res;
		NN.getResults(res);

		std::cout << "\nResult : " << res[0] << "\n";
		std::cout << "\nError : " << NN.getError() << "\n";

		//NN.writeNNToFile("xorTrain.txt");

		getchar();
	}

}

void trainLine() {
	std::vector<size_t> topo = { 2, 2 };
	std::vector<double> eta = { 0.2, 0.2 };
	std::vector<double> alpha = { 0.5, 0.5 };
	std::vector<ActivationFunction> actFuns = { ELU, ELU };

	NeuralNetwork NN;
	NN.init(topo, eta, alpha, actFuns);

	int iter = 0;

	while (true) {

		iter++;

		//sample from line y = 4x - 3;

		double x = rand() / double(RAND_MAX);
		double y = 4 * x - 3;

		std::cout << "\nInput : " << x << ", " << y << "\n";
		std::cout << "Expected Output :  (0.4, 0.3) \n";

		std::vector<double> input = { x, y };
		std::vector<double> output = { 0.4, 0.3 };

		NN.feedForward(input);

		NN.backProp(output);

		std::vector<double> res;
		NN.getResults(res);

		std::cout << "\nResult : " << res[0] << ", " << res[1];
		std::cout << "\nError : " << NN.getError() << "\n";

		if (NN.getError() < 1e-6) {
			std::cout << "\nNN converged in " << iter << " iterations!\n";
			NN.writeNNToFile("testline.nn");
			break;
		}

		//getchar();
	}
}


void testLine() {
	
	NeuralNetwork NN;
	NN.readNNFromFile("testline.nn");

	NN.writeNNToFile("copytestline.nn");

	int iter = 0;

	while (true) {
	
		iter++;

		//sample from line y = 4x - 3;

		double x = rand() / double(RAND_MAX);
		double y = 4 * x - 3;

		std::cout << "\nInput : " << x << ", " << y << "\n";
		std::cout << "Expected Output :  (0.4, 0.3) \n";

		std::vector<double> input = { x, y };
		std::vector<double> output = { 0.4, 0.3};

		NN.feedForward(input);

		//NN.backProp(output);

		std::vector<double> res;
		NN.getResults(res);

		std::cout << "\nResult : " << res[0] << ", " << res[1];
		std::cout << "\nError : " << NN.getError() << "\n";

		//if (NN.getError() < 1e-5) {
		//	std::cout << "\nNN converged in " << iter << " iterations!\n";
		//	break;
		//}

		getchar();
	}
}

void testEllipse() {

	//testing x^2 / a^2 + y^2 / b^2 = 1

	double a = 0.3;
	double b = 0.8;

	std::vector<size_t> topo = { 2, 3, 2 };
	std::vector<double> eta = { 0.15, 0.15, 0.15 };
	std::vector<double> alpha = { 0.5, 0.5, 0.5 };
	std::vector<ActivationFunction> actFuns = { ELU, ELU, ELU };

	NeuralNetwork NN;
	NN.init(topo, eta, alpha, actFuns);

	while (true) {

		//sample from line y = 4x - 3;

		double x = a * (2 * (rand() / double(RAND_MAX)) - 1.0);

		double pn = rand() % 2;

		double y = b * sqrt(1.0 - x * x / (a * a)) ;

		y *= pn == 0 ? 1 : -1;

		std::cout << "\nInput : " << x << ", " << y << "\n";
		std::cout << "Expected Output :  " << a << "," << b <<  " \n";

		std::vector<double> input = { x, y };
		std::vector<double> output = { a, b };

		NN.feedForward(input);

		NN.backProp(output);

		std::vector<double> res;
		NN.getResults(res);

		std::cout << "\nResult : " << res[0] << ", " << res[1];
		std::cout << "\nError : " << NN.getError() << "\n";

		getchar();
	}
}

void testQuadratic() {
	//testing y = ax^2 + bx + c

	double a = 0.3;
	double b = 0.8;
	double c = 0.5;

	std::vector<size_t> topo = { 2, 4, 3 };
	std::vector<double> eta = { 0.15, 0.15, 0.15 };
	std::vector<double> alpha = { 0.5, 0.5, 0.5 };
	std::vector<ActivationFunction> actFuns = { ELU, ELU, ELU };

	NeuralNetwork NN;
	NN.init(topo, eta, alpha, actFuns);

	while (true) {

		//sample from line y = 4x - 3;

		double x = rand() / double(RAND_MAX);

		double y = a * x * x + b * x + c;

		std::cout << "\nInput : " << x << ", " << y << "\n";
		std::cout << "Expected Output :  " << a << "," << b << "," << c << " \n";

		std::vector<double> input = { x, y };
		std::vector<double> output = { a, b, c };

		NN.feedForward(input);

		NN.backProp(output);

		std::vector<double> res;
		NN.getResults(res);

		std::cout << "\nResult : " << res[0] << ", " << res[1] << ", " << res[2];
		std::cout << "\nError : " << NN.getError() << "\n";

		getchar();
	}
}

int main() {
	testXOR();
	//trainLine();
	//getchar();
	//testLine();
	//getchar();
	return 0;
}