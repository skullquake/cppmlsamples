#ifndef h51849d62987c11eb96db9f67ad32d06f
#define h51849d62987c11eb96db9f67ad32d06f
#include<vector>
namespace ml{
	class Perceptron{
		double bias;
		std::vector<double>weights;
		public:
			Perceptron(int inputsP,double biasP=1.0);
			//execute
			double run(std::vector<double>xP);
			//setup weights
			void set_weights(std::vector<double>w_initP);
			//sigmoid activation function
			double sigmoid(double xP);
	};
}
#endif
