#ifndef h51849d62987c11eb96db9f67ad32d06f
#define h51849d62987c11eb96db9f67ad32d06f
#include<vector>
namespace ml{
	class MultilayerPerceptron;//fwddecl for friend
	class Perceptron{
		double bias;
		std::vector<double>weights;
		public:
			Perceptron(int inputsP,double biasP=1.0);
			double run(std::vector<double>xP);
			void set_weights(std::vector<double>w_initP);
			double sigmoid(double xP);
			friend class MultilayerPerceptron;
	};
}
#endif
