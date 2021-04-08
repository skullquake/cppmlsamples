#ifndef hbbbc14f8989a11ebb3fb4389181cfe02
#define hbbbc14f8989a11ebb3fb4389181cfe02
#include"./perceptron.hpp"
#include<vector>
namespace ml{
	class MultilayerPerceptron{
		std::vector<int>layers;
		double bias;
		double eta;
		std::vector<std::vector<Perceptron>>network;
		std::vector<std::vector<double>>values;
		std::vector<std::vector<double>>d;
		public:
			MultilayerPerceptron(std::vector<int>layersP,double biasP=1.0,double eta=0.5);
			void set_weights(std::vector<std::vector<std::vector<double>>>w_initP);
			void print_weights();
			std::vector<double>run(std::vector<double>xP);
			double hp(std::vector<double>xP,std::vector<double>yP);
	};
}
#endif
