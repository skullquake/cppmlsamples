#include"./perceptron.hpp"
#include<algorithm>
#include<iostream>
#include<random>
#include<numeric>
#include<cmath>
static std::random_device rd;
static std::mt19937 mt(rd());
static std::uniform_real_distribution<>dist(-1,1);
static double frand(){
	return dist(mt);
}
namespace ml{
	Perceptron::Perceptron(int inputsP,double biasP):bias(biasP){
		weights.resize(inputsP+1);
		std::generate(weights.begin(),weights.end(),frand);
	}
	double Perceptron::run(std::vector<double>xP){
		xP.push_back(bias);
		double sum=std::inner_product(xP.begin(),xP.end(),weights.begin(),(double)0.0);
		return sigmoid(sum);
	}
	void Perceptron::set_weights(std::vector<double>w_initP){
		weights=w_initP;
	}
	double Perceptron::sigmoid(double xP){
		return 1.0/(1.0+exp(-xP));
	}
}
