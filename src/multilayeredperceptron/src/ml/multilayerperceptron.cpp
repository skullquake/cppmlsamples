#include"./multilayerperceptron.hpp"
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
	MultilayerPerceptron::MultilayerPerceptron(std::vector<int>layersP,double biasP,double etaP):layers(layersP),bias(biasP),eta(etaP){
		for(int i=0;i<layersP.size();i++){
			values.push_back(std::vector<double>(layersP[i],0.0));
			network.push_back(std::vector<Perceptron>());
			if(i>0)/*skip input*/for(int j=0;j<layersP[i];j++)
				network[i].push_back(Perceptron(layersP[i-1],biasP));
		}
	}
	void MultilayerPerceptron::set_weights(std::vector<std::vector<std::vector<double>>>w_initP){
		//write all weights into the neural network
		//w_init is a vector of vectors vectors of doubles
		for(int i=0;i<w_initP.size();i++){
			for(int j=0;j<w_initP[i].size();j++){
				network[i+1][j].set_weights(w_initP[i][j]);
			}
		}
	}
	void MultilayerPerceptron::print_weights(){
		std::cout<<std::endl;
		for(int i=1;i<network.size();i++){
			for(int j=0;j<layers[i];j++){
				std::cout<<"Layer "<<i+1<<" Neuron "<<j<<" : ";
				for(const auto&it:network[i][j].weights)std::cout<<it<<" ";
				std::cout<<std::endl;
			}
		}
	}
	std::vector<double>MultilayerPerceptron::run(std::vector<double>xP){
		values[0]=xP;
		for(int i=1;i<network.size();i++)
			for(int j=0;j<layers[i];j++)
				values[i][j]=network[i][j].run(values[i-1]);
		return values.back();
	}
	double MultilayerPerceptron::hp(std::vector<double>xP,std::vector<double>yP){
	}
}
