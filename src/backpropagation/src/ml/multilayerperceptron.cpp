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
			d.push_back(std::vector<double>(layersP[i],0.0));
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
	double MultilayerPerceptron::bp(std::vector<double>xP,std::vector<double>yP){
		//1. feed sample to network
		std::vector<double>outputs=run(xP);
		//2. calculate mean squared error
		std::vector<double>error;
		double mse=0.0;
		for(int i=0;i<yP.size();i++){
			error.push_back(yP[i]-outputs[i]);
			mse+=error[i]*error[i];
		}
		mse/=layers.back();
		//3. calculate output error terms
		for(int i=0;i<outputs.size();i++){
			d.back()[i]=outputs[i]*(1-outputs[i])*(error[i]);
		}
		//4. calculate error term of each unit on each layer
		for(int i=network.size()-2;i>0;i--)
			for(int h=0;h<network[i].size();h++){
				double fwd_error=0.0;
				for(int k=0;k<layers[i+1];k++)
					fwd_error+=network[i+1][k].weights[h]*d[i+1][k];
				d[i][h]=values[i][h]*(1-values[i][h])*fwd_error;
			}
		//5&6. calculate the deltas and update the weights
		for(int i=1;i<network.size();i++)
			for(int j=0;j<layers[i];j++)
				for(int k=0;k<layers[i-1]+1;k++){
					double delta;
					if(k==layers[i-1])
						delta=eta*d[i][j]*bias;
					else
						delta=eta*d[i][j]*values[i-1][k];
					network[i][j].weights[k]+=delta;
				}
		return mse;
	}
}
