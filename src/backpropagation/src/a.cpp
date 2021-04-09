#include<iostream>
#include"ml/perceptron.hpp"
#include"ml/multilayerperceptron.hpp"
int main(void){
	{//AND
		ml::Perceptron p(2);
		p.set_weights({10,10,/*bias*/-15});
		std::cout<<"AND Gate:"<<std::endl;
		std::cout<<p.run({0,0})<<std::endl;//0	<-	3.05902e-07
		std::cout<<p.run({0,1})<<std::endl;//0	<-	0.00669285
		std::cout<<p.run({1,0})<<std::endl;//0	<-	0.00669285
		std::cout<<p.run({1,1})<<std::endl;//1	<-	0.993307
	}
	{//OR
		ml::Perceptron p(2);
		p.set_weights({15,15,/*bias*/-10});
		std::cout<<"OR Gate:"<<std::endl;
		std::cout<<p.run({0,0})<<std::endl;//0	<-	4.53979e-05
		std::cout<<p.run({0,1})<<std::endl;//1	<-	0.993307
		std::cout<<p.run({1,0})<<std::endl;//1	<-	0.993307
		std::cout<<p.run({1,1})<<std::endl;//1	<-	1
	}
	{//NAND
		ml::Perceptron p(2);
		p.set_weights({-10,-10,/*bias*/15});
		std::cout<<"NAND Gate:"<<std::endl;
		std::cout<<p.run({0,0})<<std::endl;//1	<-	1
		std::cout<<p.run({0,1})<<std::endl;//1	<-	0.993307
		std::cout<<p.run({1,0})<<std::endl;//1	<-	0.993307
		std::cout<<p.run({1,1})<<std::endl;//0	<-	0.00669285
	}
	{//XOR - composite
		std::vector<double>vor{15,15,/*bias*/-10};
		std::vector<double>vand{10,10,/*bias*/-15};
		std::vector<double>vnand{-10,-10,/*bias*/15};
		ml::Perceptron por(2);
		ml::Perceptron pand(2);
		ml::Perceptron pnand(2);
		por.set_weights(vor);
		pand.set_weights(vand);
		pnand.set_weights(vnand);
		auto run=[&pnand,&por,&pand](const double&a,const double&b){
			return pand.run({pnand.run({a,b}),por.run({a,b})});
		};
		std::cout<<"XOR Gate:"<<std::endl;
		std::cout<<run(0,0)<<std::endl;//0	<-	.00669585
		std::cout<<run(0,1)<<std::endl;//1	<-	0.992356
		std::cout<<run(1,0)<<std::endl;//1	<-	0.992356
		std::cout<<run(1,1)<<std::endl;//0	<-	0.00715281
	}
	{//XOR - multilayer
		std::cout<<"XOR Gate[multilayer]:"<<std::endl;
		auto mlp=ml::MultilayerPerceptron({2,2,1});
		mlp.set_weights({{{-10,-10,15},{15,15,-10}},{{10,10,-15}}});
		mlp.print_weights();
		std::cout<<mlp.run({0,0})[0]<<std::endl;//0	<-	0.00669585
		std::cout<<mlp.run({0,1})[0]<<std::endl;//1	<-	0.992356
		std::cout<<mlp.run({1,0})[0]<<std::endl;//1	<-	0.992356
		std::cout<<mlp.run({1,1})[0]<<std::endl;//0	<-	0.00715281
	}
	{//backpropagation - XOR
		double MSE;
		auto mlp=ml::MultilayerPerceptron({2,2,1});
		for(int i=0;i<4096;i++){
			MSE=0.0;
			MSE+=mlp.bp({0,0},{0});
			MSE+=mlp.bp({0,1},{1});
			MSE+=mlp.bp({1,0},{1});
			MSE+=mlp.bp({1,1},{0});
			if(i%128==0)
				std::cout<<"MSE: "<<MSE<<std::endl;
		}
		std::cout<<"Trained weights:"<<std::endl;
		mlp.print_weights();
		std::cout<<"00:"<<mlp.run({0,0})[0]<<std::endl;//0	<-	
		std::cout<<"01:"<<mlp.run({0,1})[0]<<std::endl;//1	<-	
		std::cout<<"10:"<<mlp.run({1,0})[0]<<std::endl;//1	<-	
		std::cout<<"11:"<<mlp.run({1,1})[0]<<std::endl;//0	<-	

	}
}
